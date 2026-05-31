"""Рыночная ассоциативная проверка и ценовые панели"""
from __future__ import annotations


from ..pipeline.common import *
from ..pipeline.configuration import AppConfig, _emit_progress
from ..indicators.discussion_indicators import DISCUSSION_LEVEL_TICKER

def _price_col(df: pd.DataFrame) -> str:
    """Находит колонку цены в рыночной таблице
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        название колонки цены или None
    """
    for col in ["Adj Close", "Close"]:
        if col in df.columns:
            return col
    raise ValueError("Market data has no Close/Adj Close column")


def _to_naive_timestamp(value: Any) -> pd.Timestamp:
    """Преобразует временную метку к naive timestamp
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        преобразованное значение: временную метку к naive timestamp
    """
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts.tz_localize(None) if getattr(ts, "tz", None) is not None else ts


def _fetch_market_history(ticker: str, start: pd.Timestamp, end: pd.Timestamp, interval: str = "1d") -> pd.DataFrame:
    """Загружает рыночную историю для тикера и ориентира
    
    Args:
        ticker: биржевой идентификатор финансового инструмента
        start: начало интервала или начальный индекс
        end: конец интервала или конечный индекс
        interval: пара границ анализируемого интервала
    
    Returns:
        таблица рыночной истории инструмента и ориентира
    """
    try:
        import yfinance as yf
    except Exception as exc:
        raise RuntimeError("yfinance is required for market projection. Run: pip install yfinance") from exc
    interval = str(interval or "1d")
    start = _to_naive_timestamp(start)
    end = _to_naive_timestamp(end)
    if pd.isna(start) or pd.isna(end) or end <= start:
        return pd.DataFrame()
    data = yf.download(
        ticker,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
        prepost=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty:
        return pd.DataFrame()
    data = data.reset_index()
    time_col = "Datetime" if "Datetime" in data.columns else "Date" if "Date" in data.columns else data.columns[0]
    ts = pd.to_datetime(data[time_col], errors="coerce")
    try:
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert(None)
        else:
            ts = ts.dt.tz_localize(None)
    except Exception:
        try:
            ts = ts.dt.tz_convert(None)
        except Exception:
            pass
    data["Date"] = ts
    data = data[(data["Date"] >= start) & (data["Date"] <= end)].copy()
    data["ticker"] = ticker
    data["interval"] = interval
    return data


def _normalize_market_interval(value: str | None) -> str:
    """Нормализует границы рыночного интервала отображения
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        нормализованное значение: границы рыночного интервала отображения
    """
    text = str(value or "1d").strip().lower()
    aliases = {
        "1 день": "1d",
        "день": "1d",
        "daily": "1d",
        "1d": "1d",
        "1 час": "1h",
        "час": "1h",
        "hourly": "1h",
        "1h": "1h",
        "60m": "60m",
        "30 минут": "30m",
        "30m": "30m",
        "15 минут": "15m",
        "15m": "15m",
    }
    return aliases.get(text, text if text in {"1d", "1h", "60m", "30m", "15m"} else "1d")


def _indicator_points_for_panel(indicators: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    """Готовит точки дискуссионных индикаторов для рыночной панели
    
    Args:
        indicators: таблица или набор дискуссионных индикаторов
        window_hours: длительность шага дискретизации в часах
    
    Returns:
        подготовленная структура: точки дискуссионных индикаторов для рыночной панели
    """
    if indicators is None or pd.DataFrame(indicators).empty:
        return pd.DataFrame()
    df = pd.DataFrame(indicators).copy()
    need = {"ticker", "start_time", "future_step", "forward_discussion_indicator"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    df = df[df["ticker"].astype(str) != DISCUSSION_LEVEL_TICKER].copy()
    if df.empty:
        return pd.DataFrame()
    df["future_step"] = pd.to_numeric(df["future_step"], errors="coerce")
    df["forward_discussion_indicator"] = pd.to_numeric(df["forward_discussion_indicator"], errors="coerce")
    if "combined_forward_discussion_indicator" in df.columns:
        df["combined_forward_discussion_indicator"] = pd.to_numeric(df["combined_forward_discussion_indicator"], errors="coerce")
        df["forward_discussion_indicator"] = df["combined_forward_discussion_indicator"].combine_first(df["forward_discussion_indicator"])
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna(subset=["future_step", "forward_discussion_indicator", "start_time"])
    if df.empty:
        return pd.DataFrame()
    # для визуального сравнения точка +k ставится в конец k-го смоделированного шага
    # это не прогноз цены
    # точка показывает момент доступности дискуссионного индикатора для сопоставления
    df["Date"] = df["start_time"] + pd.to_timedelta(df["future_step"].astype(int) * int(window_hours), unit="h")
    keep = ["ticker", "Date", "future_step", "forward_discussion_indicator"]
    for c in ["ticker_share", "ticker_sentiment", "ticker_avg_influence", "ticker_realized_nrmse"]:
        if c in df.columns:
            keep.append(c)
    return df[keep].copy()


def load_market_price_panel(
    cfg: AppConfig,
    indicators: pd.DataFrame,
    window_hours: int,
    top_n: int = 3,
    lookback_days: int = 7,
    market_index: str | None = None,
    progress=None,
    interval: str = "1d",
    display_start: Any | None = None,
    display_end: Any | None = None,
    after_days: int = 5,
) -> pd.DataFrame:
    """Загружает ценовую панель для визуального сопоставления с индикаторами
    
    Args:
        cfg: конфигурация проекта или приложения
        indicators: таблица или набор дискуссионных индикаторов
        window_hours: длительность шага дискретизации в часах
        top_n: число наиболее значимых записей для отбора
        lookback_days: глубина рыночной истории для интервала оценки в днях
        market_index: тикер рыночного ориентира для ассоциативной проверки
        progress: callback для передачи статуса выполнения
        interval: пара границ анализируемого интервала
        display_start: начало интервала отображения
        display_end: конец интервала отображения
        after_days: число дней после события для рыночной проверки
    
    Returns:
        таблица ценовой панели
    """
    if indicators is None or pd.DataFrame(indicators).empty:
        return pd.DataFrame()
    df = pd.DataFrame(indicators).copy()
    if "ticker" not in df or "start_time" not in df:
        return pd.DataFrame()
    df = df[df["ticker"].astype(str) != DISCUSSION_LEVEL_TICKER].copy()
    if df.empty:
        return pd.DataFrame()
    df["forward_discussion_indicator"] = pd.to_numeric(df.get("forward_discussion_indicator"), errors="coerce").fillna(0.0)
    if "combined_forward_discussion_indicator" in df.columns:
        df["combined_forward_discussion_indicator"] = pd.to_numeric(df.get("combined_forward_discussion_indicator"), errors="coerce").fillna(df["forward_discussion_indicator"])
    else:
        df["combined_forward_discussion_indicator"] = df["forward_discussion_indicator"]
    df["ticker_post_count"] = pd.to_numeric(df.get("ticker_post_count"), errors="coerce").fillna(0.0)
    df["ticker_activity_rate"] = pd.to_numeric(df.get("ticker_activity_rate"), errors="coerce").fillna(0.0)
    df["ticker_observed_support_pvalue"] = pd.to_numeric(df.get("ticker_observed_support_pvalue"), errors="coerce").fillna(1.0)
    candidate_df = df[(df["ticker_post_count"] > 0) & (df["ticker_activity_rate"] > 0)].copy()
    if candidate_df.empty:
        candidate_df = df[(df["forward_discussion_indicator"] >= 0.25) | (df.get("combined_forward_discussion_indicator", pd.Series(index=df.index, dtype=float)) >= 0.25)].copy()
    if candidate_df.empty:
        return pd.DataFrame()
    ranked = (
        candidate_df.groupby("ticker", as_index=False)
        .agg(
            forward_discussion_indicator=("forward_discussion_indicator", "max"),
            combined_forward_discussion_indicator=("combined_forward_discussion_indicator", "max"),
            ticker_post_count=("ticker_post_count", "max"),
            ticker_activity_rate=("ticker_activity_rate", "max"),
            ticker_observed_support_pvalue=("ticker_observed_support_pvalue", "min"),
        )
        .sort_values(["combined_forward_discussion_indicator", "forward_discussion_indicator", "ticker_post_count", "ticker_activity_rate"], ascending=[False, False, False, False])
    )
    top_tickers = ranked.head(int(top_n))["ticker"].astype(str).str.upper().tolist()
    if not top_tickers:
        return pd.DataFrame()
    start_ts = pd.to_datetime(df["start_time"].dropna().iloc[0], errors="coerce")
    if pd.isna(start_ts):
        return pd.DataFrame()
    start_ts = _to_naive_timestamp(start_ts)
    max_step = int(pd.to_numeric(df.get("future_step"), errors="coerce").max() or 1)
    context_steps = 0
    if "context_windows" in df.columns:
        context_values = pd.to_numeric(df["context_windows"], errors="coerce").dropna()
        if not context_values.empty:
            context_steps = max(0, int(context_values.max()))
    step_end = start_ts + pd.Timedelta(hours=int(window_hours))
    horizon_end = start_ts + pd.Timedelta(hours=int(window_hours) * max_step)
    comparison_end = max(step_end, horizon_end)
    context_start = start_ts - pd.Timedelta(hours=int(window_hours) * context_steps)
    interval = _normalize_market_interval(interval)
    start = _to_naive_timestamp(display_start) if display_start is not None else context_start
    end = _to_naive_timestamp(display_end) if display_end is not None else comparison_end
    if pd.isna(start) or pd.isna(end) or end <= start:
        return pd.DataFrame()
    market_index = str(market_index or cfg.get("market", "market_index", default="SPY")).upper()
    rows: list[pd.DataFrame] = []
    tickers = list(dict.fromkeys(top_tickers + [market_index]))
    for idx, ticker in enumerate(tickers, start=1):
        _emit_progress(progress, f"Загрузка ценового ряда: {ticker}", idx, len(tickers), stage="market_timeseries")
        try:
            hist = _fetch_market_history(ticker, start, end, interval=interval)
        except Exception:
            hist = pd.DataFrame()
        if hist.empty:
            continue
        price_col = _price_col(hist)
        cols = ["Date", price_col, "ticker"]
        if "Volume" in hist.columns:
            cols.append("Volume")
        part = hist[cols].copy().rename(columns={price_col: "price", "Volume": "volume"})
        part["ticker"] = ticker
        part["event_time"] = start_ts
        part["step_end_time"] = step_end
        part["context_start_time"] = context_start
        part["horizon_end_time"] = horizon_end
        part["comparison_end_time"] = comparison_end
        part["context_windows"] = context_steps
        part["display_start"] = start
        part["display_end"] = end
        part["market_interval"] = interval
        part["is_benchmark"] = ticker == market_index
        base_candidates = part[part["Date"] <= start_ts]
        base = float(base_candidates["price"].iloc[-1]) if not base_candidates.empty else float(part["price"].iloc[0])
        if base == 0 or np.isnan(base):
            valid = part["price"].replace(0, np.nan).dropna()
            base = float(valid.iloc[0]) if len(valid) else 1.0
        part["normalized_price"] = part["price"].astype(float) / base * 100.0
        part["period"] = np.where(part["Date"] < start_ts, "до события", "после события")
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
