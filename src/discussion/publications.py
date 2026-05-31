"""Отбор публикаций, связанных с дискуссионным событием"""
from __future__ import annotations


from ..pipeline.common import *

_PUBLICATION_BAD_TICKERS = {"", "NAN", "NONE", "NULL", "[]", "-", "__DISCUSSION__"}

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_ingestion.preparation import parse_ticker_mentions
    from ..nlp.ticker_extraction import _accept_ticker_candidate_for_analytics
    from ..pipeline.configuration import _clean_author_value, _clean_column_name, _first_present, is_service_author, service_author_label

def _publication_clean_ticker(value: Any) -> str | None:
    """Нормализует тикер для поиска публикаций
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        нормализованный тикер публикации
    """
    token = str(value).strip().upper().replace("$", "")
    token = token.strip(" [](){}'\"\n\r\t")
    if not token or token in _PUBLICATION_BAD_TICKERS:
        return None
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", token):
        return None
    return token


def _publication_tickers_from_value(value: Any) -> set[str]:
    """Извлекает тикеры из значения публикации
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        множество тикеров из значения публикации
    """
    out: set[str] = set()
    if value is None:
        return out
    try:
        if pd.isna(value):
            return out
    except Exception:
        pass
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        raw = str(value).strip()
        if not raw or raw.upper() in _PUBLICATION_BAD_TICKERS:
            return out
        items = None
        if raw.startswith("[") or raw.startswith("{"):
            import ast
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(raw)
                    items = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]
                    break
                except Exception:
                    items = None
        if items is None:
            cleaned = raw.replace(";", ",").replace("|", ",").replace("[", ",").replace("]", ",")
            items = [x for x in cleaned.split(",") if x.strip()]
    for item in items:
        if isinstance(item, dict):
            for key in ("ticker", "symbol", "value"):
                token = _publication_clean_ticker(item.get(key))
                if token:
                    out.add(token)
                    break
        else:
            token = _publication_clean_ticker(item)
            if token:
                out.add(token)
    return out


def _publication_row_tickers(row: pd.Series | dict[str, Any]) -> set[str]:
    """Извлекает тикеры из строки публикации
    
    Args:
        row: строка таблицы с сообщением, инструментом или событием
    
    Returns:
        множество тикеров из строки публикации
    """
    out: set[str] = set()
    try:
        participation = parse_ticker_mentions(row.get("ticker_participation_mentions", "[]"))
        out |= {str(m.get("ticker", "")).upper().strip() for m in participation if _accept_ticker_candidate_for_analytics(m)}
    except Exception:
        pass
    try:
        mentions = parse_ticker_mentions(row.get("ticker_mentions", "[]"))
        out |= {str(m.get("ticker", "")).upper().strip() for m in mentions if _accept_ticker_candidate_for_analytics(m)}
    except Exception:
        pass
    if out:
        return {x for x in out if x}
    for key in ["discussion_tickers", "publication_tickers", "tickers", "ticker", "тикеры", "mentions"]:
        try:
            out |= _publication_tickers_from_value(row.get(key))
        except Exception:
            pass
    return out


def _publication_text_has_ticker(row: pd.Series | dict[str, Any], ticker: str) -> bool:
    """Проверяет наличие тикера в тексте публикации
    
    Args:
        row: строка таблицы с сообщением, инструментом или событием
        ticker: биржевой идентификатор финансового инструмента
    
    Returns:
        bool: признак наличия тикера в тексте публикации
    """
    ticker = str(ticker).upper().strip()
    text = " ".join(str(row.get(k, "")) for k in ["text", "text_clean", "title", "selftext", "body", "content"]).upper()
    return bool(ticker and re.search(rf"(?<![A-Z0-9])\$?{re.escape(ticker)}(?![A-Z0-9])", text))

def _prepare_publications_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Готовит таблицу публикаций для ранжирования и отображения
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        подготовленная таблица публикаций
    """
    df = df.copy()
    df.columns = [_clean_column_name(c) for c in df.columns]
    if "author" not in df.columns:
        col = _first_present(df, ["Author", "author_name", "username", "user", "screen_name", "source", "channel", "subreddit"])
        df["author"] = df[col].map(_clean_author_value) if col else "unknown"
    else:
        df["author"] = df["author"].map(_clean_author_value)
    df["author_is_service"] = df["author"].map(is_service_author)
    df["author_display"] = df["author"].map(service_author_label)
    if "text" not in df.columns:
        text_col = _first_present(df, ["text_clean", "selftext", "body", "content", "message", "title"])
        df["text"] = df[text_col].fillna("").astype(str) if text_col else ""
    if "timestamp" not in df.columns:
        time_col = _first_present(df, ["timestamp", "created_utc", "created_at", "date", "datetime", "published_at"])
        if time_col:
            df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    return df



def _rank_publications_frame(part: pd.DataFrame, tickers: list[str] | None = None, top_n: int = 10, source_period: str = "шаг дискретизации") -> pd.DataFrame:
    """Ранжирует публикации по релевантности выбранному событию
    
    Args:
        part: фрагмент таблицы сообщений внутри шага дискретизации
        tickers: набор биржевых идентификаторов финансовых инструментов
        top_n: число наиболее значимых записей для отбора
        source_period: описание исходного периода данных
    
    Returns:
        таблица публикаций с оценкой релевантности
    """
    if part is None or pd.DataFrame(part).empty:
        return pd.DataFrame()
    part = pd.DataFrame(part).copy()
    requested = {t for t in (_publication_clean_ticker(x) for x in (tickers or [])) if t}
    part["publication_tickers"] = part.apply(lambda r: ",".join(sorted(_publication_row_tickers(r))), axis=1)
    if requested:
        strict_mask = part.apply(lambda r: bool(_publication_row_tickers(r) & requested), axis=1)
        filtered = part[strict_mask].copy()
        if filtered.empty:
            soft_mask = part.apply(lambda r: any(_publication_text_has_ticker(r, t) for t in requested), axis=1)
            filtered = part[soft_mask].copy()
            if not filtered.empty:
                filtered["publication_tickers"] = filtered.apply(
                    lambda r: ",".join(sorted((_publication_row_tickers(r) | {t for t in requested if _publication_text_has_ticker(r, t)}))),
                    axis=1,
                )
        if not filtered.empty:
            part = filtered
    part["abs_sentiment"] = pd.to_numeric(part.get("sentiment", 0.0), errors="coerce").fillna(0.0).abs()
    part["influence"] = pd.to_numeric(part.get("influence", 0.0), errors="coerce").fillna(0.0)
    def ticker_weight(row):
        """Возвращает вес строки при построении тематического профиля тикера
        
        Args:
            row: строка таблицы с сообщением, инструментом или событием
        
        Returns:
            числовой вес строки для тематического профиля тикера
        """
        mentions = parse_ticker_mentions(row.get("ticker_participation_mentions")) if "ticker_participation_mentions" in row else []
        mentions = [m for m in mentions if _accept_ticker_candidate_for_analytics(m)]
        if not mentions and "ticker_mentions" in row:
            mentions = [m for m in parse_ticker_mentions(row.get("ticker_mentions")) if _accept_ticker_candidate_for_analytics(m)]
        if mentions:
            return sum(float(m.get("confidence", 0.0) or 0.0) for m in mentions)
        return min(1.0, float(len(_publication_row_tickers(row))))
    part["ticker_weight"] = part.apply(ticker_weight, axis=1)
    rank_cols = []
    for source_col, rank_col in [("ticker_weight", "ticker_weight_rank"), ("influence", "influence_rank"), ("abs_sentiment", "sentiment_rank")]:
        values = pd.to_numeric(part.get(source_col, 0.0), errors="coerce").fillna(0.0)
        part[rank_col] = values.rank(pct=True, method="average")
        rank_cols.append(rank_col)
    part["impact_score"] = part[rank_cols].max(axis=1)
    part["source_period"] = source_period
    cols = [c for c in ["timestamp", "source_period", "author", "author_display", "text", "topic", "sentiment", "influence", "tickers", "direct_tickers", "inherited_tickers", "discussion_tickers", "publication_tickers", "ticker_mentions", "ticker_participation_mentions", "impact_score"] if c in part.columns]
    out = part.sort_values("impact_score", ascending=False).head(int(top_n))[cols].copy()
    if "text" in out.columns:
        out["text"] = out["text"].astype(str).str.slice(0, 800)
    return out.reset_index(drop=True)


def _top_publications_from_range_frame(df: pd.DataFrame, start_time: Any, end_time: Any, tickers: list[str] | None = None, top_n: int = 10, source_period: str = "контекст") -> pd.DataFrame:
    """Выбирает наиболее релевантные публикации из диапазона сообщений
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
        start_time: временная метка начала анализируемого шага дискретизации
        end_time: временная метка конца интервала
        tickers: набор биржевых идентификаторов финансовых инструментов
        top_n: число наиболее значимых записей для отбора
        source_period: описание исходного периода данных
    
    Returns:
        выбранное значение или подтаблица: наиболее релевантные публикации из диапазона сообщений
    """
    if df is None or pd.DataFrame(df).empty:
        return pd.DataFrame()
    df = _prepare_publications_frame(pd.DataFrame(df))
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    start_ts = pd.Timestamp(pd.to_datetime(start_time, errors="coerce")).tz_localize(None)
    end_ts = pd.Timestamp(pd.to_datetime(end_time, errors="coerce")).tz_localize(None)
    part = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    return _rank_publications_frame(part, tickers=tickers, top_n=top_n, source_period=source_period)

def _top_publications_from_frame(df: pd.DataFrame, start_time: Any, window_hours: int, tickers: list[str] | None = None, top_n: int = 10) -> pd.DataFrame:
    """Выбирает наиболее релевантные публикации из подготовленной таблицы
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
        start_time: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
        tickers: набор биржевых идентификаторов финансовых инструментов
        top_n: число наиболее значимых записей для отбора
    
    Returns:
        выбранное значение или подтаблица: наиболее релевантные публикации из подготовленной таблицы
    """
    if df is None or pd.DataFrame(df).empty:
        return pd.DataFrame()
    df = _prepare_publications_frame(pd.DataFrame(df))
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    start_ts = pd.Timestamp(pd.to_datetime(start_time, errors="coerce")).tz_localize(None)
    end_ts = start_ts + pd.Timedelta(hours=int(window_hours))
    part = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    if part.empty:
        return pd.DataFrame()

    return _rank_publications_frame(part, tickers=tickers, top_n=top_n, source_period="текущий шаг дискретизации")


def top_publications_for_window(messages_path: str | Path, start_time: Any, window_hours: int, tickers: list[str] | None = None, top_n: int = 10) -> pd.DataFrame:
    """Возвращает публикации, объясняющие выбранный шаг дискретизации
    
    Args:
        messages_path: путь к таблице обработанных сообщений
        start_time: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
        tickers: набор биржевых идентификаторов финансовых инструментов
        top_n: число наиболее значимых записей для отбора
    
    Returns:
        таблица наиболее релевантных публикаций шага дискретизации
    """
    path = Path(messages_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return _top_publications_from_frame(df, start_time, window_hours, tickers=tickers, top_n=top_n)
