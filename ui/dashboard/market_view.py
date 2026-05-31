"""Рыночная панель, интерпретация цен и выгрузка таблиц"""
from __future__ import annotations


from .common import *

from .controls import fmt, info_panel, metric_card, pct
from .summaries import _discussion_rows, _empty_ticker_summary, _forward_ticker_summary, _ticker_rows, _truthy_series

def render_price_interpretation(panel: pd.DataFrame, indicators: pd.DataFrame | None = None) -> None:
    """Показывает осторожную текстовую интерпретацию рыночной панели
    
    Args:
        panel: таблица рыночной панели для отображения
        indicators: таблица или набор дискуссионных индикаторов
    
    Returns:
        None
    """
    if panel is None or pd.DataFrame(panel).empty:
        return
    p = pd.DataFrame(panel).copy()
    p["Date"] = pd.to_datetime(p["Date"], errors="coerce")
    p["normalized_price"] = pd.to_numeric(p["normalized_price"], errors="coerce")
    event_time = pd.to_datetime(p["event_time"].dropna().iloc[0], errors="coerce") if "event_time" in p and p["event_time"].notna().any() else None
    forecast_end = pd.to_datetime(p["forecast_end_time"].dropna().iloc[0], errors="coerce") if "forecast_end_time" in p and p["forecast_end_time"].notna().any() else None
    if pd.isna(event_time) or pd.isna(forecast_end):
        return
    rows = []
    for ticker, part in p.groupby("ticker"):
        part = part.sort_values("Date")
        before = part[part["Date"] <= event_time]
        after = part[part["Date"] <= forecast_end]
        if before.empty or after.empty:
            continue
        base = float(before["normalized_price"].iloc[-1])
        end = float(after["normalized_price"].iloc[-1])
        rows.append({"ticker": ticker, "change_pct": (end / base - 1.0) if base else None})
    res = pd.DataFrame(rows)
    if res.empty:
        return
    market_rows = res[res["ticker"].astype(str).str.upper().isin({"SPY", "QQQ"})]
    benchmark_change = float(market_rows["change_pct"].iloc[0]) if not market_rows.empty else None
    invest = res[~res["ticker"].astype(str).str.upper().isin({"SPY", "QQQ"})].copy()
    if invest.empty:
        return
    invest = invest.sort_values("change_pct", ascending=False)
    best = invest.iloc[0]
    worst = invest.iloc[-1]
    bench_text = f" Относительно бенчмарка: {pct(benchmark_change, 2)}." if benchmark_change is not None else ""
    info_panel(
        "Что произошло с ценой после анализируемого шага",
        f"На выбранном горизонте сильнее всего вырос {best['ticker']} ({pct(best['change_pct'], 2)}), сильнее всего снизился {worst['ticker']} ({pct(worst['change_pct'], 2)}).{bench_text} Этот граф не доказывает причинность, но показывает, сопровождался ли дискуссионный индикатор фактическим движением рынка.",
        "blue",
    )

def plot_forward_validation(df: pd.DataFrame) -> None:
    """Строит график проверки смоделированной тематической динамики
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        None
    """
    if df is None or pd.DataFrame(df).empty or "actual_available" not in pd.DataFrame(df):
        return
    disc = _discussion_rows(df)
    plot = disc if not disc.empty else pd.DataFrame(df).drop_duplicates(subset=["future_step"])
    plot = plot[_truthy_series(plot["actual_available"])] if "actual_available" in plot else pd.DataFrame()
    if plot.empty or "realized_nrmse" not in plot:
        return
    plot = plot.copy()
    plot["realized_nrmse"] = pd.to_numeric(plot["realized_nrmse"], errors="coerce")
    plot = plot.dropna(subset=["realized_nrmse"])
    if plot.empty:
        return
    st.markdown("#### Проверка модельной симуляции дискуссии по доступным будущим данным")
    c1, c2, c3 = st.columns(3)
    mean_nrmse = float(plot["realized_nrmse"].mean())
    good_share = float((plot["realized_nrmse"] <= 15.0).mean())
    weak_share = float((plot["realized_nrmse"] > 35.0).mean())
    with c1: metric_card("Средний проверочный nRMSE", fmt(mean_nrmse, 2) + "%", "меньше = лучше", "green" if mean_nrmse <= 15.0 else "orange" if mean_nrmse <= 35.0 else "red")
    with c2: metric_card("Хороших совпадений", pct(good_share, 1), "nRMSE ≤ 15%", "green")
    with c3: metric_card("Слабых совпадений", pct(weak_share, 1), "nRMSE > 35%", "red" if weak_share > 0.3 else "orange")
    fig = px.line(
        plot.sort_values("future_step"),
        x="future_step",
        y="realized_nrmse",
        markers=True,
        title="Согласованность модельной и фактической динамики дискуссии",
        labels={"future_step": "следующий шаг (+N)", "realized_nrmse": "nRMSE, % по доступным данным"}
    )
    fig.add_hline(y=15.0, line_dash="dash", line_color="#16a34a", annotation_text="хорошее совпадение")
    fig.add_hline(y=35.0, line_dash="dash", line_color="#f59e0b", annotation_text="граница слабого совпадения")
    fig.update_traces(mode="lines+markers")
    fig.update_xaxes(dtick=1)
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    plotly_chart(fig)



def plot_ticker_validation(df: pd.DataFrame) -> None:
    """Строит график проверки динамики для выбранных финансовых инструментов
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        None
    """
    tick = _ticker_rows(df)
    if tick.empty or "ticker_realized_nrmse" not in tick.columns:
        return
    plot = tick.copy()
    if "ticker_has_topic_profile" in plot.columns:
        plot = plot[_truthy_series(plot["ticker_has_topic_profile"])]
    if plot.empty:
        st.info("Тикерная согласованность не отображается: для выбранных тикеров не восстановлен профиль тем. Общий анализ дискуссии доступен выше.")
        return
    plot["ticker_realized_nrmse"] = pd.to_numeric(plot["ticker_realized_nrmse"], errors="coerce")
    plot = plot.dropna(subset=["ticker_realized_nrmse"])
    if plot.empty:
        return
    top = _forward_ticker_summary(plot).head(6)
    if not top.empty and "ticker" in top.columns:
        keep = set(top["ticker"].astype(str))
        plot = plot[plot["ticker"].astype(str).isin(keep)]
    st.markdown("#### Тикерная согласованность по доступным будущим данным")
    st.caption("Это не копия общей nRMSE. Для каждого тикера ошибка считается только по темам, с которыми этот тикер был связан в выбранном шаге дискретизации. Если будущих данных нет, этот блок не отображается.")
    fig = px.line(
        plot.sort_values(["ticker", "future_step"]),
        x="future_step",
        y="ticker_realized_nrmse",
        color="ticker",
        markers=True,
        title="Согласованность модельной и фактической динамики по тикерам",
        labels={"future_step": "следующий шаг (+N)", "ticker_realized_nrmse": "nRMSE, % по темам тикера", "ticker": "тикер"}
    )
    fig.add_hline(y=15.0, line_dash="dash", line_color="#16a34a", annotation_text="хорошее совпадение")
    fig.add_hline(y=35.0, line_dash="dash", line_color="#f59e0b", annotation_text="слабое совпадение")
    fig.update_traces(mode="lines+markers")
    fig.update_xaxes(dtick=1)
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    plotly_chart(fig)


def _market_indicator_points(indicators: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    """Формирует точки дискуссионных индикаторов для наложения на рыночный график
    
    Args:
        indicators: таблица или набор дискуссионных индикаторов
        window_hours: длительность шага дискретизации в часах
    
    Returns:
        pd.DataFrame: таблица точек индикаторов для нанесения на рыночный график
    """
    if indicators is None or pd.DataFrame(indicators).empty:
        return pd.DataFrame()
    df = pd.DataFrame(indicators).copy()
    req = {"ticker", "start_time", "future_step", "forward_discussion_indicator"}
    if not req.issubset(df.columns):
        return pd.DataFrame()
    df = df[df["ticker"].astype(str) != "__DISCUSSION__"].copy()
    if df.empty:
        return pd.DataFrame()
    df["future_step"] = pd.to_numeric(df["future_step"], errors="coerce")
    df["forward_discussion_indicator"] = pd.to_numeric(df["forward_discussion_indicator"], errors="coerce")
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna(subset=["future_step", "forward_discussion_indicator", "start_time"])
    if df.empty:
        return pd.DataFrame()
    df["Date"] = df["start_time"] + pd.to_timedelta(df["future_step"].astype(int) * int(window_hours), unit="h")
    return df


def render_market_indicator_cards(indicators: pd.DataFrame | None, max_tickers: int = 4) -> None:
    """Показывает карточки дискуссионных событий рядом с рыночной панелью
    
    Args:
        indicators: таблица или набор дискуссионных индикаторов
        max_tickers: максимальное число финансовых инструментов для вывода
    
    Returns:
        None
    """
    summary = _forward_ticker_summary(pd.DataFrame(indicators)) if indicators is not None else _empty_ticker_summary()
    if summary is None or summary.empty:
        info_panel(
            "Дискуссионные индикаторы",
            "В выбранном шаге дискретизации не найдено устойчивых тикеров. Рыночный график можно строить только для вручную заданных тикеров, а интерпретация будет относиться к дискуссии в целом.",
            "gray",
        )
        return

    st.markdown("##### Дискуссионные индикаторы по тикерам")
    st.caption("Это не финансовые коэффициенты. Это объяснение, почему сервис подсветил тикер как аналитическое событие в текущем шаге дискретизации дискуссии.")
    cards = summary.head(max_tickers).copy()
    cols = st.columns(min(len(cards), 2) or 1)
    for i, (_, row) in enumerate(cards.iterrows()):
        ticker = str(row.get("ticker", "-")).upper()
        indicator = float(row.get("indicator_max", 0.0) or 0.0)
        share = row.get("ticker_share")
        tone = row.get("sentiment")
        intensity = row.get("sentiment_intensity")
        influence = row.get("avg_influence")
        direction = row.get("direction_hint", "-")
        quality = row.get("ticker_realized_nrmse_mean", row.get("realized_nrmse_mean"))
        if indicator >= 0.45:
            level, color = "сильное аналитическое событие", "red"
        elif indicator >= 0.25:
            level, color = "умеренное аналитическое событие", "orange"
        else:
            level, color = "слабое аналитическое событие", "gray"
        quality_text = "проверка недоступна"
        if pd.notna(quality):
            q = float(quality)
            quality_text = "хорошая согласованность" if q <= 15.0 else "приемлемая согласованность" if q <= 35.0 else "слабая согласованность"
        body = (
            f"<b>{ticker}</b>: {level}. "
            f"Доля внимания: <b>{pct(share, 1)}</b>. "
            f"Доля сообщений: <b>{pct(row.get('activity_rate'), 1)}</b>, рост к контексту: <b>{fmt(row.get('activity_lift'), 2)}x</b>. "
            f"Текущая необычность: <b>{fmt(row.get('discussion_indicator_max'), 3)}</b>, ожидаемая тематическая поддержка: <b>{fmt(row.get('topic_indicator_max'), 3)}</b>. "
            f"Аномальность активности: <b>{fmt(row.get('activity_anomaly_score'), 3)}</b>, поддержка авторов/публикаций: <b>{fmt(row.get('support_score'), 3)}</b>. "
            f"Достоверность тикерной привязки: <b>{fmt(row.get('confidence_avg'), 3)}</b>. "
            f"Направление обсуждения: <b>{direction}</b>. "
            f"Эмоциональная интенсивность: <b>{fmt(intensity, 3)}</b>. "
            f"Средняя графовая авторитетность участников: <b>{fmt(influence, 3)}</b>. "
            f"Проверка по будущим шагам: <b>{quality_text}</b>."
        )
        with cols[i % len(cols)]:
            info_panel(f"{ticker}: {fmt(indicator, 3)}", body, color)


def render_benchmark_explanation(benchmark: str) -> None:
    """Показывает пояснение к выбранному рыночному ориентиру
    
    Args:
        benchmark: рыночный ориентир для сравнения
    
    Returns:
        None
    """
    benchmark = (benchmark or "-").upper()
    info_panel(
        "Как используется бенчмарк",
        f"Бенчмарк <b>{benchmark}</b> сейчас один для всех тикеров. <br>Он не определяет индикатор и не говорит, что все компании надо сравнивать с одним сектором. <br>Он нужен как общий ориентир: двигался ли тикер вместе с широким рынком или заметно отклонялся от него. <br>Для точного исследования можно заменить его вручную: например SPY для широкого рынка, QQQ для технологических компаний, XLF для финансового сектора, XLK для IT-сектора. <br>Дискуссионный индикатор рассчитывается отдельно от бенчмарка.",
        "blue",
    )


def plot_market_price_panel(panel: pd.DataFrame, indicators: pd.DataFrame | None = None, window_hours: int | None = None) -> None:
    """Строит панель цен с фоновыми зонами дискуссионного анализа
    
    Args:
        panel: таблица рыночной панели для отображения
        indicators: таблица или набор дискуссионных индикаторов
        window_hours: длительность шага дискретизации в часах
    
    Returns:
        None
    """
    if panel is None or pd.DataFrame(panel).empty:
        st.info("Не удалось загрузить ценовые ряды по выбранным тикерам.")
        return
    panel = pd.DataFrame(panel).copy()
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce")
    panel["normalized_price"] = pd.to_numeric(panel["normalized_price"], errors="coerce")
    panel = panel.dropna(subset=["Date", "normalized_price"])
    if panel.empty:
        st.info("Ценовой ряд пуст после очистки дат.")
        return

    step_start = pd.to_datetime(panel["event_time"].dropna().iloc[0], errors="coerce") if "event_time" in panel and panel["event_time"].notna().any() else None
    step_end = pd.to_datetime(panel["step_end_time"].dropna().iloc[0], errors="coerce") if "step_end_time" in panel and panel["step_end_time"].notna().any() else None
    context_start = pd.to_datetime(panel["context_start_time"].dropna().iloc[0], errors="coerce") if "context_start_time" in panel and panel["context_start_time"].notna().any() else None
    comparison_end = pd.to_datetime(panel["comparison_end_time"].dropna().iloc[0], errors="coerce") if "comparison_end_time" in panel and panel["comparison_end_time"].notna().any() else None
    horizon_end = pd.to_datetime(panel["horizon_end_time"].dropna().iloc[0], errors="coerce") if "horizon_end_time" in panel and panel["horizon_end_time"].notna().any() else None
    if comparison_end is None or pd.isna(comparison_end):
        comparison_end = horizon_end
    if step_end is None or pd.isna(step_end):
        step_end = step_start + pd.Timedelta(hours=int(window_hours or 1)) if step_start is not None and not pd.isna(step_start) else None
    if context_start is None or pd.isna(context_start):
        context_start = panel["Date"].min() if step_start is not None and not pd.isna(step_start) else None
    interval_label = str(panel["market_interval"].dropna().iloc[0]) if "market_interval" in panel and panel["market_interval"].notna().any() else "1d"

    fig = go.Figure()

    def _add_period_band(x0: Any, x1: Any, fillcolor: str, text: str, font_color: str) -> None:
        x0 = pd.to_datetime(x0, errors="coerce")
        x1 = pd.to_datetime(x1, errors="coerce")
        if pd.isna(x0) or pd.isna(x1) or x1 <= x0:
            return
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            fillcolor=fillcolor,
            line=dict(width=0),
            layer="below",
        )
        fig.add_annotation(
            x=x0 + (x1 - x0) / 2,
            y=1.055,
            xref="x",
            yref="paper",
            text=text,
            showarrow=False,
            font=dict(color=font_color, size=11),
            align="center",
        )

    _add_period_band(context_start, step_start, "rgba(148, 163, 184, 0.14)", "исторический контекст", "#94a3b8")
    _add_period_band(step_start, step_end, "rgba(239, 68, 68, 0.13)", "анализируемый шаг", "#ef4444")
    _add_period_band(step_end, comparison_end, "rgba(37, 99, 235, 0.11)", "сравнение с фактом", "#60a5fa")

    for ticker, part in panel.sort_values(["ticker", "Date"]).groupby("ticker"):
        is_benchmark = bool(part.get("is_benchmark", pd.Series([False])).iloc[0]) if "is_benchmark" in part else False
        fig.add_trace(
            go.Scatter(
                x=part["Date"],
                y=part["normalized_price"],
                mode="lines",
                name=f"{ticker}" + (" (бенчмарк)" if is_benchmark else ""),
                line=dict(width=2.4 if not is_benchmark else 1.7, dash="dash" if is_benchmark else "solid"),
                hovertemplate="%{x}<br>%{fullData.name}: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Движение цены вокруг анализируемого шага дискретизации ({interval_label})",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.24, xanchor="left", x=0),
        margin=dict(l=70, r=20, t=105, b=90),
        height=490,
    )
    fig.update_yaxes(title_text="нормированная цена", automargin=True)
    fig.update_xaxes(title_text="время")
    plotly_chart(fig)
    st.caption("Нормировка выполнена так, что 100 соответствует цене в начале анализируемого шага дискретизации. График показывает только фактическое движение цены и бенчмарка; дискуссионные индикаторы вынесены в интерпретационные заметки выше.")

def render_publications_table(posts: pd.DataFrame) -> None:
    """Показывает таблицу публикаций, связанных с выбранным событием
    
    Args:
        posts: таблица или список публикаций
    
    Returns:
        None
    """
    if posts is None or pd.DataFrame(posts).empty:
        st.info("Для выбранного шага дискретизации не найдено публикаций по выбранным тикерам.")
        return
    df = pd.DataFrame(posts).copy()
    rename = {
        "timestamp": "время",
        "author": "автор",
        "text": "публикация",
        "topic": "тема",
        "sentiment": "тональность",
        "influence": "авторитетность",
        "tickers": "тикеры",
        "impact_score": "вклад",
    }
    cols = [c for c in ["timestamp", "author", "text", "topic", "sentiment", "influence", "tickers", "impact_score"] if c in df.columns]
    st.dataframe(df[cols].rename(columns=rename), width="stretch", hide_index=True)


def download_dataframe_button(df: pd.DataFrame, filename: str, label: str, key: str) -> None:
    """Создаёт кнопку скачивания таблицы в CSV-формате
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
        filename: имя файла для сохранения результата
        label: подпись элемента интерфейса
        key: ключ параметра, виджета или словаря
    
    Returns:
        None
    """
    if df is None or pd.DataFrame(df).empty:
        st.caption("Нет данных для выгрузки.")
        return
    csv = pd.DataFrame(df).to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, csv, file_name=filename, mime="text/csv", key=key)
