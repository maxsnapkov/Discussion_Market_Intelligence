"""Отчёты по дискуссии, тикерам и компонентам индикаторов"""
from __future__ import annotations


from .common import *

from .controls import describe_result_availability, fmt, info_panel, metric_card, pct
from .summaries import (
    _discussion_rows,
    _forecast_quality_label,
    _forward_ticker_summary,
    _research_verdict,
    _ticker_rows,
    _truthy_series,
)

def render_analysis_context_header(df: pd.DataFrame, window_hours: int) -> None:
    """Показывает параметры выбранного анализа и доступность будущей проверки
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
        window_hours: длительность шага дискретизации в часах
    
    Returns:
        None
    """
    if df is None or pd.DataFrame(df).empty:
        return
    x = pd.DataFrame(df).copy()
    availability = describe_result_availability(x)
    summary = _forward_ticker_summary(x)
    discussion = _discussion_rows(x)
    top = summary.iloc[0] if not summary.empty else None
    start_time = "-"
    if "start_time" in x and x["start_time"].notna().any():
        start_time = str(pd.to_datetime(x["start_time"].dropna().iloc[0], errors="coerce"))
    max_step = int(pd.to_numeric(x.get("future_step"), errors="coerce").max()) if "future_step" in x else 1
    # верхний отчёт описывает главный тикерный дискуссионный индикатор
    # строки общего уровня дискуссии показываются в отдельном разделе
    # они не подменяют тикерный показатель в заголовке отчёта
    if top is not None and pd.notna(top.get("indicator_max")):
        best_indicator = float(top.get("indicator_max"))
    elif not summary.empty and "indicator_max" in summary:
        best_indicator = float(summary["indicator_max"].max())
    else:
        d = discussion.copy()
        d["forward_discussion_indicator"] = pd.to_numeric(d.get("forward_discussion_indicator"), errors="coerce")
        best_indicator = float(d["forward_discussion_indicator"].max()) if not d.empty and d["forward_discussion_indicator"].notna().any() else None
    verified = x[_truthy_series(x["actual_available"])] if "actual_available" in x else pd.DataFrame()
    mean_nrmse = float(pd.to_numeric(verified.get("realized_nrmse"), errors="coerce").dropna().mean()) if not verified.empty and "realized_nrmse" in verified else None
    quality, _, _ = _forecast_quality_label(mean_nrmse) if availability["has_any_fact"] else ("ожидает данных", availability["detail"], availability["tone"])
    ticker_text = str(top.get("ticker", "нет")) if top is not None else "нет"
    event_level = "нормальная динамика"
    if best_indicator is not None:
        if best_indicator >= 0.45:
            event_level = "выраженное отклонение"
        elif best_indicator >= 0.25:
            event_level = "умеренное отклонение"
        else:
            event_level = "низкое отклонение"
    caption = (
        f"Шаг дискретизации дискуссии: {start_time}. Горизонт моделирования: {max_step} шагов "
        f"≈ {int(max_step) * int(window_hours)} ч. Статус проверки: {availability['label']}. "
        f"Вкладки ниже - равноправные представления одного результата: отчёт, динамика, публикации, граф авторов и рыночная проверка."
    )
    analysis_context_banner(
        "Симуляционный анализ дискуссии",
        caption,
        [
            ("состояние", event_level, None),
            ("комбинированный тикерный индикатор", fmt(best_indicator, 3), "0.70 текущая необычность + 0.30 модельная поддержка"),
            ("главный тикер", ticker_text, None),
            ("будущие данные", availability["label"], f"nRMSE {fmt(mean_nrmse, 2)}%" if availability["has_any_fact"] else None),
        ],
    )



def render_executive_report(df: pd.DataFrame, window_hours: int) -> None:
    """Показывает краткое резюме главного дискуссионного события
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
        window_hours: размер шага дискретизации в часах
    
    Returns:
        None
    """
    if df is None or pd.DataFrame(df).empty:
        st.info("Индикаторы ещё не рассчитаны.")
        return
    x = pd.DataFrame(df).copy()
    availability = describe_result_availability(x)
    for col in ["forward_discussion_indicator", "combined_forward_discussion_indicator", "ticker_expected_topic_indicator", "model_shift_indicator", "realized_nrmse", "ticker_share", "ticker_sentiment_intensity", "ticker_avg_influence"]:
        if col in x:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    summary = _forward_ticker_summary(x)
    discussion = _discussion_rows(x)
    top = summary.iloc[0] if not summary.empty else None

    # В отчёте главный тикерный показатель всегда берётся из строк по тикерам
    # Дискуссионный модельный сдвиг показывается отдельно и не подменяет тикерный индикатор
    best_indicator = float(summary["indicator_max"].max()) if not summary.empty and "indicator_max" in summary else None
    discussion_model_indicator = None
    if not discussion.empty:
        d = discussion.copy()
        if "model_shift_indicator" in d:
            d["model_shift_indicator"] = pd.to_numeric(d.get("model_shift_indicator"), errors="coerce")
            if d["model_shift_indicator"].notna().any():
                discussion_model_indicator = float(d["model_shift_indicator"].max())

    verified = x[_truthy_series(x["actual_available"])] if "actual_available" in x else pd.DataFrame()
    mean_nrmse = float(pd.to_numeric(verified.get("realized_nrmse"), errors="coerce").dropna().mean()) if not verified.empty and "realized_nrmse" in verified else None
    quality, quality_desc, quality_tone = _forecast_quality_label(mean_nrmse) if availability["has_any_fact"] else ("ожидает данных", availability["detail"], availability["tone"])
    verdict_title, verdict_body, verdict_tone = _research_verdict(summary)
    max_step = int(pd.to_numeric(x.get("future_step"), errors="coerce").max()) if "future_step" in x else 1

    if best_indicator is None or pd.isna(best_indicator):
        level = "Недостаточно данных"
        level_tone = "gray"
    elif best_indicator >= 0.45:
        level = "Выраженное отклонение"
        level_tone = "red"
    elif best_indicator >= 0.25:
        level = "Умеренное отклонение"
        level_tone = "orange"
    else:
        level = "Низкое отклонение"
        level_tone = "green"

    ticker_text = str(top.get("ticker", "нет")) if top is not None else "нет устойчивого тикера"
    direction = str(top.get("direction_hint", "смешанное/нейтральное давление")) if top is not None else "нет тикерной привязки"
    ticker_share = pct(top.get("ticker_share"), 1) if top is not None else "-"
    activity_rate = pct(top.get("activity_rate"), 1) if top is not None and pd.notna(top.get("activity_rate")) else "-"
    activity_lift = fmt(top.get("activity_lift"), 2) + "x" if top is not None and pd.notna(top.get("activity_lift")) else "-"
    context_posts = str(int(top.get("context_post_count"))) if top is not None and pd.notna(top.get("context_post_count")) else "-"
    sent_int = fmt(top.get("sentiment_intensity"), 3) if top is not None else "-"
    avg_infl = fmt(top.get("avg_influence"), 3) if top is not None else "-"

    current_unusualness = top.get("discussion_indicator_max") if top is not None else None
    expected_topic_support = top.get("topic_indicator_max") if top is not None else None
    topic_lift = top.get("topic_lift_max") if top is not None else None
    body = (
        f"В выбранном шаге дискретизации система выделила <b>{level.lower()}</b> по комбинированному тикерному индикатору. "
        f"Ключевой объект проверки: <b>{ticker_text}</b>. Показатели ниже разделены по смыслу: "
        f"текущая статистическая необычность тикерного обсуждения, модельная ожидаемая тематическая поддержка и их комбинированная оценка. "
        f"Это не инвестиционная рекомендация и не утверждение причинности. {verdict_body} {quality_desc}"
    )
    facts = [
        ("Комбинированный тикерный индикатор", fmt(best_indicator, 3)),
        ("Текущая необычность обсуждения", fmt(current_unusualness, 3)),
        ("Ожидаемая тематическая поддержка", fmt(expected_topic_support, 3)),
        ("Общий ожидаемый сдвиг тем", fmt(discussion_model_indicator, 3)),
        ("Ожидаемый тематический прирост", fmt(topic_lift, 3)),
        ("Главный тикер", ticker_text),
        ("Доля внимания", ticker_share),
        ("Доля сообщений", activity_rate),
        ("Сообщений в контексте", context_posts),
        ("Отношение к контексту", activity_lift),
        ("Эмоц. интенсивность", sent_int),
        ("Графовая авторитетность", avg_infl),
        ("Будущие данные", f"{availability['label']}" if not availability["has_any_fact"] else f"{quality}; nRMSE {fmt(mean_nrmse, 2)}%"),
        ("Горизонт", f"{max_step} шагов дискретизации ≈ {max_step * window_hours} ч"),
        ("Направление", direction),
    ]
    chips = [(level, level_tone), (verdict_title, verdict_tone), (availability["title"], availability["tone"])]
    executive_board("Симуляционный анализ шага дискретизации динамики дискуссии", body, facts, chips=chips)

    if not summary.empty:
        render_indicator_decomposition(x)

    if summary.empty:
        info_panel("Тикерная привязка", "В выбранном шаге нет устойчивой тикерной привязки. Анализ дискуссии остаётся валидным, но рыночную проверку лучше не интерпретировать как тикерное событие.", "gray")
        return

    section_header("Приоритет тикеров", "Одна строка - один финансовый инструмент, связанный с текущим шагом дискретизации. Подробные публикации и рынок открываются в соседних разделах результата", term="тикерная привязка")
    rank = summary[[c for c in ["ticker", "indicator_max", "discussion_indicator_max", "topic_indicator_max", "model_shift_indicator_max", "topic_lift_max", "topic_shift_max", "current_topic_exposure_max", "expected_topic_exposure_max", "event_pvalue", "event_qvalue", "observed_support_pvalue", "predictive_evidence_pvalue", "activity_pvalue", "growth_pvalue", "current_prominence_pvalue", "presence_pvalue", "model_pvalue", "text_pvalue", "ticker_share", "activity_rate", "activity_lift", "activity_robust_z", "post_count", "unique_authors", "direct_post_count", "inherited_post_count", "participation_mode", "sentiment_intensity", "avg_influence", "confidence_avg", "direction_hint"] if c in summary.columns]].head(8).copy()
    rank = rank.rename(columns={
        "ticker": "Тикер",
        "indicator_max": "Комбинированный индикатор",
        "discussion_indicator_max": "Текущая необычность",
        "topic_indicator_max": "Модельная тематическая поддержка",
        "model_shift_indicator_max": "Общий ожидаемый сдвиг тем",
        "topic_lift_max": "Ожидаемый тематический прирост",
        "topic_shift_max": "Сдвиг тематической экспозиции",
        "current_topic_exposure_max": "Текущая тематическая экспозиция",
        "expected_topic_exposure_max": "Ожидаемая тематическая экспозиция",
        "event_pvalue": "p-value события",
        "event_qvalue": "q-value события",
        "activity_pvalue": "p-value активности",
        "observed_support_pvalue": "p-value текущей поддержки",
        "predictive_evidence_pvalue": "p-value симуляционных признаков",
        "growth_pvalue": "p-value роста",
        "current_prominence_pvalue": "p-value концентрации",
        "presence_pvalue": "p-value присутствия",
        "model_pvalue": "p-value модели",
        "text_pvalue": "p-value текста",
        "ticker_share": "Доля внимания",
        "activity_rate": "Доля сообщений",
        "activity_lift": "Отношение к контексту",
        "activity_robust_z": "Robust z активности",
        "attention_score": "Внимание",
        "activity_anomaly_score": "1 - p активности",
        "support_score": "1 - p активности",
        "current_post_count": "Сообщений в текущем шаге",
        "current_unique_authors": "Авторов в текущем шаге",
        "context_post_count": "Сообщений в контексте",
        "context_unique_authors": "Авторов в контексте",
        "context_share": "Доля в контексте",
        "historical_support_score": "Историческая поддержка",
        "post_count": "Участий в ветке",
        "unique_authors": "Авторов",
        "direct_post_count": "Прямых упоминаний",
        "inherited_post_count": "Ответов в ветке",
        "participation_mode": "Учёт участия",
        "sentiment_intensity": "Эмоц. интенсивность",
        "avg_influence": "Графовая авторитетность",
        "confidence_avg": "Достоверность привязки",
        "direction_hint": "Направление",
    })
    st.dataframe(rank, width="stretch", hide_index=True)
    plot_forward_summary(x)


def render_discussion_state_overview(df: pd.DataFrame) -> None:
    """Показывает состояние дискуссии и общий ожидаемый тематический сдвиг
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        None
    """
    availability = describe_result_availability(pd.DataFrame(df))
    d = _discussion_rows(df)
    if d.empty:
        return
    d = d.copy().sort_values("future_step")
    for col in ["forward_discussion_indicator", "model_shift_indicator", "model_expected_topic_shift", "predicted_topic_drift", "ticker_topic_exposure", "predicted_topic_concentration", "realized_nrmse"]:
        if col in d:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    best_sort_col = "model_shift_indicator" if "model_shift_indicator" in d.columns and d["model_shift_indicator"].notna().any() else "forward_discussion_indicator"
    best = d.sort_values(best_sort_col, ascending=False).iloc[0]
    verified = d[_truthy_series(d["actual_available"])] if "actual_available" in d else pd.DataFrame()
    mean_nrmse = float(verified["realized_nrmse"].dropna().mean()) if not verified.empty and "realized_nrmse" in verified else None
    q_title, q_desc, q_tone = _forecast_quality_label(mean_nrmse) if availability["has_any_fact"] else ("ожидает данных", availability["detail"], availability["tone"])
    st.markdown("### Отчёт по дискуссии")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            "Индикатор ожидаемого тематического сдвига",
            fmt(best.get("model_shift_indicator", best.get("forward_discussion_indicator")), 3),
            "Нормированная шкала: насколько атипичное изменение распределения тем ожидает модель",
            "violet",
        )
    with c2:
        metric_card(
            "Локальный сдвиг тем",
            fmt(best.get("predicted_topic_drift"), 3),
            "L1 между соседними смоделированными шагами",
            "blue",
        )
    with c3:
        metric_card(
            "Накопленный сдвиг тем",
            fmt(best.get("model_expected_topic_shift", best.get("ticker_topic_exposure")), 3),
            "Ожидаемое L1-расстояние от выбранного шага Wₜ",
            "orange",
        )
    with c4:
        metric_card("Будущие данные", availability["label"] if not availability["has_any_fact"] else q_title, f"mean nRMSE: {fmt(mean_nrmse, 2)}%", q_tone)
    info_panel(
        "Как читать эти метрики",
        "Индикатор ожидаемого тематического сдвига не использует фактические будущие данные. Он показывает, насколько сильное перестроение тематической структуры ожидает агентная модель относительно исторического контекста. Тикерный комбинированный индикатор отдельно объединяет текущую необычность обсуждения и ожидаемую тематическую поддержку финансового инструмента.",
        "blue",
    )
    plot_cols = [c for c in ["future_step", "model_shift_indicator", "model_expected_topic_shift", "predicted_topic_drift", "ticker_topic_exposure"] if c in d.columns]
    plot = d[plot_cols].copy()
    plot = plot.rename(columns={
        "model_shift_indicator": "индикатор ожидаемого сдвига",
        "model_expected_topic_shift": "ожидаемый сдвиг от Wₜ",
        "predicted_topic_drift": "локальный сдвиг тем",
        "ticker_topic_exposure": "накопленный сдвиг тем",
    })
    long = plot.melt(id_vars="future_step", var_name="метрика", value_name="значение")
    fig = px.line(
        long,
        x="future_step",
        y="значение",
        color="метрика",
        markers=True,
        title="Модельная траектория дискуссии: что именно меняется в следующих шагах",
        labels={"future_step": "следующий шаг (+N)", "значение": "нормированное значение / L1", "метрика": "метрика"}
    )
    fig.update_xaxes(dtick=1)
    fig.update_traces(mode="lines+markers")
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    plotly_chart(fig)

def render_current_interest_feed(df: pd.DataFrame) -> None:
    """Показывает текущую динамику ценообразования финансовых инструментов
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        None
    """
    summary = _forward_ticker_summary(df)
    if summary.empty:
        st.info("В текущем шаге нет устойчивого тикерного аналитического события. Можно анализировать темы и публикации, но рыночную привязку лучше не делать.")
        return
    st.markdown("### Аналитические события текущего шага")
    st.caption("Строки показывают не список тикеров сам по себе, а факторы, из-за которых тикер получил высокий комбинированный индикатор: текущую необычность обсуждения, модельную тематическую поддержку, долю внимания, эмоциональность и графовую авторитетность участников.")
    cols = st.columns(min(3, max(1, len(summary))))
    for idx, (_, row) in enumerate(summary.head(6).iterrows()):
        ind = float(row.get("indicator_max", 0.0) or 0.0)
        share = row.get("ticker_share")
        infl = row.get("avg_influence")
        sent_int = row.get("sentiment_intensity")
        if ind >= 0.45:
            level, tone = "Сильный индикатор", "red"
        elif ind >= 0.25:
            level, tone = "Умеренный индикатор", "orange"
        else:
            level, tone = "Наблюдать", "blue"
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="post-card" data-tone="{tone}">
                  <div class="report-title">{row.get('ticker', '-')} · {level}</div>
                  <div class="report-body">
                    <b>Комбинированный индикатор:</b> {fmt(ind, 3)}<br>
                    <b>Текущая необычность:</b> {fmt(row.get('discussion_indicator_max'), 3)}<br>
                    <b>Модельная поддержка:</b> {fmt(row.get('topic_indicator_max'), 3)}<br>
                    <b>Доля внимания:</b> {pct(share, 1)}<br>
                    <b>Доля сообщений:</b> {pct(row.get('activity_rate'), 1)} · рост {fmt(row.get('activity_lift'), 2)}x<br>
                    <b>Эмоциональная интенсивность:</b> {fmt(sent_int, 3)}<br>
                    <b>Графовая авторитетность участников:</b> {fmt(infl, 3)}<br>
                    <b>Интерпретация:</b> {row.get('direction_hint', 'смешанное/нейтральное давление')}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_ticker_signal_matrix(df: pd.DataFrame) -> None:
    """Показывает матрицу дискуссионных показателей по финансовым инструментам
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        None
    """
    summary = _forward_ticker_summary(df)
    if summary.empty:
        return
    out = summary[[c for c in ["ticker", "indicator_max", "discussion_indicator_max", "topic_indicator_max", "model_shift_indicator_max", "topic_lift_max", "topic_shift_max", "current_topic_exposure_max", "expected_topic_exposure_max", "ticker_share", "activity_rate", "activity_lift", "current_post_count", "current_unique_authors", "context_post_count", "context_unique_authors", "context_share", "historical_support_score", "sentiment", "sentiment_intensity", "avg_influence", "ticker_has_topic_profile", "ticker_realized_nrmse_mean", "realized_nrmse_mean", "direction_hint", "source_scope"] if c in summary.columns]].copy()
    out = out.rename(columns={
        "ticker": "Тикер",
        "indicator_max": "Комбинированный индикатор",
        "discussion_indicator_max": "Текущая необычность",
        "topic_indicator_max": "Модельная тематическая поддержка",
        "model_shift_indicator_max": "Общий ожидаемый сдвиг тем",
        "topic_lift_max": "Ожидаемый тематический прирост",
        "topic_shift_max": "Сдвиг тематической экспозиции",
        "current_topic_exposure_max": "Текущая тематическая экспозиция",
        "expected_topic_exposure_max": "Ожидаемая тематическая экспозиция",
        "event_pvalue": "p-value события",
        "event_qvalue": "q-value события",
        "activity_pvalue": "p-value активности",
        "observed_support_pvalue": "p-value текущей поддержки",
        "predictive_evidence_pvalue": "p-value симуляционных признаков",
        "growth_pvalue": "p-value роста",
        "current_prominence_pvalue": "p-value концентрации",
        "presence_pvalue": "p-value присутствия",
        "model_pvalue": "p-value модели",
        "text_pvalue": "p-value текста",
        "ticker_share": "Доля внимания",
        "activity_rate": "Доля сообщений",
        "activity_lift": "Отношение к контексту",
        "current_post_count": "Сообщений в шаге",
        "current_unique_authors": "Авторов в шаге",
        "context_post_count": "Сообщений в контексте",
        "context_unique_authors": "Авторов в контексте",
        "context_share": "Доля в контексте",
        "historical_support_score": "Историческая поддержка",
        "post_count": "Участий в ветке",
        "unique_authors": "Авторов",
        "direct_post_count": "Прямых упоминаний",
        "inherited_post_count": "Ответов в ветке",
        "participation_mode": "Учёт участия",
        "sentiment": "Средняя тональность",
        "sentiment_intensity": "Эмоц. интенсивность",
        "avg_influence": "Графовая авторитетность",
        "ticker_has_topic_profile": "есть профиль тем",
        "ticker_realized_nrmse_mean": "nRMSE, % тикера по доступным данным",
        "realized_nrmse_mean": "nRMSE, % дискуссии по доступным данным",
        "direction_hint": "Направление обсуждения",
        "source_scope": "Источник тикера",
    })
    st.markdown("#### Сводка по тикерам текущего шага")
    st.caption("Одна строка = один тикер в текущем эксперименте. Детализация по следующим шагам дискретизации спрятана в техническом блоке.")
    st.dataframe(out, width="stretch", hide_index=True)

def render_forward_report(df: pd.DataFrame, window_hours: int) -> None:
    """Показывает отчёт о дискуссионных индикаторах и модельно-тематической поддержке
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
        window_hours: размер шага дискретизации в часах
    
    Returns:
        None
    """
    if df is None or pd.DataFrame(df).empty:
        st.info("Индикаторы ещё не рассчитаны.")
        return
    df = pd.DataFrame(df).copy()
    for col in ["forward_discussion_indicator", "realized_nrmse", "ticker_share", "ticker_sentiment_intensity", "ticker_avg_influence"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    summary = _forward_ticker_summary(df)
    verdict_title, verdict_body, verdict_tone = _research_verdict(summary)
    verified = df[_truthy_series(df["actual_available"])] if "actual_available" in df else pd.DataFrame()
    mean_nrmse = float(verified["realized_nrmse"].dropna().mean()) if not verified.empty and "realized_nrmse" in verified else None
    q_title, q_desc, q_tone = _forecast_quality_label(mean_nrmse)
    max_step = int(pd.to_numeric(df.get("future_step"), errors="coerce").max()) if "future_step" in df else 1
    top = summary.iloc[0] if not summary.empty else None

    st.markdown("### Исследовательский отчёт по выбранному шагу дискретизации")
    a, b, c, d = st.columns(4)
    with a:
        metric_card("Главный тикер", str(top.get("ticker", "-")) if top is not None else "-", f"комбинированный индикатор {fmt(top.get('indicator_max'), 3) if top is not None else '-'}", "violet")
    with b:
        metric_card("Вывод", verdict_title, "оценка пригодности анализа выбранного шага", verdict_tone)
    with c:
        metric_card("Дискуссия", q_title, f"mean nRMSE: {fmt(mean_nrmse, 2)}%", q_tone)
    with d:
        metric_card("Период проверки", f"{max_step} шагов", f"примерно {max_step * window_hours} ч", "blue")

    if top is not None:
        direction = str(top.get("direction_hint", "смешанное/нейтральное давление"))
        st.markdown(
            f"""
            <div class="report-card">
              <div class="report-title">Что произошло в дискуссии</div>
              <div class="report-body">
                В выбранном шаге дискретизации сильнее всего выделяется <b>{top.get('ticker', '-')}</b>.
                Доля внимания к тикеру - <b>{pct(top.get('ticker_share'), 1)}</b>,
                эмоциональная интенсивность - <b>{fmt(top.get('sentiment_intensity'), 3)}</b>,
                среднее значение вычисленной авторитетности авторов - <b>{fmt(top.get('avg_influence'), 3)}</b>.
                Направление обсуждения: <b>{direction}</b>.
                <br><br><b>Интерпретация:</b> {verdict_body} {q_desc}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        info_panel("Что произошло в дискуссии", verdict_body, "orange")

    if not summary.empty:
        st.markdown("#### Приоритетные тикеры для проверки")
        cols = st.columns(min(3, len(summary)))
        for idx, (_, row) in enumerate(summary.head(6).iterrows()):
            with cols[idx % len(cols)]:
                nrmse = row.get("realized_nrmse_mean")
                quality, _, tone = _forecast_quality_label(nrmse)
                metric_card(
                    str(row.get("ticker", "-")),
                    fmt(row.get("indicator_max"), 3),
                    f"{row.get('direction_hint', '')}; {quality}",
                    tone if tone != "gray" else "blue",
                )
                st.caption(f"Внимание: {pct(row.get('ticker_share'), 1)} · среднее значение авторитетности авторов: {fmt(row.get('avg_influence'), 3)}")

def plot_forward_indicators(df: pd.DataFrame) -> None:
    """Строит график итоговых дискуссионных индикаторов по финансовым инструментам
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        None
    """
    tick = _ticker_rows(df)
    if tick.empty:
        st.info("В текущем шаге дискретизации не выявлено тикерных событий для анализа. Доступна общая динамика дискуссии выше.")
        return
    y_col = "combined_forward_discussion_indicator" if "combined_forward_discussion_indicator" in tick.columns and tick["combined_forward_discussion_indicator"].notna().any() else "forward_discussion_indicator"
    plot = tick.copy().sort_values(["future_step", y_col])
    fig = px.line(
        plot,
        x="future_step",
        y=y_col,
        color="ticker",
        markers=True,
        title="Тикерные события по следующим шагам дискретизации",
        labels={"future_step": "следующий шаг дискретизации (+N)", y_col: "комбинированный индикатор" if y_col == "combined_forward_discussion_indicator" else "текущая необычность", "ticker": "тикер"}
    )
    fig.update_traces(mode="lines+markers")
    fig.update_xaxes(dtick=1)
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    plotly_chart(fig)



def plot_forward_summary(df: pd.DataFrame) -> None:
    """Строит сводный график текущей необычности и модельно-тематической поддержки
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        None
    """
    summary = _forward_ticker_summary(df)
    if summary.empty:
        return
    plot = summary.head(8).sort_values("indicator_max", ascending=True)
    fig = px.bar(
        plot,
        x="indicator_max",
        y="ticker",
        orientation="h",
        title="Комбинированный индикатор тикерного события",
        labels={"indicator_max": "комбинированный индикатор", "ticker": "тикер"},
        hover_data=[c for c in ["discussion_indicator_max", "topic_indicator_max", "topic_lift_max", "ticker_share", "sentiment_intensity", "avg_influence", "ticker_realized_nrmse_mean", "realized_nrmse_mean"] if c in plot.columns],
    )
    fig.add_vline(x=0.25, line_dash="dash", line_color="#f59e0b", annotation_text="умеренный")
    fig.add_vline(x=0.45, line_dash="dash", line_color="#16a34a", annotation_text="сильный")
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20), coloraxis_showscale=False)
    plotly_chart(fig)


def render_indicator_decomposition(df: pd.DataFrame) -> None:
    """Показывает разложение итогового индикатора на компоненты
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        None
    """
    if df is None or pd.DataFrame(df).empty:
        return
    x = pd.DataFrame(df).copy()
    summary = _forward_ticker_summary(x)
    if summary.empty:
        return

    section_header(
        "Декомпозиция индикаторов",
        "Комбинированный показатель складывается из текущей статистической необычности тикерного обсуждения и модельной тематической поддержки, рассчитанной по симуляцияу агентной модели без будущих фактических данных.",
        term="индикаторы",
    )
    info_panel(
        "Что именно разделено",
        "1) Текущая необычность обсуждения - насколько необычно тикер представлен сейчас относительно истории: сообщения, авторы, ветви, p-значения и фильтр минимальной поддержки. <br>2) Модельная тематическая поддержка - связан ли тикер с темами, которые агентная модель ожидает усилить в следующих шагах дискретизации. <br>3) Комбинированный индикатор - инженерная агрегация: 0.70 текущей необычности и 0.30 модельной поддержки.",
        "blue",
    )

    top = summary.iloc[0]
    a, b, c = st.columns(3)
    with a:
        metric_card("Текущая необычность", fmt(top.get("discussion_indicator_max"), 3), "статистика тикерного обсуждения", "orange")
    with b:
        metric_card("Модельная поддержка", fmt(top.get("topic_indicator_max"), 3), "ожидаемая тематическая экспозиция", "violet")
    with c:
        metric_card("Комбинированный индикатор", fmt(top.get("indicator_max"), 3), "0.70 × текущая + 0.30 × модельная", "green")

    comp_cols = [
        ("discussion_indicator_max", "текущая необычность"),
        ("topic_indicator_max", "модельная поддержка"),
        ("indicator_max", "комбинированный индикатор"),
    ]
    comp = summary.head(8)[["ticker"] + [c for c, _ in comp_cols if c in summary.columns]].copy()
    if len(comp.columns) > 1:
        comp = comp.rename(columns={c: label for c, label in comp_cols})
        long = comp.melt(id_vars="ticker", var_name="компонент", value_name="значение").dropna(subset=["значение"])
        fig = px.bar(
            long,
            x="ticker",
            y="значение",
            color="компонент",
            barmode="group",
            title="Из чего состоит итоговый тикерный показатель",
            labels={"ticker": "тикер", "значение": "значение 0-1", "компонент": "компонент"},
        )
        fig.add_hline(y=0.25, line_dash="dash", line_color="#f59e0b", annotation_text="умеренный")
        fig.add_hline(y=0.45, line_dash="dash", line_color="#16a34a", annotation_text="сильный")
        fig.update_layout(margin=dict(l=20, r=20, t=55, b=20), legend_title_text="Компонент")
        plotly_chart(fig)

    exposure_cols = [c for c in ["current_topic_exposure_max", "expected_topic_exposure_max", "topic_lift_max", "topic_shift_max"] if c in summary.columns]
    if exposure_cols:
        exposure = summary.head(8)[["ticker"] + exposure_cols].copy()
        rename = {
            "current_topic_exposure_max": "текущая тематическая экспозиция",
            "expected_topic_exposure_max": "ожидаемая тематическая экспозиция",
            "topic_lift_max": "ожидаемый тематический прирост",
            "topic_shift_max": "сдвиг тематической экспозиции",
        }
        exposure = exposure.rename(columns=rename)
        long_exp = exposure.melt(id_vars="ticker", var_name="метрика", value_name="значение").dropna(subset=["значение"])
        fig = px.bar(
            long_exp,
            x="ticker",
            y="значение",
            color="метрика",
            barmode="group",
            title="Как агентная модель поддерживает тикеры через ожидаемое усиление тем",
            labels={"ticker": "тикер", "значение": "значение", "метрика": "метрика"},
        )
        fig.update_layout(margin=dict(l=20, r=20, t=55, b=20), legend_title_text="Метрика")
        plotly_chart(fig)

    tick = _ticker_rows(x)
    if not tick.empty and "future_step" in tick.columns:
        show_tickers = summary.head(5)["ticker"].dropna().astype(str).str.upper().tolist()
        plot = tick[tick["ticker"].astype(str).str.upper().isin(show_tickers)].copy()
        for col in ["forward_discussion_indicator", "ticker_expected_topic_indicator", "combined_forward_discussion_indicator", "ticker_expected_topic_lift"]:
            if col in plot:
                plot[col] = pd.to_numeric(plot[col], errors="coerce")
        line_cols = [c for c in ["forward_discussion_indicator", "ticker_expected_topic_indicator", "combined_forward_discussion_indicator"] if c in plot.columns]
        if line_cols:
            labels = {
                "forward_discussion_indicator": "текущая необычность",
                "ticker_expected_topic_indicator": "модельная поддержка",
                "combined_forward_discussion_indicator": "комбинированный индикатор",
            }
            line = plot[["future_step", "ticker"] + line_cols].rename(columns=labels)
            long_line = line.melt(id_vars=["future_step", "ticker"], var_name="компонент", value_name="значение").dropna(subset=["значение"])
            fig = px.line(
                long_line,
                x="future_step",
                y="значение",
                color="ticker",
                line_dash="компонент",
                markers=True,
                title="Динамика компонентов по смоделированным шагам будущего тематического распределения дискуссии",
                labels={"future_step": "следующий шаг (+N)", "значение": "значение 0-1", "ticker": "тикер", "компонент": "компонент"},
            )
            fig.update_xaxes(dtick=1)
            fig.update_layout(margin=dict(l=20, r=20, t=55, b=20), legend_title_text="Тикер / компонент")
            plotly_chart(fig)

        # if "ticker_expected_topic_indicator" in plot.columns and plot["ticker_expected_topic_indicator"].notna().any():
        #     heat = plot.pivot_table(index="ticker", columns="future_step", values="ticker_expected_topic_indicator", aggfunc="max")
        #     if not heat.empty:
        #         fig = px.imshow(
        #             heat,
        #             aspect="auto",
        #             title="Тепловая карта модельной тематической поддержки",
        #             labels={"x": "следующий шаг (+N)", "y": "тикер", "color": "поддержка"},
        #         )
        #         fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
        #         plotly_chart(fig)

def render_forecast_check_text(df: pd.DataFrame) -> None:
    """Показывает текстовую интерпретацию проверки моделирования по фактическим данным
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        None
    """
    availability = describe_result_availability(pd.DataFrame(df) if df is not None else None)
    if df is None or pd.DataFrame(df).empty or "actual_available" not in df:
        info_panel(availability["title"], availability["detail"], availability["tone"])
        return
    x = pd.DataFrame(df).copy()
    verified = x[_truthy_series(x["actual_available"])]
    if verified.empty or "realized_nrmse" not in verified:
        info_panel(availability["title"], availability["detail"], availability["tone"])
        return
    verified["realized_nrmse"] = pd.to_numeric(verified["realized_nrmse"], errors="coerce")
    mean_nrmse = float(verified["realized_nrmse"].dropna().mean())
    title, desc, tone = _forecast_quality_label(mean_nrmse)
    good = float((verified["realized_nrmse"] <= 15.0).mean())
    weak = float((verified["realized_nrmse"] > 35.0).mean())
    info_panel(
        "Проверка модельной симуляции по доступным будущим данным",
        f"Средний nRMSE = {fmt(mean_nrmse, 2)}%. Доля хороших совпадений = {pct(good, 1)}, доля слабых = {pct(weak, 1)}. Вывод: {title}. {desc}",
        tone,
    )
