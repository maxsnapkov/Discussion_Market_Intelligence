"""Подготовка отчетов дискуссионных и рыночных индикаторов"""
from __future__ import annotations


from .common import *

from .controls import fmt, pct

def compact_forward_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Готовит компактную таблицу дискуссионных индикаторов для отображения
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        pd.DataFrame: таблица дискуссионных индикаторов с русскими названиями колонок
    """
    if df.empty:
        return df
    cols = [
        "future_step", "future_time", "ticker", "combined_forward_discussion_indicator",
        "forward_discussion_indicator", "ticker_expected_topic_indicator", "model_shift_indicator",
        "ticker_share", "direction_hint", "predicted_topic_drift", "ticker_topic_exposure",
        "ticker_expected_topic_lift", "ticker_sentiment", "ticker_sentiment_intensity", "ticker_avg_influence",
        "actual_available", "realized_nrmse", "realized_l1", "interpretation",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    rename = {
        "future_step": "+шагов моделирования",
        "future_time": "следующий шаг",
        "ticker": "тикер",
        "combined_forward_discussion_indicator": "комбинированный индикатор",
        "forward_discussion_indicator": "текущая необычность",
        "ticker_expected_topic_indicator": "ожидаемая тематическая поддержка",
        "model_shift_indicator": "ожидаемый сдвиг дискуссии",
        "ticker_share": "доля внимания",
        "direction_hint": "направление",
        "predicted_topic_drift": "локальный сдвиг тем",
        "ticker_topic_exposure": "тикерный вклад в сдвиг",
        "ticker_expected_topic_lift": "ожидаемый тематический прирост",
        "ticker_sentiment": "средняя тональность",
        "ticker_sentiment_intensity": "эмоц. интенсивность",
        "ticker_avg_influence": "графовая авторитетность",
        "actual_available": "будущее уже известно",
        "realized_nrmse": "nRMSE, % по доступным данным",
        "realized_l1": "проверочный L1",
        "interpretation": "интерпретация",
    }
    return out.rename(columns=rename)


def _tone_from_value(value: float, high: float, mid: float) -> tuple[str, str]:
    """Подбирает визуальный тон карточки по числовому значению
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
        high: порог высокого значения метрики
        mid: порог среднего значения метрики
    
    Returns:
        tuple[str, str]: текстовый уровень и визуальный тон для значения метрики
    """
    try:
        v = float(value)
    except Exception:
        return "серый", "gray"
    if v >= high:
        return "высокий", "red"
    if v >= mid:
        return "умеренный", "orange"
    return "низкий", "green"


def _forecast_quality_label(nrmse: float | None) -> tuple[str, str, str]:
    """Формирует текстовую оценку качества моделирования по nRMSE
    
    Args:
        nrmse: значение нормированной среднеквадратичной ошибки
    
    Returns:
        tuple[str, str, str]: краткая оценка качества, пояснение и визуальный тон по nRMSE
    """
    if nrmse is None or pd.isna(nrmse):
        return "нельзя проверить", "Будущий шаг ещё отсутствует в истории: индикатор сформирован ex-ante, но факт пока неизвестен.", "gray"
    n = float(nrmse)
    if n <= 15.0:
        return "хорошее совпадение", "Модельная динамика близка к фактической по nRMSE, рассчитанному в процентах от размаха фактической матрицы тематических долей.", "green"
    if n <= 35.0:
        return "приемлемое совпадение", "Модель улавливает часть динамики, но интерпретацию стоит проверять по темам, тикерам и публикациям.", "orange"
    return "слабое совпадение", "Модельная и фактическая динамика заметно расходятся: индикатор полезен как предупреждение, но доверие к нему ограничено.", "red"




def _truthy_series(values: Any) -> pd.Series:
    """Преобразует серию признаков в булеву маску
    
    Args:
        values: последовательность числовых или строковых значений
    
    Returns:
        pd.Series: булева маска, полученная из строковых, числовых и логических значений
    """
    if isinstance(values, pd.Series):
        s = values.copy()
    else:
        s = pd.Series(values)
    def conv(x: Any) -> bool:
        """Преобразует значение серии в булев признак
        
        Args:
            x: числовое значение или элемент последовательности
        
        Returns:
            bool: логическое значение после нормализации элемента серии
        """
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return False
        text = str(x).strip().lower()
        return text in {"true", "1", "yes", "y", "да", "истина"}
    return s.map(conv).fillna(False).astype(bool)

def _direction_russian(value: Any) -> str:
    """Переводит направление рыночного движения
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        str: подпись направления дискуссионного давления
    """
    text = str(value or "").lower()
    if "позит" in text or "bull" in text:
        return "скорее позитивное давление"
    if "нег" in text or "bear" in text:
        return "скорее негативное давление"
    return "смешанное/нейтральное давление"


def _ticker_universe_caption(universe: pd.DataFrame) -> str:
    """Формирует подпись о размере и источнике справочника тикеров
    
    Args:
        universe: таблица справочника тикеров и компаний
    
    Returns:
        str: строка с описанием размера и источника справочника тикеров
    """
    if universe is None or pd.DataFrame(universe).empty:
        return "справочник пуст"
    df = pd.DataFrame(universe).copy()
    if "source" not in df.columns:
        return "локальный справочник без метаданных источника"
    counts = df["source"].fillna("unknown").astype(str).value_counts().to_dict()
    if set(counts) == {"fallback"}:
        return f"fallback-словарь: {int(counts.get('fallback', 0))}"
    parts = [f"{source}: {count}" for source, count in sorted(counts.items())]
    return "локальный кэш, " + "; ".join(parts[:4])



DISCUSSION_LEVEL_TICKER = "__DISCUSSION__"


def _empty_ticker_summary() -> pd.DataFrame:
    """Возвращает пустую таблицу итогов по финансовым инструментам
    
    Returns:
        pd.DataFrame: пустая таблица с колонками сводки финансовых инструментов
    """
    return pd.DataFrame(columns=[
        "ticker", "indicator_max", "indicator_mean", "best_step", "ticker_share",
        "sentiment", "sentiment_intensity", "avg_influence", "realized_nrmse_mean",
        "ticker_realized_nrmse_mean", "n_steps_verified", "direction_hint", "interpretation",
    ])


def _split_discussion_and_tickers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разделяет строки общего дискуссионного уровня и строки финансовых инструментов
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: пара таблиц с общим уровнем дискуссии и тикерными строками
    """
    if df is None or pd.DataFrame(df).empty:
        return pd.DataFrame(), pd.DataFrame()
    x = pd.DataFrame(df).copy()
    if "ticker" not in x.columns:
        # если колонки тикера нет, считаем таблицу результатом общего уровня дискуссии
        return x.copy(), pd.DataFrame()
    ticker_norm = x["ticker"].astype(str).str.strip().str.upper()
    is_discussion = ticker_norm.eq(DISCUSSION_LEVEL_TICKER)
    bad_ticker = ticker_norm.isin({"", "NAN", "NONE", "NULL", "[]", "-"})
    discussion = x[is_discussion].copy()
    tickers = x[(~is_discussion) & (~bad_ticker)].copy()
    return discussion, tickers


def _discussion_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Выбирает строки результатов, относящиеся ко всей дискуссии
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        pd.DataFrame: таблица строк, относящихся к общему уровню дискуссии
    """
    return _split_discussion_and_tickers(df)[0]


def _ticker_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Выбирает строки результатов, относящиеся к финансовым инструментам
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        pd.DataFrame: таблица строк, относящихся к финансовым инструментам
    """
    return _split_discussion_and_tickers(df)[1]


def _forward_ticker_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Строит ранжированную сводку финансовых инструментов по дискуссионным индикаторам
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        pd.DataFrame: ранжированная сводка финансовых инструментов по компонентам индикатора
    """
    if df is None or pd.DataFrame(df).empty:
        return _empty_ticker_summary()
    x = _ticker_rows(pd.DataFrame(df).copy())
    if x.empty or "ticker" not in x.columns:
        return _empty_ticker_summary()
    for col in [
        "forward_discussion_indicator", "combined_forward_discussion_indicator", "ticker_expected_topic_indicator",
        "ticker_expected_topic_lift", "ticker_expected_topic_shift", "ticker_expected_step_shift",
        "ticker_current_topic_exposure", "ticker_expected_topic_exposure", "model_shift_indicator",
        "ticker_share", "ticker_sentiment",
        "ticker_sentiment_intensity", "ticker_avg_influence", "realized_nrmse",
        "predicted_topic_drift", "ticker_topic_exposure", "ticker_realized_nrmse",
        "ticker_model_step_shift", "ticker_model_from_start_shift",
        "ticker_activity_rate", "ticker_activity_lift", "ticker_activity_robust_zscore",
        "ticker_post_count", "ticker_unique_authors", "ticker_direct_post_count", "ticker_inherited_post_count", "ticker_attention_score",
        "ticker_activity_anomaly_score", "ticker_support_score", "ticker_confidence_avg", "ticker_context_score_avg",
        "ticker_current_post_count", "ticker_current_unique_authors", "ticker_context_post_count", "ticker_context_unique_authors",
        "ticker_context_share", "ticker_historical_support_score", "ticker_context_direct_post_count", "ticker_context_inherited_post_count",
        "ticker_posts_pvalue", "ticker_authors_pvalue", "ticker_thread_pvalue", "ticker_growth_pvalue",
        "ticker_current_prominence_pvalue", "ticker_context_prominence_pvalue", "ticker_presence_pvalue", "ticker_activity_pvalue",
        "ticker_model_pvalue", "ticker_topic_pvalue", "ticker_text_pvalue", "ticker_global_activity_rate_pvalue", "ticker_observed_support_pvalue", "ticker_predictive_evidence_pvalue", "ticker_discussion_event_pvalue", "ticker_discussion_event_qvalue",
    ]:
        if col in x:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    rows = []
    for ticker, part in x.groupby("ticker", dropna=False):
        rank_col = "combined_forward_discussion_indicator" if "combined_forward_discussion_indicator" in part.columns and part["combined_forward_discussion_indicator"].notna().any() else "forward_discussion_indicator"
        part = part.sort_values(rank_col, ascending=False)
        best = part.iloc[0].copy()
        verified = part[_truthy_series(part["actual_available"])] if "actual_available" in part else pd.DataFrame()
        rows.append({
            "ticker": str(ticker).upper(),
            "indicator_max": float(part[rank_col].max()),
            "indicator_mean": float(part[rank_col].mean()),
            "discussion_indicator_max": float(part["forward_discussion_indicator"].max()) if "forward_discussion_indicator" in part else None,
            "topic_indicator_max": float(part["ticker_expected_topic_indicator"].max()) if "ticker_expected_topic_indicator" in part and part["ticker_expected_topic_indicator"].notna().any() else None,
            "topic_lift_max": float(part["ticker_expected_topic_lift"].max()) if "ticker_expected_topic_lift" in part and part["ticker_expected_topic_lift"].notna().any() else None,
            "topic_shift_max": float(part["ticker_expected_topic_shift"].max()) if "ticker_expected_topic_shift" in part and part["ticker_expected_topic_shift"].notna().any() else None,
            "topic_step_shift_max": float(part["ticker_expected_step_shift"].max()) if "ticker_expected_step_shift" in part and part["ticker_expected_step_shift"].notna().any() else None,
            "current_topic_exposure_max": float(part["ticker_current_topic_exposure"].max()) if "ticker_current_topic_exposure" in part and part["ticker_current_topic_exposure"].notna().any() else None,
            "expected_topic_exposure_max": float(part["ticker_expected_topic_exposure"].max()) if "ticker_expected_topic_exposure" in part and part["ticker_expected_topic_exposure"].notna().any() else None,
            "model_shift_indicator_max": float(part["model_shift_indicator"].max()) if "model_shift_indicator" in part and part["model_shift_indicator"].notna().any() else None,
            "event_pvalue": float(part["ticker_discussion_event_pvalue"].min()) if "ticker_discussion_event_pvalue" in part and part["ticker_discussion_event_pvalue"].notna().any() else None,
            "event_qvalue": float(part["ticker_discussion_event_qvalue"].min()) if "ticker_discussion_event_qvalue" in part and part["ticker_discussion_event_qvalue"].notna().any() else None,
            "growth_pvalue": float(part["ticker_growth_pvalue"].min()) if "ticker_growth_pvalue" in part and part["ticker_growth_pvalue"].notna().any() else None,
            "current_prominence_pvalue": float(part["ticker_current_prominence_pvalue"].min()) if "ticker_current_prominence_pvalue" in part and part["ticker_current_prominence_pvalue"].notna().any() else None,
            "context_prominence_pvalue": float(part["ticker_context_prominence_pvalue"].min()) if "ticker_context_prominence_pvalue" in part and part["ticker_context_prominence_pvalue"].notna().any() else None,
            "presence_pvalue": float(part["ticker_presence_pvalue"].min()) if "ticker_presence_pvalue" in part and part["ticker_presence_pvalue"].notna().any() else None,
            "activity_pvalue": float(part["ticker_activity_pvalue"].min()) if "ticker_activity_pvalue" in part and part["ticker_activity_pvalue"].notna().any() else None,
            "observed_support_pvalue": float(part["ticker_observed_support_pvalue"].min()) if "ticker_observed_support_pvalue" in part and part["ticker_observed_support_pvalue"].notna().any() else None,
            "predictive_evidence_pvalue": float(part["ticker_predictive_evidence_pvalue"].min()) if "ticker_predictive_evidence_pvalue" in part and part["ticker_predictive_evidence_pvalue"].notna().any() else None,
            "global_activity_rate_pvalue": float(part["ticker_global_activity_rate_pvalue"].min()) if "ticker_global_activity_rate_pvalue" in part and part["ticker_global_activity_rate_pvalue"].notna().any() else None,
            "model_pvalue": float(part["ticker_model_pvalue"].min()) if "ticker_model_pvalue" in part and part["ticker_model_pvalue"].notna().any() else None,
            "text_pvalue": float(part["ticker_text_pvalue"].min()) if "ticker_text_pvalue" in part and part["ticker_text_pvalue"].notna().any() else None,
            "best_step": int(best.get("future_step", 1) or 1),
            "ticker_share": float(part["ticker_share"].max()) if "ticker_share" in part else None,
            "activity_rate": float(part["ticker_activity_rate"].max()) if "ticker_activity_rate" in part else None,
            "activity_lift": float(part["ticker_activity_lift"].max()) if "ticker_activity_lift" in part and part["ticker_activity_lift"].notna().any() else None,
            "activity_robust_z": float(part["ticker_activity_robust_zscore"].max()) if "ticker_activity_robust_zscore" in part and part["ticker_activity_robust_zscore"].notna().any() else None,
            "attention_score": float(part["ticker_attention_score"].max()) if "ticker_attention_score" in part and part["ticker_attention_score"].notna().any() else None,
            "activity_anomaly_score": float(part["ticker_activity_anomaly_score"].max()) if "ticker_activity_anomaly_score" in part and part["ticker_activity_anomaly_score"].notna().any() else None,
            "support_score": float(part["ticker_support_score"].max()) if "ticker_support_score" in part and part["ticker_support_score"].notna().any() else None,
            "confidence_avg": float(part["ticker_confidence_avg"].mean()) if "ticker_confidence_avg" in part and part["ticker_confidence_avg"].notna().any() else None,
            "post_count": int(float(part["ticker_post_count"].max())) if "ticker_post_count" in part and part["ticker_post_count"].notna().any() else None,
            "unique_authors": int(float(part["ticker_unique_authors"].max())) if "ticker_unique_authors" in part and part["ticker_unique_authors"].notna().any() else None,
            "direct_post_count": int(float(part["ticker_direct_post_count"].max())) if "ticker_direct_post_count" in part and part["ticker_direct_post_count"].notna().any() else None,
            "inherited_post_count": int(float(part["ticker_inherited_post_count"].max())) if "ticker_inherited_post_count" in part and part["ticker_inherited_post_count"].notna().any() else None,
            "participation_mode": str(part["ticker_participation_mode"].dropna().iloc[0]) if "ticker_participation_mode" in part and part["ticker_participation_mode"].notna().any() else "direct_mentions",
            "current_post_count": int(float(part["ticker_current_post_count"].max())) if "ticker_current_post_count" in part and part["ticker_current_post_count"].notna().any() else None,
            "current_unique_authors": int(float(part["ticker_current_unique_authors"].max())) if "ticker_current_unique_authors" in part and part["ticker_current_unique_authors"].notna().any() else None,
            "context_post_count": int(float(part["ticker_context_post_count"].max())) if "ticker_context_post_count" in part and part["ticker_context_post_count"].notna().any() else None,
            "context_unique_authors": int(float(part["ticker_context_unique_authors"].max())) if "ticker_context_unique_authors" in part and part["ticker_context_unique_authors"].notna().any() else None,
            "context_share": float(part["ticker_context_share"].max()) if "ticker_context_share" in part and part["ticker_context_share"].notna().any() else None,
            "historical_support_score": float(part["ticker_historical_support_score"].max()) if "ticker_historical_support_score" in part and part["ticker_historical_support_score"].notna().any() else None,
            "context_direct_post_count": int(float(part["ticker_context_direct_post_count"].max())) if "ticker_context_direct_post_count" in part and part["ticker_context_direct_post_count"].notna().any() else None,
            "context_inherited_post_count": int(float(part["ticker_context_inherited_post_count"].max())) if "ticker_context_inherited_post_count" in part and part["ticker_context_inherited_post_count"].notna().any() else None,
            "source_scope": str(part["ticker_source_scope"].dropna().iloc[0]) if "ticker_source_scope" in part and part["ticker_source_scope"].notna().any() else "current_window",
            "sentiment": float(part["ticker_sentiment"].mean()) if "ticker_sentiment" in part else None,
            "sentiment_intensity": float(part["ticker_sentiment_intensity"].mean()) if "ticker_sentiment_intensity" in part else None,
            "avg_influence": float(part["ticker_avg_influence"].mean()) if "ticker_avg_influence" in part else None,
            "realized_nrmse_mean": float(verified["realized_nrmse"].dropna().mean()) if not verified.empty and "realized_nrmse" in verified else None,
            "ticker_realized_nrmse_mean": float(verified["ticker_realized_nrmse"].dropna().mean()) if not verified.empty and "ticker_realized_nrmse" in verified else None,
            "n_steps_verified": int(verified["future_step"].nunique()) if not verified.empty and "future_step" in verified else 0,
            "ticker_has_topic_profile": bool(_truthy_series(part["ticker_has_topic_profile"]).any()) if "ticker_has_topic_profile" in part else False,
            "direction_hint": _direction_russian(best.get("direction_hint")),
            "interpretation": str(best.get("interpretation", "")),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ["event_qvalue", "event_pvalue", "current_prominence_pvalue", "post_count", "ticker_share", "indicator_max", "discussion_indicator_max", "topic_indicator_max", "topic_lift_max", "topic_shift_max", "topic_step_shift_max", "current_topic_exposure_max", "expected_topic_exposure_max", "model_shift_indicator_max"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    sort_cols = [c for c in ["indicator_max", "topic_indicator_max", "event_qvalue", "event_pvalue", "post_count", "ticker_share"] if c in out.columns]
    ascending = [False, False, True, True, False, False][:len(sort_cols)]
    return out.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)


def _research_verdict(summary: pd.DataFrame) -> tuple[str, str, str]:
    """Формирует осторожную интерпретацию ведущего дискуссионного события
    
    Args:
        summary: таблица или словарь сводных показателей
    
    Returns:
        tuple[str, str, str]: заголовок, пояснение и визуальный тон исследовательской интерпретации
    """
    if summary is None or summary.empty:
        return "Нет тикерного индикатора", "В выбранном шаге моделирования не найдено устойчивых тикеров. Можно анализировать динамику дискуссии, но рыночную проекцию строить рано.", "gray"
    top = summary.iloc[0]
    ind = float(top.get("indicator_max", 0.0) or 0.0)
    nrmse = top.get("realized_nrmse_mean")
    activity = float(top.get("activity_rate", 0.0) or 0.0)
    lift = top.get("activity_lift")
    lift_text = f" Рост упоминаний к историческому контексту: {fmt(lift, 2)}x." if pd.notna(lift) else ""
    if ind >= 0.45 and (pd.isna(nrmse) or float(nrmse) <= 35.0):
        return "Есть исследовательский индикатор", f"В шаге моделирования есть выраженная тикерная концентрация, пользовательская активность или эмоциональное давление. Доля сообщений с тикером: {pct(activity, 1)}.{lift_text} Такое событие стоит проверять по цене и объёму.", "green"
    if ind >= 0.25:
        return "Индикатор умеренный", f"Шаг может быть полезным для мониторинга: часть признаков отличается от фоновой динамики, но перед выводами нужно смотреть публикации и фактическое движение тикера после шага моделирования. Доля сообщений с тикером: {pct(activity, 1)}.{lift_text}", "orange"
    return "Индикатор слабый", "Дискуссия не даёт сильного рыночного индикатора. Подход в этом шаге моделирования скорее работает как фильтр: событие можно не приоритизировать.", "gray"
