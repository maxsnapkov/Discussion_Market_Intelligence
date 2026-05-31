"""Расчёт дискуссионных индикаторов и модельно-тематической поддержки"""
from __future__ import annotations


from ..pipeline.common import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_ingestion.preparation import _window_ticker_analytics
    from ..pipeline.configuration import is_service_author
    from ..simulation.modeling import _safe_actual_range

def _direction_label(avg_sentiment: float) -> str:
    """Возвращает текстовое направление тональности
    
    Args:
        avg_sentiment: средняя тональность
    
    Returns:
        подпись направления тональности
    """
    if avg_sentiment >= 0.15:
        return "позитивный / bullish"
    if avg_sentiment <= -0.15:
        return "негативный / bearish"
    return "нейтральный или смешанный"


def _indicator_reasons(row: dict[str, Any]) -> str:
    """Формирует пояснения причин высокого дискуссионного индикатора
    
    Args:
        row: строка таблицы с сообщением, инструментом или событием
    
    Returns:
        список текстовых причин значения индикатора
    """
    reasons = []
    qvalue = row.get("ticker_discussion_event_qvalue")
    p_event = row.get("ticker_discussion_event_pvalue")
    indicator = row.get("forward_discussion_indicator")
    combined = row.get("combined_forward_discussion_indicator")
    model_indicator = row.get("ticker_expected_topic_indicator")
    support_gate = row.get("ticker_current_support_gate")
    if indicator is not None and not pd.isna(indicator):
        reasons.append(f"текущая необычность обсуждения: {_finite_float(indicator, 0.0):.4f}")
    if model_indicator is not None and not pd.isna(model_indicator) and _finite_float(model_indicator, 0.0) > 0:
        reasons.append(f"ожидаемая тематическая поддержка: {_finite_float(model_indicator, 0.0):.4f}")
    if combined is not None and not pd.isna(combined):
        reasons.append(f"комбинированный индикатор: {_finite_float(combined, 0.0):.4f}")
    if support_gate is not None and not pd.isna(support_gate):
        reasons.append(f"текущая поддержка тикера: {_finite_float(support_gate, 0.0):.2f}")
    if p_event is not None and not pd.isna(p_event):
        reasons.append(f"p-value дискуссионного события: {_finite_float(p_event, 1.0):.4f}")
    if qvalue is not None and not pd.isna(qvalue):
        reasons.append(f"q-value FDR-контроля: {_finite_float(qvalue, 1.0):.4f}")
    component_labels = [
        ("ticker_activity_pvalue", "активность обсуждения"),
        ("ticker_posts_pvalue", "число сообщений"),
        ("ticker_authors_pvalue", "число авторов"),
        ("ticker_thread_pvalue", "ветвление обсуждения"),
        ("ticker_model_pvalue", "модельная тематическая траектория"),
        ("ticker_text_pvalue", "текстовая интенсивность"),
    ]
    for col, label in component_labels:
        value = row.get(col)
        if value is not None and not pd.isna(value) and _finite_float(value, 1.0) <= 0.15:
            reasons.append(f"{label}: p={_finite_float(value, 1.0):.4f}")
    if row.get("has_future_fact") is False:
        reasons.append("фактическая проверка будущих шагов дискретизации пока недоступна")
    return "; ".join(reasons) if reasons else "симуляционный индикатор без статистически выраженного компонента"

FORWARD_INDICATOR_COLUMNS = [
    "start_index", "start_time", "future_step", "future_time", "ticker",
    "ticker_share", "ticker_sentiment", "ticker_sentiment_intensity",
    "ticker_avg_influence", "predicted_topic_drift",
    "predicted_topic_concentration", "ticker_topic_exposure",
    "model_expected_topic_shift", "model_shift_pvalue", "model_shift_surprisal", "model_shift_indicator",
    "ticker_current_topic_exposure", "ticker_expected_topic_exposure", "ticker_expected_topic_lift",
    "ticker_expected_topic_shift", "ticker_expected_step_shift", "ticker_expected_topic_pvalue",
    "ticker_expected_topic_surprisal", "ticker_expected_topic_indicator",
    "forward_discussion_indicator", "combined_forward_discussion_indicator",
    "ticker_event_surprisal", "ticker_indicator_scale", "ticker_current_support_gate", "ticker_fdr_significant",
    "direction_hint", "actual_available",
    "has_future_fact", "forecast_only", "validation_status",
    "analysis_mode", "source_mode", "window_status", "window_complete",
    "planned_end_time", "observed_until", "raw_source_path", "raw_source_mtime", "raw_source_size",
    "context_windows", "topic_rebuild_mode", "no_future_leakage",
    "realized_nrmse", "realized_l1", "realized_rmse",
    "ticker_realized_nrmse", "ticker_realized_l1", "ticker_realized_rmse",
    "ticker_model_step_shift", "ticker_model_from_start_shift",
    "ticker_has_topic_profile",
    "ticker_post_count", "ticker_unique_authors", "ticker_activity_rate", "ticker_author_rate",
    "ticker_activity_lift", "ticker_activity_zscore", "ticker_activity_robust_zscore",
    "ticker_baseline_posts", "ticker_baseline_median_posts", "ticker_attention_score",
    "ticker_activity_anomaly_score", "ticker_support_score", "ticker_confidence_avg", "ticker_context_score_avg",
    "ticker_current_post_count", "ticker_current_unique_authors", "ticker_context_post_count", "ticker_context_unique_authors",
    "ticker_context_share", "ticker_historical_support_score",
    "ticker_posts_pvalue", "ticker_authors_pvalue", "ticker_thread_pvalue",
    "ticker_global_posts_pvalue", "ticker_global_authors_pvalue", "ticker_global_thread_pvalue", "ticker_global_activity_rate_pvalue",
    "ticker_observed_support_pvalue", "ticker_predictive_evidence_pvalue",
    "ticker_growth_pvalue", "ticker_current_prominence_pvalue", "ticker_context_prominence_pvalue", "ticker_presence_pvalue",
    "ticker_sentiment_pvalue", "ticker_influence_pvalue", "ticker_model_pvalue",
    "ticker_activity_pvalue", "ticker_text_pvalue", "ticker_topic_pvalue",
    "ticker_discussion_event_pvalue", "ticker_discussion_event_qvalue",
    "ticker_source_scope", "ticker_ner_methods",
    "ticker_direct_post_count", "ticker_inherited_post_count", "ticker_root_thread_count", "ticker_participation_mode",
    "ticker_context_direct_post_count", "ticker_context_inherited_post_count", "ticker_context_root_thread_count",
    "interpretation",
]

DISCUSSION_LEVEL_TICKER = "__DISCUSSION__"


def _discussion_label() -> str:
    """Возвращает служебную подпись строки общего дискуссионного уровня
    
    Returns:
        служебную подпись строки общего дискуссионного уровня
    """
    return DISCUSSION_LEVEL_TICKER


def _indicator_metadata_from_window(start_window: dict, actual_available: bool) -> dict[str, Any]:
    """Собирает метаданные шага дискретизации для таблицы индикаторов
    
    Args:
        start_window: строка или индекс стартового шага дискретизации
        actual_available: признак доступности фактического будущего распределения
    
    Returns:
        словарь метаданных шага дискретизации
    """
    has_fact = bool(actual_available)
    return {
        "has_future_fact": has_fact,
        "forecast_only": not has_fact,
        "validation_status": "available" if has_fact else "awaiting_future_data",
        "analysis_mode": start_window.get("analysis_mode", "window_forecast"),
        "source_mode": start_window.get("source_mode", "precomputed_windows"),
        "window_status": start_window.get("window_status", "observed_window"),
        "window_complete": bool(start_window.get("window_complete", True)),
        "planned_end_time": start_window.get("planned_end_time") or start_window.get("window_end"),
        "observed_until": start_window.get("observed_until") or start_window.get("planned_end_time") or start_window.get("window_end"),
        "raw_source_path": start_window.get("raw_source_path"),
        "raw_source_mtime": start_window.get("raw_source_mtime"),
        "raw_source_size": start_window.get("raw_source_size"),
        "context_windows": start_window.get("context_windows"),
        "topic_rebuild_mode": start_window.get("topic_rebuild_mode", "precomputed_topic_space"),
        "no_future_leakage": bool(start_window.get("no_future_leakage", True)),
    }


def _step_to_arrays(sim: pd.DataFrame, prop_cols: list[str], step: int) -> np.ndarray | None:
    """Достаёт фактический и смоделированный векторы тем для шага проверки
    
    Args:
        sim: результат агентной симуляции
        prop_cols: колонки долей тем
        step: шаг изменения значения или номер шага анализа
    
    Returns:
        пара фактического и смоделированного векторов тем
    """
    row = sim[sim["step"] == step]
    if row.empty:
        return None
    return row.iloc[-1][prop_cols].astype(float).to_numpy()


def _align_arrays(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Выравнивает два числовых вектора по общей длине
    
    Args:
        a: первый числовой вектор
        b: второй числовой вектор
    
    Returns:
        пара выровненных числовых векторов
    """
    n = min(len(a), len(b))
    return a[:n], b[:n]


def _finite_float(value: Any, default: float = 0.0) -> float:
    """Приводит значение к конечному числу с резервным значением
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
        default: резервное значение
    
    Returns:
        конечное число или резервное значение
    """
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _finite_series(values: Any) -> np.ndarray:
    """Оставляет в последовательности только конечные числовые значения
    
    Args:
        values: последовательность числовых или строковых значений
    
    Returns:
        numpy-массив конечных числовых значений
    """
    if values is None:
        return np.asarray([], dtype=float)
    try:
        arr = np.asarray(list(values), dtype=float)
    except Exception:
        return np.asarray([], dtype=float)
    return arr[np.isfinite(arr)]


def _empirical_upper_pvalue(current: Any, history: Any) -> float | None:
    """Рассчитывает эмпирическое p-значение верхнего хвоста
    
    Args:
        current: текущее значение
        history: исторические значения для сравнения
    
    Returns:
        эмпирическое p-значение верхнего хвоста
    """
    current_value = _finite_float(current, float("nan"))
    hist = _finite_series(history)
    if not math.isfinite(current_value) or len(hist) == 0:
        return None
    return float((1.0 + float(np.sum(hist >= current_value))) / (float(len(hist)) + 1.0))


def _robust_zscore(current: Any, history: Any) -> float | None:
    """Рассчитывает устойчивый z-score через медиану и MAD
    
    Args:
        current: текущее значение
        history: исторические значения для сравнения
    
    Returns:
        устойчивый z-score текущего значения
    """
    current_value = _finite_float(current, float("nan"))
    hist = _finite_series(history)
    if not math.isfinite(current_value) or len(hist) == 0:
        return None
    median = float(np.median(hist))
    mad = float(np.median(np.abs(hist - median)))
    scale = 1.4826 * mad
    if scale <= 0:
        scale = max(math.sqrt(max(median, 0.0) + 1.0), 1.0)
    return float((current_value - median) / scale)


def _empirical_baseline(values: Any) -> tuple[float, float]:
    """Возвращает медиану и верхний квартиль исторического ряда
    
    Args:
        values: последовательность числовых или строковых значений
    
    Returns:
        пара медианы и верхнего квартиля
    """
    hist = _finite_series(values)
    if len(hist) == 0:
        return 0.0, 0.0
    return float(np.mean(hist)), float(np.median(hist))


def _combine_pvalues_fisher(pvalues: list[float | None]) -> float | None:
    """Объединяет p-значения методом Фишера
    
    Args:
        pvalues: список p-значений
    
    Returns:
        объединённое p-значение метода Фишера
    """
    vals = []
    for pvalue in pvalues:
        if pvalue is None:
            continue
        pv = _finite_float(pvalue, float("nan"))
        if math.isfinite(pv):
            vals.append(min(max(pv, 1e-12), 1.0))
    if not vals:
        return None
    stat = -2.0 * float(np.sum(np.log(vals)))
    dof = 2 * len(vals)
    try:
        from scipy.stats import chi2
        return float(chi2.sf(stat, dof))
    except Exception:
        return float(min(1.0, math.exp(-0.5 * stat)))


def _combine_pvalues_tippett(pvalues: list[float | None]) -> float | None:
    """Объединяет p-значения по минимальному значению с поправкой
    
    Args:
        pvalues: список p-значений
    
    Returns:
        объединённое p-значение метода Типпетта
    """
    vals = []
    for pvalue in pvalues:
        if pvalue is None:
            continue
        pv = _finite_float(pvalue, float("nan"))
        if math.isfinite(pv):
            vals.append(min(max(pv, 1e-12), 1.0))
    if not vals:
        return None
    min_p = min(vals)
    return float(max(0.0, min(1.0, 1.0 - (1.0 - min_p) ** len(vals))))


def _cross_section_upper_pvalues(values: list[Any]) -> list[float | None]:
    """Рассчитывает p-значения верхнего хвоста по срезу инструментов
    
    Args:
        values: последовательность числовых или строковых значений
    
    Returns:
        список p-значений по поперечному срезу
    """
    arr = np.asarray([_finite_float(v, float("nan")) for v in values], dtype=float)
    out: list[float | None] = []
    n = int(np.isfinite(arr).sum())
    if n <= 1:
        return [None for _ in values]
    for i, current in enumerate(arr):
        if not math.isfinite(float(current)):
            out.append(None)
            continue
        others = np.delete(arr[np.isfinite(arr)], np.where(np.where(np.isfinite(arr))[0] == i)[0]) if np.isfinite(arr[i]) else arr[np.isfinite(arr)]
        out.append(float((1.0 + float(np.sum(others >= current))) / (float(len(others)) + 1.0)))
    return out


def _bh_qvalues(pvalues: list[float | None]) -> list[float | None]:
    """Рассчитывает q-значения Бенджамини Хохберга
    
    Args:
        pvalues: список p-значений
    
    Returns:
        список q-значений Бенджамини Хохберга
    """
    indexed = [(i, _finite_float(pv, float("nan"))) for i, pv in enumerate(pvalues)]
    valid = [(i, pv) for i, pv in indexed if math.isfinite(pv)]
    if not valid:
        return [None for _ in pvalues]
    order = sorted(valid, key=lambda x: x[1])
    m = float(len(order))
    qvals = [None for _ in pvalues]
    running = 1.0
    for rank, (idx, pv) in reversed(list(enumerate(order, start=1))):
        running = min(running, pv * m / float(rank))
        qvals[idx] = float(min(max(running, 0.0), 1.0))
    return qvals


def _event_surprisal(pvalue: float | None) -> float:
    """Преобразует p-значение в информационную меру необычности
    
    Args:
        pvalue: p-значение
    
    Returns:
        информационная мера необычности события
    """
    if pvalue is None:
        return 0.0
    pv = min(max(_finite_float(pvalue, 1.0), 1e-12), 1.0)
    return float(max(0.0, -math.log(pv)))


def _indicator_scale_from_context(context_windows: Any = None, fallback_count: Any = None) -> float:
    """Подбирает масштаб нормировки по глубине исторического контекста
    
    Args:
        context_windows: число предыдущих шагов дискретизации для исторического контекста
        fallback_count: резервная глубина контекста для выбора масштаба
    
    Returns:
        масштаб нормировки индикатора
    """
    for item in (context_windows, fallback_count):
        value = _finite_float(item, float("nan"))
        if math.isfinite(value) and value > 0:
            return float(max(math.log1p(value), -math.log(0.20)))
    return float(-math.log(0.10))


def _event_strength_from_pvalue(pvalue: float | None, context_windows: Any = None, fallback_count: Any = None) -> float:
    """Преобразует p-значение в нормированную силу события
    
    Args:
        pvalue: p-значение
        context_windows: число предыдущих шагов дискретизации для исторического контекста
        fallback_count: резервная глубина контекста для выбора масштаба
    
    Returns:
        нормированная сила события в диапазоне от нуля до единицы
    """
    surprisal = _event_surprisal(pvalue)
    if surprisal <= 0.0:
        return 0.0
    scale = _indicator_scale_from_context(context_windows, fallback_count)
    return float(max(0.0, min(1.0, surprisal / (surprisal + scale))))


def _ticker_current_support_gate_from_counts(post_count: Any, unique_authors: Any, root_thread_count: Any) -> float:
    """Рассчитывает фильтр поддержки по сообщениям, авторам и ветвям
    
    Args:
        post_count: число сообщений по инструменту
        unique_authors: число уникальных авторов по инструменту
        root_thread_count: число независимых ветвей обсуждения по инструменту
    
    Returns:
        значение фильтра минимальной поддержки
    """
    posts = max(0.0, _finite_float(post_count, 0.0))
    authors = max(0.0, _finite_float(unique_authors, 0.0))
    threads = max(0.0, _finite_float(root_thread_count, 0.0))
    if posts <= 0.0 or authors <= 0.0:
        return 0.0
    if threads <= 0.0:
        # если идентификаторы ветвей недоступны, сохраняем слабый сигнал одной ветви
        # вместо полного удаления тикера
        threads = 1.0
    posts_gate = min(1.0, posts / 3.0)
    authors_gate = min(1.0, authors / 2.0)
    thread_gate = min(1.0, threads / 2.0)
    return float(max(0.0, min(posts_gate, authors_gate, thread_gate)))


def _ticker_current_support_gate(row: dict[str, Any] | pd.Series) -> float:
    """Рассчитывает фильтр поддержки для строки финансового инструмента
    
    Args:
        row: строка таблицы с сообщением, инструментом или событием
    
    Returns:
        значение фильтра поддержки для строки тикера
    """
    getter = row.get if hasattr(row, "get") else (lambda _k, d=None: d)
    posts = getter("ticker_current_post_count", getter("ticker_post_count", getter("post_count", 0)))
    authors = getter("ticker_current_unique_authors", getter("ticker_unique_authors", getter("unique_authors", 0)))
    threads = getter("ticker_root_thread_count", getter("root_thread_count", 0))
    return _ticker_current_support_gate_from_counts(posts, authors, threads)


def _ticker_indicator_from_pvalue(pvalue: float | None, row: dict[str, Any] | pd.Series, context_windows: Any = None) -> float:
    """Получает текущую необычность тикера из p-значения и поддержки
    
    Args:
        pvalue: p-значение
        row: строка таблицы с сообщением, инструментом или событием
        context_windows: число предыдущих шагов дискретизации для исторического контекста
    
    Returns:
        текущая статистическая необычность тикера
    """
    support_gate = _ticker_current_support_gate(row)
    if support_gate <= 0.0:
        return 0.0
    fallback_count = row.get("ticker_context_post_count", row.get("ticker_post_count", None)) if hasattr(row, "get") else None
    base = _event_strength_from_pvalue(pvalue, context_windows=context_windows, fallback_count=fallback_count)
    return float(max(0.0, min(1.0, base * support_gate)))


def _safe_prob_vector(values: Any) -> np.ndarray:
    """Нормирует вектор в вероятностное распределение с безопасным fallback
    
    Args:
        values: последовательность числовых или строковых значений
    
    Returns:
        вероятностный вектор, сумма которого равна единице
    """
    arr = _finite_series(values)
    if len(arr) == 0:
        return arr
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0:
        return np.ones(len(arr), dtype=float) / float(len(arr))
    return arr / total


def _topic_l1_shift(a: Any, b: Any) -> float | None:
    """Рассчитывает L1-сдвиг между двумя тематическими распределениями
    
    Args:
        a: первый числовой вектор
        b: второй числовой вектор
    
    Returns:
        L1-сдвиг между тематическими распределениями
    """
    avec = _safe_prob_vector(a)
    bvec = _safe_prob_vector(b)
    n = min(len(avec), len(bvec))
    if n <= 0:
        return None
    return float(np.sum(np.abs(avec[:n] - bvec[:n])))


def _topic_shift_history_from_windows(windows: list[dict[str, Any]], start_index: int, context_windows: int = 8) -> list[float]:
    """Извлекает историю тематических сдвигов из таблицы шагов дискретизации
    
    Args:
        windows: таблица шагов дискретизации
        start_index: индекс начального шага дискретизации
        context_windows: число предыдущих шагов дискретизации для исторического контекста
    
    Returns:
        извлечённые значения: историю тематических сдвигов из таблицы шагов дискретизации
    """
    if not windows or int(start_index) <= 0:
        return []
    begin = max(0, int(start_index) - max(1, int(context_windows)))
    hist = windows[begin:int(start_index) + 1]
    shifts: list[float] = []
    for prev, cur in zip(hist, hist[1:]):
        shift = _topic_l1_shift(prev.get("real_props", []), cur.get("real_props", []))
        if shift is not None and math.isfinite(shift):
            shifts.append(float(shift))
    return shifts


def _topic_shift_history_from_context_df(context_df: pd.DataFrame, start_ts: Any, window_hours: int) -> list[float]:
    """Строит историю тематических сдвигов из контекстных сообщений
    
    Args:
        context_df: таблица сообщений исторического контекста
        start_ts: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
    
    Returns:
        построенная структура: историю тематических сдвигов из контекстных сообщений
    """
    if context_df is None or pd.DataFrame(context_df).empty or "timestamp" not in pd.DataFrame(context_df).columns:
        return []
    data = pd.DataFrame(context_df).copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data[data["timestamp"] < pd.Timestamp(start_ts)].dropna(subset=["timestamp"]).copy()
    if data.empty or "topic" not in data.columns:
        return []
    data["_window_start"] = data["timestamp"].dt.floor(f"{int(window_hours)}h")
    topic_ids = sorted([str(x) for x in data["topic"].dropna().astype(int).unique().tolist()])
    if not topic_ids:
        return []
    props: list[np.ndarray] = []
    for _, part in data.groupby("_window_start", sort=True):
        dist = _topic_distribution_for_part(part, topic_ids)
        if len(dist):
            props.append(dist)
    shifts: list[float] = []
    for prev, cur in zip(props, props[1:]):
        shift = _topic_l1_shift(prev, cur)
        if shift is not None and math.isfinite(shift):
            shifts.append(float(shift))
    return shifts


def _model_shift_fields(expected_shift: float, history: Any, context_windows: Any = None) -> dict[str, Any]:
    """Формирует поля индикатора ожидаемого тематического сдвига
    
    Args:
        expected_shift: ожидаемый тематический сдвиг
        history: исторические значения для сравнения
        context_windows: число предыдущих шагов дискретизации для исторического контекста
    
    Returns:
        словарь полей индикатора ожидаемого тематического сдвига
    """
    hist = _finite_series(history)
    pvalue = _empirical_upper_pvalue(expected_shift, hist)
    return {
        "model_expected_topic_shift": float(max(0.0, _finite_float(expected_shift, 0.0))),
        "model_shift_pvalue": pvalue,
        "model_shift_surprisal": _event_surprisal(pvalue),
        "model_shift_indicator": _event_strength_from_pvalue(pvalue, context_windows=context_windows, fallback_count=len(hist)),
    }


def _ticker_expected_topic_fields(
    info: dict[str, Any],
    weights: np.ndarray,
    baseline: np.ndarray,
    pred: np.ndarray,
    step_shift_vec: np.ndarray,
    from_start_vec: np.ndarray,
    has_topic_profile: bool,
    context_windows: Any = None,
) -> dict[str, Any]:
    """Рассчитывает ожидаемую тематическую экспозицию финансового инструмента
    
    Args:
        info: словарь признаков финансового инструмента
        weights: веса тем или компонентов расчёта
        baseline: базовое или текущее тематическое распределение
        pred: смоделированное тематическое распределение
        step_shift_vec: вектор изменения тем на одном шаге симуляции
        from_start_vec: вектор изменения тем от начального состояния
        has_topic_profile: признак наличия тематического профиля инструмента
        context_windows: число предыдущих шагов дискретизации для исторического контекста
    
    Returns:
        словарь полей модельно-тематической поддержки инструмента
    """
    hist = _finite_series(info.get("model_exposure_history", []))
    if not has_topic_profile or len(weights) == 0:
        pvalue = None
        return {
            "ticker_current_topic_exposure": 0.0,
            "ticker_expected_topic_exposure": 0.0,
            "ticker_expected_topic_lift": 0.0,
            "ticker_expected_topic_shift": 0.0,
            "ticker_expected_step_shift": 0.0,
            "ticker_expected_topic_pvalue": pvalue,
            "ticker_expected_topic_surprisal": 0.0,
            "ticker_expected_topic_indicator": 0.0,
        }
    n = min(len(weights), len(baseline), len(pred), len(step_shift_vec), len(from_start_vec))
    w = np.asarray(weights[:n], dtype=float)
    wsum = float(w.sum())
    if wsum <= 0:
        w = np.ones(n, dtype=float) / max(1, n)
    else:
        w = w / wsum
    cur_exp = float(np.sum(w * np.asarray(baseline[:n], dtype=float)))
    pred_exp = float(np.sum(w * np.asarray(pred[:n], dtype=float)))
    lift = float(pred_exp - cur_exp)
    abs_shift = float(np.sum(w * np.asarray(from_start_vec[:n], dtype=float)))
    step_shift = float(np.sum(w * np.asarray(step_shift_vec[:n], dtype=float)))
    current_value = max(0.0, lift, abs_shift)
    pvalue = _empirical_upper_pvalue(current_value, hist)
    raw_indicator = _event_strength_from_pvalue(pvalue, context_windows=context_windows, fallback_count=len(hist))
    support_gate = _ticker_current_support_gate_from_counts(
        info.get("post_count", info.get("current_post_count", 0)),
        info.get("unique_authors", info.get("current_unique_authors", 0)),
        info.get("root_thread_count", 0),
    )
    return {
        "ticker_current_topic_exposure": cur_exp,
        "ticker_expected_topic_exposure": pred_exp,
        "ticker_expected_topic_lift": lift,
        "ticker_expected_topic_shift": abs_shift,
        "ticker_expected_step_shift": step_shift,
        "ticker_expected_topic_pvalue": pvalue,
        "ticker_expected_topic_surprisal": _event_surprisal(pvalue),
        "ticker_expected_topic_indicator": float(max(0.0, min(1.0, raw_indicator * support_gate))),
    }


def _combined_forward_indicator(stat_indicator: Any, topic_indicator: Any, stat_weight: float = 0.70) -> float:
    """Объединяет текущую необычность и модельно-тематическую поддержку
    
    Args:
        stat_indicator: текущая статистическая необычность инструмента
        topic_indicator: модельно-тематическая поддержка инструмента
        stat_weight: вес текущей статистической необычности в итоговом индикаторе
    
    Returns:
        комбинированный дискуссионный индикатор
    """
    w = min(1.0, max(0.0, _finite_float(stat_weight, 0.70)))
    stat = max(0.0, min(1.0, _finite_float(stat_indicator, 0.0)))
    topic = max(0.0, min(1.0, _finite_float(topic_indicator, 0.0)))
    return float(max(0.0, min(1.0, w * stat + (1.0 - w) * topic)))


def _jensen_shannon_distance(p: Any, q: Any) -> float | None:
    """Рассчитывает расстояние Дженсена Шеннона между распределениями
    
    Args:
        p: путь или вероятностное значение в зависимости от контекста вызова
        q: второе вероятностное распределение
    
    Returns:
        расстояние Дженсена Шеннона
    """
    pvec = _safe_prob_vector(p)
    qvec = _safe_prob_vector(q)
    n = min(len(pvec), len(qvec))
    if n == 0:
        return None
    pvec = pvec[:n]
    qvec = qvec[:n]
    m = 0.5 * (pvec + qvec)
    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        """Рассчитывает KL-слагаемое для расстояния Дженсена Шеннона
        
        Args:
            a: первый числовой вектор
            b: второй числовой вектор
        
        Returns:
            числовое KL-слагаемое для пары распределений
        """
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))
    jsd = 0.5 * _kl(pvec, m) + 0.5 * _kl(qvec, m)
    return float(math.sqrt(max(0.0, jsd)))


def _max_available_pvalue(pvalues: list[float | None]) -> float | None:
    """Выбирает наименее экстремальное доступное p-значение из набора
    
    Args:
        pvalues: список p-значений
    
    Returns:
        выбранное значение или подтаблица: наименее экстремальное доступное p-значение из набора
    """
    vals: list[float] = []
    for pvalue in pvalues:
        pv = _finite_float(pvalue, float("nan"))
        if math.isfinite(pv):
            vals.append(min(max(pv, 1e-12), 1.0))
    if not vals:
        return None
    return float(max(vals))


def _ticker_discussion_pvalues(info: dict[str, Any], model_pvalue: float | None = None) -> dict[str, Any]:
    """Рассчитывает p-значения компонент тикерного дискуссионного события
    
    Args:
        info: словарь признаков финансового инструмента
        model_pvalue: p-значение модельного тематического сдвига
    
    Returns:
        словарь p-значений компонент тикерного события
    """
    p_posts = info.get("p_posts", info.get("ticker_posts_pvalue"))
    p_authors = info.get("p_authors", info.get("ticker_authors_pvalue"))
    p_thread = info.get("p_thread", info.get("ticker_thread_pvalue"))
    p_sentiment = info.get("p_sentiment", info.get("ticker_sentiment_pvalue"))
    p_influence = info.get("p_influence", info.get("ticker_influence_pvalue"))
    p_current_prominence = info.get("p_current_prominence", info.get("ticker_current_prominence_pvalue"))
    p_context_prominence = info.get("p_context_prominence", info.get("ticker_context_prominence_pvalue"))
    p_presence = info.get("p_presence", info.get("ticker_presence_pvalue"))
    p_global_posts = info.get("p_global_posts", info.get("ticker_global_posts_pvalue"))
    p_global_authors = info.get("p_global_authors", info.get("ticker_global_authors_pvalue"))
    p_global_thread = info.get("p_global_thread", info.get("ticker_global_thread_pvalue"))
    p_global_activity_rate = info.get("p_global_activity_rate", info.get("ticker_global_activity_rate_pvalue"))
    p_observed_support = info.get("p_observed_support", info.get("ticker_observed_support_pvalue"))
    p_model = model_pvalue if model_pvalue is not None else info.get("p_model")

    p_growth = _combine_pvalues_fisher([p_posts, p_authors, p_thread])
    if p_presence is None:
        p_presence = _combine_pvalues_tippett([p_current_prominence, p_context_prominence])
    if p_observed_support is None:
        p_observed_support = _max_available_pvalue([p_global_posts, p_global_authors, p_global_thread, p_global_activity_rate])
    p_text = _combine_pvalues_fisher([p_sentiment, p_influence])
    p_topic = _combine_pvalues_fisher([p_model])

    # метод Фишера объединяет каналы подтверждения без ручных весов
    # максимум с observed support работает как правило intersection-union
    # событие не может быть значимее подтверждения того, что тикер реально поддержан
    # достаточной текущей активностью
    p_predictive = _combine_pvalues_fisher([p_current_prominence, p_topic, p_text])
    if p_observed_support is None:
        p_event = p_predictive
    elif p_predictive is None:
        p_event = p_observed_support
    else:
        p_event = float(max(_finite_float(p_observed_support, 1.0), _finite_float(p_predictive, 1.0)))
    p_activity = p_observed_support
    return {
        "ticker_posts_pvalue": p_posts,
        "ticker_authors_pvalue": p_authors,
        "ticker_thread_pvalue": p_thread,
        "ticker_global_posts_pvalue": p_global_posts,
        "ticker_global_authors_pvalue": p_global_authors,
        "ticker_global_thread_pvalue": p_global_thread,
        "ticker_global_activity_rate_pvalue": p_global_activity_rate,
        "ticker_observed_support_pvalue": p_observed_support,
        "ticker_predictive_evidence_pvalue": p_predictive,
        "ticker_sentiment_pvalue": p_sentiment,
        "ticker_influence_pvalue": p_influence,
        "ticker_current_prominence_pvalue": p_current_prominence,
        "ticker_context_prominence_pvalue": p_context_prominence,
        "ticker_presence_pvalue": p_presence,
        "ticker_growth_pvalue": p_growth,
        "ticker_model_pvalue": p_model,
        "ticker_activity_pvalue": p_activity,
        "ticker_text_pvalue": p_text,
        "ticker_topic_pvalue": p_topic,
        "ticker_discussion_event_pvalue": p_event,
    }

def _discussion_score(step_shift: float, from_start_shift: float, concentration: float, sentiment_intensity: float, avg_influence: float) -> float:
    """Рассчитывает общий скор изменения состояния дискуссии
    
    Args:
        step_shift: сдвиг тематического распределения на шаге симуляции
        from_start_shift: сдвиг тематического распределения относительно начального состояния
        concentration: концентрация тематического распределения или внимания
        sentiment_intensity: эмоциональная интенсивность обсуждения
        avg_influence: средняя авторитетность участников
    
    Returns:
        нормированный скор общего состояния дискуссии
    """
    # L1-расстояние между вероятностными распределениями ограничено значением 2
    return float(max(0.0, min(1.0, max(_finite_float(step_shift) / 2.0, _finite_float(from_start_shift) / 2.0))))


def _ticker_interest_score(
    ticker_exposure: float,
    ticker_share: float,
    sentiment_intensity: float,
    avg_influence: float,
    step_shift: float,
    activity_rate: float = 0.0,
    activity_lift: float | None = None,
    post_count: int = 0,
    unique_authors: int = 0,
    root_thread_count: int = 0,
    ticker_from_start: float = 0.0,
    activity_anomaly_score: float = 0.0,
    attention_score: float = 0.0,
    support_score: float = 0.0,
    confidence_avg: float = 0.0,
    historical_support_score: float = 0.0,
    context_share: float = 0.0,
    ticker: str | None = None,
    component_pvalues: dict[str, Any] | None = None,
    model_exposure_history: list[float] | None = None,
    context_windows: Any = None,
) -> float:
    """Рассчитывает итоговую силу дискуссионного события финансового инструмента
    
    Args:
        ticker_exposure: текущая тематическая экспозиция инструмента
        ticker_share: доля сообщений инструмента
        sentiment_intensity: эмоциональная интенсивность обсуждения
        avg_influence: средняя авторитетность участников
        step_shift: сдвиг тематического распределения на шаге симуляции
        activity_rate: доля активности инструмента в текущем шаге
        activity_lift: отношение активности к историческому контексту
        post_count: число сообщений по инструменту
        unique_authors: число уникальных авторов по инструменту
        root_thread_count: число независимых ветвей обсуждения по инструменту
        ticker_from_start: изменение экспозиции инструмента от начального состояния
        activity_anomaly_score: скор аномальности активности инструмента
        attention_score: скор концентрации внимания к инструменту
        support_score: скор минимальной дискуссионной поддержки
        confidence_avg: средняя уверенность извлечения инструмента
        historical_support_score: историческая поддержка инструмента
        context_share: доля инструмента в историческом контексте
        ticker: биржевой идентификатор финансового инструмента
        component_pvalues: p-значения компонент дискуссионного события
        model_exposure_history: история модельной тематической экспозиции
        context_windows: число предыдущих шагов дискретизации для исторического контекста
    
    Returns:
        нормированный скор дискуссионного события инструмента
    """
    component_pvalues = component_pvalues or {}
    model_current = max(_finite_float(ticker_exposure), _finite_float(ticker_from_start))
    model_pvalue = _empirical_upper_pvalue(model_current, model_exposure_history)
    pvals = _ticker_discussion_pvalues(component_pvalues, model_pvalue=model_pvalue)
    row = dict(component_pvalues)
    row.setdefault("ticker_post_count", post_count)
    row.setdefault("ticker_current_post_count", post_count)
    row.setdefault("ticker_unique_authors", unique_authors)
    row.setdefault("ticker_current_unique_authors", unique_authors)
    row.setdefault("ticker_root_thread_count", root_thread_count)
    return _ticker_indicator_from_pvalue(pvals.get("ticker_discussion_event_pvalue"), row, context_windows=context_windows)


def _activity_anomaly_score(lift: float | None, robust_z: float | None) -> float:
    """Рассчитывает вклад аномальной активности инструмента
    
    Args:
        lift: отношение текущего значения к историческому уровню
        robust_z: устойчивый z-score
    
    Returns:
        скор аномальной активности инструмента
    """
    if robust_z is not None and not (isinstance(robust_z, float) and math.isnan(robust_z)):
        return float(max(0.0, _finite_float(robust_z)))
    if lift is not None and not (isinstance(lift, float) and math.isnan(lift)):
        return float(max(0.0, _finite_float(lift) - 1.0))
    return 0.0


def _attention_score(activity_rate: float, ticker_share: float) -> float:
    """Рассчитывает скор внимания к инструменту
    
    Args:
        activity_rate: доля активности инструмента в текущем шаге
        ticker_share: доля сообщений инструмента
    
    Returns:
        скор концентрации внимания
    """
    return float(max(0.0, min(1.0, max(_finite_float(activity_rate), _finite_float(ticker_share)))))


def _topic_distribution_for_part(part: pd.DataFrame, topic_ids: list[str]) -> np.ndarray:
    """Рассчитывает распределение тем внутри фрагмента сообщений
    
    Args:
        part: фрагмент таблицы сообщений внутри шага дискретизации
        topic_ids: список идентификаторов тем
    
    Returns:
        распределение тем во фрагменте сообщений
    """
    counts = np.zeros(len(topic_ids), dtype=float)
    if part is None or pd.DataFrame(part).empty or "topic" not in part.columns:
        return counts
    pos = {str(t): i for i, t in enumerate(topic_ids)}
    for topic, count in pd.DataFrame(part)["topic"].astype(str).value_counts().items():
        idx = pos.get(str(topic))
        if idx is not None:
            counts[idx] = float(count)
    total = float(counts.sum())
    return counts / total if total > 0 else counts


def _historical_context_statistics(context_df: pd.DataFrame, start_ts: pd.Timestamp, window_hours: int) -> dict[str, Any]:
    """Строит исторические характеристики для нормировки текущего шага
    
    Args:
        context_df: таблица сообщений исторического контекста
        start_ts: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
    
    Returns:
        словарь статистик исторического контекста
    """
    if context_df is None or pd.DataFrame(context_df).empty or "timestamp" not in context_df.columns:
        return {"features": {}, "model_exposure": {}}
    data = pd.DataFrame(context_df).copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    hist = data[data["timestamp"] < pd.Timestamp(start_ts)].copy()
    if hist.empty:
        return {"features": {}, "model_exposure": {}}
    hist["_window_start"] = hist["timestamp"].dt.floor(f"{int(window_hours)}h")
    topic_ids = sorted([str(x) for x in hist.get("topic", pd.Series(dtype=int)).dropna().astype(int).unique().tolist()]) if "topic" in hist.columns else []
    windows: list[tuple[pd.Timestamp, pd.DataFrame, dict[str, Any], np.ndarray]] = []
    for win, part in hist.groupby("_window_start", sort=True):
        if pd.isna(win):
            continue
        part_df = pd.DataFrame(part).copy()
        analytics = _window_ticker_analytics(part_df)
        props = _topic_distribution_for_part(part_df, topic_ids) if topic_ids else np.asarray([], dtype=float)
        windows.append((pd.Timestamp(win), part_df, analytics.get("ticker_summary", {}) or {}, props))
    all_tickers = sorted({ticker for _, _, summary, _ in windows for ticker in summary.keys()})
    features: dict[str, dict[str, list[float]]] = {ticker: {
        "post_count": [], "unique_authors": [], "thread_count": [], "direct_post_count": [],
        "inherited_post_count": [], "activity_rate": [], "author_rate": [], "sentiment_intensity": [],
        "avg_influence": [], "share": []
    } for ticker in all_tickers}
    for _, part_df, summary, _ in windows:
        n_posts = max(1, int(len(part_df)))
        n_authors = max(1, int(part_df["author"].astype(str).nunique()) if "author" in part_df.columns else 1)
        for ticker in all_tickers:
            info = summary.get(ticker, {}) or {}
            posts = float(info.get("post_count", 0) or 0)
            authors = float(info.get("unique_authors", 0) or 0)
            direct_posts = float(info.get("direct_post_count", 0) or 0)
            inherited_posts = float(info.get("inherited_post_count", 0) or 0)
            features[ticker]["post_count"].append(posts)
            features[ticker]["unique_authors"].append(authors)
            features[ticker]["direct_post_count"].append(direct_posts)
            features[ticker]["inherited_post_count"].append(inherited_posts)
            features[ticker]["thread_count"].append(float(info.get("root_thread_count", 0) or 0))
            features[ticker]["activity_rate"].append(posts / n_posts)
            features[ticker]["author_rate"].append(authors / n_authors)
            features[ticker]["sentiment_intensity"].append(float(info.get("sentiment_intensity", 0.0) or 0.0))
            features[ticker]["avg_influence"].append(float(info.get("avg_influence", 0.0) or 0.0))
            features[ticker]["share"].append(float(info.get("share", 0.0) or 0.0))
    model_exposure: dict[str, list[float]] = {ticker: [] for ticker in all_tickers}
    topic_shift_history: list[float] = []
    for i in range(1, len(windows)):
        _, _, prev_summary, prev_props = windows[i - 1]
        _, _, _, cur_props = windows[i]
        if len(prev_props) == 0 or len(cur_props) == 0:
            continue
        n = min(len(prev_props), len(cur_props))
        shift_vec = np.abs(cur_props[:n] - prev_props[:n])
        topic_shift_history.append(float(np.sum(shift_vec)))
        for ticker in all_tickers:
            info = prev_summary.get(ticker, {}) or {}
            weights = _topic_weight_vector(info.get("topic_shares", {}) or {}, topic_ids[:n], n) if n else np.asarray([], dtype=float)
            if len(weights):
                model_exposure[ticker].append(float(np.sum(weights * shift_vec[: len(weights)])))
            else:
                model_exposure[ticker].append(0.0)
    return {"features": features, "model_exposure": model_exposure, "topic_shift_history": topic_shift_history}


def _apply_empirical_ticker_statistics(summary: dict[str, Any], historical: dict[str, Any]) -> dict[str, Any]:
    """Добавляет эмпирические p-значения и q-значения к тикерной сводке
    
    Args:
        summary: таблица или словарь сводных показателей
        historical: исторический контекст для нормировки текущих значений
    
    Returns:
        таблица тикеров с эмпирическими статистиками
    """
    features_by_ticker = historical.get("features", {}) or {}
    model_by_ticker = historical.get("model_exposure", {}) or {}
    entries = [(str(ticker).upper(), info) for ticker, info in (summary or {}).items()]
    global_features: dict[str, list[float]] = {"post_count": [], "unique_authors": [], "thread_count": [], "activity_rate": []}
    for feature_set in features_by_ticker.values():
        posts_list = list(feature_set.get("post_count", []) or [])
        authors_list = list(feature_set.get("unique_authors", []) or [])
        thread_list = list(feature_set.get("thread_count", []) or [])
        rate_list = list(feature_set.get("activity_rate", []) or [])
        for idx, posts_value in enumerate(posts_list):
            posts_float = _finite_float(posts_value, 0.0)
            if posts_float <= 0:
                continue
            global_features["post_count"].append(posts_float)
            if idx < len(authors_list):
                global_features["unique_authors"].append(_finite_float(authors_list[idx], 0.0))
            if idx < len(thread_list):
                global_features["thread_count"].append(_finite_float(thread_list[idx], 0.0))
            if idx < len(rate_list):
                global_features["activity_rate"].append(_finite_float(rate_list[idx], 0.0))
    current_posts_values = [_finite_float(info.get("post_count", 0.0)) for _, info in entries]
    current_authors_values = [_finite_float(info.get("unique_authors", 0.0)) for _, info in entries]
    current_thread_values = [_finite_float(info.get("root_thread_count", 0.0)) for _, info in entries]
    current_share_values = [_finite_float(info.get("share", 0.0)) for _, info in entries]
    context_posts_values = [_finite_float(info.get("context_post_count", info.get("post_count", 0.0))) for _, info in entries]
    context_authors_values = [_finite_float(info.get("context_unique_authors", info.get("unique_authors", 0.0))) for _, info in entries]
    context_share_values = [_finite_float(info.get("context_share", info.get("share", 0.0))) for _, info in entries]
    p_cur_posts = _cross_section_upper_pvalues(current_posts_values)
    p_cur_authors = _cross_section_upper_pvalues(current_authors_values)
    p_cur_thread = _cross_section_upper_pvalues(current_thread_values)
    p_cur_share = _cross_section_upper_pvalues(current_share_values)
    p_ctx_posts = _cross_section_upper_pvalues(context_posts_values)
    p_ctx_authors = _cross_section_upper_pvalues(context_authors_values)
    p_ctx_share = _cross_section_upper_pvalues(context_share_values)
    for pos, (ticker_key, info) in enumerate(entries):
        features = features_by_ticker.get(ticker_key, {}) or {}
        current_posts = _finite_float(info.get("post_count", 0.0))
        current_authors = _finite_float(info.get("unique_authors", 0.0))
        current_thread = _finite_float(info.get("root_thread_count", 0.0))
        current_sentiment = _finite_float(info.get("sentiment_intensity", 0.0))
        current_influence = _finite_float(info.get("avg_influence", 0.0))
        p_posts = _empirical_upper_pvalue(current_posts, features.get("post_count", []))
        p_authors = _empirical_upper_pvalue(current_authors, features.get("unique_authors", []))
        p_thread = _empirical_upper_pvalue(current_thread, features.get("thread_count", []))
        p_sentiment = _empirical_upper_pvalue(current_sentiment, features.get("sentiment_intensity", []))
        p_influence = _empirical_upper_pvalue(current_influence, features.get("avg_influence", []))
        current_rate = _finite_float(info.get("activity_rate", 0.0))
        p_global_posts = _empirical_upper_pvalue(current_posts, global_features.get("post_count", [])) if current_posts > 0 else 1.0
        p_global_authors = _empirical_upper_pvalue(current_authors, global_features.get("unique_authors", [])) if current_posts > 0 else 1.0
        p_global_thread = _empirical_upper_pvalue(current_thread, global_features.get("thread_count", [])) if current_posts > 0 else 1.0
        p_global_activity_rate = _empirical_upper_pvalue(current_rate, global_features.get("activity_rate", [])) if current_posts > 0 else 1.0
        p_observed_support = _max_available_pvalue([p_global_posts, p_global_authors, p_global_thread, p_global_activity_rate])
        p_current_prominence = _combine_pvalues_fisher([p_cur_posts[pos], p_cur_authors[pos], p_cur_thread[pos], p_cur_share[pos]])
        p_context_prominence = _combine_pvalues_fisher([p_ctx_posts[pos], p_ctx_authors[pos], p_ctx_share[pos]])
        p_presence = _combine_pvalues_tippett([p_current_prominence, p_context_prominence])
        p_growth = _combine_pvalues_fisher([p_posts, p_authors, p_thread])
        p_activity = _combine_pvalues_tippett([p_growth, p_presence])
        p_text = _combine_pvalues_fisher([p_sentiment, p_influence])
        baseline_mean, baseline_median = _empirical_baseline(features.get("post_count", []))
        robust_z = _robust_zscore(current_posts, features.get("post_count", []))
        activity_lift = (current_posts / baseline_mean) if baseline_mean > 0 else (None if current_posts == 0 else float("inf"))
        info["baseline_posts"] = baseline_mean
        info["baseline_median_posts"] = baseline_median
        info["activity_lift"] = activity_lift
        info["activity_zscore"] = None
        info["activity_robust_zscore"] = robust_z
        info["attention_score"] = _attention_score(float(info.get("activity_rate", 0.0) or 0.0), float(info.get("share", 0.0) or 0.0))
        info["p_posts"] = p_posts
        info["p_authors"] = p_authors
        info["p_thread"] = p_thread
        info["p_sentiment"] = p_sentiment
        info["p_influence"] = p_influence
        info["p_current_prominence"] = p_current_prominence
        info["p_context_prominence"] = p_context_prominence
        info["p_presence"] = p_presence
        info["p_growth"] = p_growth
        info["p_global_posts"] = p_global_posts
        info["p_global_authors"] = p_global_authors
        info["p_global_thread"] = p_global_thread
        info["p_global_activity_rate"] = p_global_activity_rate
        info["p_observed_support"] = p_observed_support
        info["p_activity"] = p_observed_support
        info["p_text"] = p_text
        info["activity_anomaly_score"] = _event_strength_from_pvalue(p_observed_support)
        info["support_score"] = _event_strength_from_pvalue(p_observed_support)
        info["historical_support_score"] = _event_strength_from_pvalue(p_context_prominence)
        info["model_exposure_history"] = model_by_ticker.get(ticker_key, [])
    return summary

def _attach_ticker_activity_baseline(payload: dict[str, Any], context_df: pd.DataFrame, start_ts: pd.Timestamp, window_hours: int) -> dict[str, Any]:
    """Добавляет историческую базу активности тикеров к payload результата
    
    Args:
        payload: словарь промежуточных результатов шага дискретизации
        context_df: таблица сообщений исторического контекста
        start_ts: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
    
    Returns:
        обновлённая структура с добавленными полями: историческую базу активности тикеров к payload результата
    """
    summary = payload.get("ticker_summary", {}) or {}
    if not summary:
        return payload
    historical = _historical_context_statistics(context_df, pd.Timestamp(start_ts), int(window_hours))
    payload["ticker_summary"] = _apply_empirical_ticker_statistics(summary, historical)
    payload["model_topic_shift_history"] = historical.get("topic_shift_history", [])
    return payload


def _attach_ticker_activity_baseline_from_window_history(
    ticker_summary: dict[str, Any],
    windows: list[dict[str, Any]],
    start_index: int,
    context_windows: int = 8,
) -> dict[str, Any]:
    """Добавляет базу активности тикеров из истории шагов дискретизации
    
    Args:
        ticker_summary: сводка финансовых инструментов
        windows: таблица шагов дискретизации
        start_index: индекс начального шага дискретизации
        context_windows: число предыдущих шагов дискретизации для исторического контекста
    
    Returns:
        обновлённая структура с добавленными полями: базу активности тикеров из истории шагов дискретизации
    """
    summary = ticker_summary or {}
    if not summary:
        return summary
    begin = max(0, int(start_index) - max(0, int(context_windows)))
    hist = windows[begin:int(start_index)] if start_index > 0 else []
    features: dict[str, dict[str, list[float]]] = {}
    model_exposure: dict[str, list[float]] = {}
    all_tickers = sorted({t for w in hist for t in (w.get("ticker_summary", {}) or {}).keys()} | set(summary.keys()))
    for ticker in all_tickers:
        features[ticker] = {"post_count": [], "unique_authors": [], "thread_count": [], "sentiment_intensity": [], "avg_influence": []}
        model_exposure[ticker] = []
    for w in hist:
        wsum = w.get("ticker_summary", {}) or {}
        for ticker in all_tickers:
            info = wsum.get(ticker) or wsum.get(str(ticker).upper()) or {}
            features[ticker]["post_count"].append(float(info.get("post_count", 0) or 0))
            features[ticker]["unique_authors"].append(float(info.get("unique_authors", 0) or 0))
            features[ticker]["thread_count"].append(float(info.get("direct_post_count", 0) or 0) + float(info.get("inherited_post_count", 0) or 0))
            features[ticker]["sentiment_intensity"].append(float(info.get("sentiment_intensity", 0.0) or 0.0))
            features[ticker]["avg_influence"].append(float(info.get("avg_influence", 0.0) or 0.0))
    for i in range(1, len(hist)):
        prev = hist[i - 1]
        cur = hist[i]
        prev_props = np.asarray(prev.get("real_props", []), dtype=float)
        cur_props = np.asarray(cur.get("real_props", []), dtype=float)
        n = min(len(prev_props), len(cur_props))
        if n <= 0:
            continue
        topic_ids = [str(x) for x in prev.get("strategy_topics", [])][:n]
        shift_vec = np.abs(cur_props[:n] - prev_props[:n])
        prev_summary = prev.get("ticker_summary", {}) or {}
        for ticker in all_tickers:
            info = prev_summary.get(ticker) or prev_summary.get(str(ticker).upper()) or {}
            weights = _topic_weight_vector(info.get("topic_shares", {}) or {}, topic_ids, n)
            model_exposure[ticker].append(float(np.sum(weights * shift_vec[: len(weights)]))) if len(weights) else model_exposure[ticker].append(0.0)
    return _apply_empirical_ticker_statistics(summary, {"features": features, "model_exposure": model_exposure})


def _topic_weight_vector(topic_shares: dict | None, topic_ids: list[str], n: int) -> np.ndarray:
    """Преобразует тематическую структуру инструмента в вектор весов
    
    Args:
        topic_shares: распределение инструмента по темам
        topic_ids: список идентификаторов тем
        n: число элементов, шагов или будущих горизонтов
    
    Returns:
        вектор тематических весов инструмента
    """
    weights = np.zeros(int(n), dtype=float)
    topic_shares = topic_shares or {}
    pos = {str(tid): i for i, tid in enumerate(topic_ids[:n])}
    for topic_id, value in topic_shares.items():
        idx = pos.get(str(topic_id))
        if idx is not None:
            try:
                weights[idx] += max(0.0, float(value))
            except Exception:
                continue
    total = float(weights.sum())
    if total <= 0:
        # если у тикера нет тематической атрибуции, равномерный вектор используется только как численный резерв
        # вызывающий код обязан проверять ticker_has_topic_profile перед интерпретацией
        # метрик согласованности по конкретному тикеру
        return np.ones(int(n), dtype=float) / max(1, int(n))
    return weights / total


def _weighted_distribution_metrics(real: np.ndarray | list, pred: np.ndarray | list, weights: np.ndarray | list) -> dict[str, float | None]:
    """Рассчитывает метрики согласованности с весами по темам инструмента
    
    Args:
        real: фактическое тематическое распределение
        pred: смоделированное тематическое распределение
        weights: веса тем или компонентов расчёта
    
    Returns:
        словарь взвешенных метрик распределений
    """
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = min(len(real), len(pred), len(weights))
    if n <= 0:
        return {"rmse": None, "nrmse": None, "l1": None}
    real = real[:n]
    pred = pred[:n]
    weights = weights[:n]
    wsum = float(weights.sum())
    if wsum <= 0:
        weights = np.ones(n, dtype=float) / n
    else:
        weights = weights / wsum
    diff = real - pred
    rmse = float(np.sqrt(np.sum(weights * (diff ** 2))))
    l1 = float(np.sum(weights * np.abs(diff)))
    scale = _safe_actual_range(real, fallback=1.0)
    nrmse = float((rmse / scale) * 100.0)
    return {"rmse": rmse, "nrmse": nrmse, "l1": l1}

def _discussion_indicator_row(
    *,
    start_index,
    start_time,
    future_step: int,
    future_time,
    step_shift: float,
    from_start_shift: float,
    concentration: float,
    start_window: dict,
    realized: dict[str, Any],
    actual_available: bool,
    model_shift_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Формирует строку общего индикатора состояния дискуссии
    
    Args:
        start_index: индекс начального шага дискретизации
        start_time: временная метка начала анализируемого шага дискретизации
        future_step: номер будущего шага дискретизации
        future_time: время будущего шага дискретизации
        step_shift: сдвиг тематического распределения на шаге симуляции
        from_start_shift: сдвиг тематического распределения относительно начального состояния
        concentration: концентрация тематического распределения или внимания
        start_window: строка или индекс стартового шага дискретизации
        realized: фактически наблюдавшиеся рыночные или дискуссионные значения
        actual_available: признак доступности фактического будущего распределения
        model_shift_fields: поля индикатора ожидаемого тематического сдвига
    
    Returns:
        строка общего дискуссионного индикатора
    """
    sent_int = float(start_window.get("sentiment_intensity", 0.0) or 0.0)
    avg_sent = float(start_window.get("avg_sentiment", 0.0) or 0.0)
    avg_infl = float(start_window.get("avg_influence", 0.0) or 0.0)
    model_shift_fields = model_shift_fields or _model_shift_fields(from_start_shift, [], start_window.get("context_windows"))
    if model_shift_fields.get("model_shift_pvalue") is None:
        score = _discussion_score(step_shift, from_start_shift, concentration, sent_int, avg_infl)
    else:
        score = float(model_shift_fields.get("model_shift_indicator", 0.0))
    return {
        "start_index": start_index,
        "start_time": start_time,
        "future_step": int(future_step),
        "future_time": future_time,
        "ticker": DISCUSSION_LEVEL_TICKER,
        "ticker_share": 1.0,
        "ticker_sentiment": avg_sent,
        "ticker_sentiment_intensity": sent_int,
        "ticker_avg_influence": avg_infl,
        "predicted_topic_drift": float(step_shift),
        "predicted_topic_concentration": float(concentration),
        "ticker_topic_exposure": float(from_start_shift),
        **model_shift_fields,
        "forward_discussion_indicator": float(score),
        "combined_forward_discussion_indicator": float(score),
        "direction_hint": "discussion_state",
        "actual_available": bool(actual_available),
        **_indicator_metadata_from_window(start_window, actual_available),
        "realized_nrmse": realized.get("nrmse"),
        "realized_l1": realized.get("l1"),
        "realized_rmse": realized.get("rmse"),
        "interpretation": "общий индекс необычности дискуссии без обязательной привязки к тикеру",
    }


def _finalize_forward_indicator(rows: list[dict[str, Any]], output_path: str | Path | None) -> pd.DataFrame:
    """Завершает расчёт дискуссионных индикаторов для всех инструментов
    
    Args:
        rows: строки, подлежащие сохранению или объединению
        output_path: путь для сохранения результата
    
    Returns:
        итоговая таблица дискуссионных индикаторов
    """
    if not rows:
        return _empty_forward_indicator(output_path)
    df = pd.DataFrame(rows)
    for numeric_col in [
        "forward_discussion_indicator", "combined_forward_discussion_indicator",
        "ticker_event_surprisal", "ticker_indicator_scale", "ticker_current_support_gate",
        "model_expected_topic_shift", "model_shift_surprisal", "model_shift_indicator",
        "ticker_expected_topic_indicator", "ticker_expected_topic_shift", "ticker_expected_topic_lift",
    ]:
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce").astype(float)
    if "ticker_discussion_event_pvalue" in df.columns:
        df["ticker_discussion_event_qvalue"] = None
        if "future_step" in df.columns and "ticker" in df.columns:
            ticker_mask = df["ticker"].astype(str) != DISCUSSION_LEVEL_TICKER
            for _, idx in df[ticker_mask].groupby("future_step").groups.items():
                pvals = df.loc[idx, "ticker_discussion_event_pvalue"].tolist()
                df.loc[idx, "ticker_discussion_event_qvalue"] = _bh_qvalues(pvals)
            for idx in df[ticker_mask].index:
                row = df.loc[idx]
                p_event = row.get("ticker_discussion_event_pvalue")
                context_n = row.get("context_windows")
                gate = _ticker_current_support_gate(row)
                scale = _indicator_scale_from_context(context_n, row.get("ticker_context_post_count", row.get("ticker_post_count", None)))
                stat_indicator = _ticker_indicator_from_pvalue(p_event, row, context_windows=context_n)
                topic_indicator = _finite_float(row.get("ticker_expected_topic_indicator"), 0.0)
                qvalue = _finite_float(row.get("ticker_discussion_event_qvalue"), 1.0)
                df.at[idx, "ticker_event_surprisal"] = _event_surprisal(p_event)
                df.at[idx, "ticker_indicator_scale"] = scale
                df.at[idx, "ticker_current_support_gate"] = gate
                df.at[idx, "ticker_fdr_significant"] = bool(qvalue <= 0.15)
                df.at[idx, "forward_discussion_indicator"] = stat_indicator
                df.at[idx, "combined_forward_discussion_indicator"] = _combined_forward_indicator(stat_indicator, topic_indicator, 0.70)
    for col in FORWARD_INDICATOR_COLUMNS:
        if col not in df.columns:
            df[col] = None
    for sort_col in ["combined_forward_discussion_indicator", "forward_discussion_indicator", "ticker_expected_topic_indicator", "ticker_discussion_event_pvalue", "ticker_discussion_event_qvalue", "ticker_observed_support_pvalue", "ticker_current_prominence_pvalue", "ticker_post_count", "ticker_share"]:
        if sort_col not in df.columns:
            df[sort_col] = None
    df = df[FORWARD_INDICATOR_COLUMNS].sort_values(
        ["future_step", "combined_forward_discussion_indicator", "forward_discussion_indicator", "ticker_expected_topic_indicator", "ticker_discussion_event_pvalue", "ticker_post_count", "ticker_share"],
        ascending=[True, False, False, False, True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def _empty_forward_indicator(output_path: str | Path | None = None) -> pd.DataFrame:
    """Возвращает пустую таблицу дискуссионных индикаторов
    
    Args:
        output_path: путь для сохранения результата
    
    Returns:
        пустая таблица с колонками индикаторов
    """
    df = pd.DataFrame(columns=FORWARD_INDICATOR_COLUMNS)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def _window_payload_from_part(part: pd.DataFrame, window_time: Any, strategy_topics: list[int]) -> dict[str, Any]:
    """Формирует payload текущего шага дискретизации из сообщений
    
    Args:
        part: фрагмент таблицы сообщений внутри шага дискретизации
        window_time: время начала шага дискретизации
        strategy_topics: список тем, используемых как стратегии агентной модели
    
    Returns:
        payload шага дискретизации для дальнейших расчётов
    """
    topic_to_idx = {int(t): i for i, t in enumerate(strategy_topics)}
    counts = np.zeros(len(strategy_topics), dtype=float)
    if not part.empty and "topic" in part.columns:
        for topic, c in part["topic"].value_counts().items():
            try:
                t = int(topic)
            except Exception:
                continue
            if t in topic_to_idx:
                counts[topic_to_idx[t]] = float(c)
    total = float(counts.sum())
    props = (counts / total).tolist() if total > 0 else [0.0] * len(strategy_topics)
    ticker_data = _window_ticker_analytics(part) if not part.empty else {"top_tickers": {}, "ticker_total_weight": 0.0, "ticker_concentration": 0.0, "ticker_summary": {}}
    author_counts = part["author"].value_counts() if not part.empty and "author" in part.columns else pd.Series(dtype=float)
    if "influence" in part.columns and len(part):
        author_part = part[~part["author"].map(is_service_author)].copy() if "author" in part.columns else part
        influence_by_author = author_part.groupby("author")["influence"].mean().sort_values(ascending=False).head(10).to_dict() if len(author_part) else {}
    else:
        influence_by_author = {}
    coverage = float((part["topic"] != -1).mean()) if len(part) and "topic" in part.columns else 0.0
    emerging_share = float((part["topic"] == -1).mean()) if len(part) and "topic" in part.columns else 0.0
    return {
        "time": pd.Timestamp(window_time).isoformat(),
        "n_posts": int(len(part)),
        "strategy_topics": strategy_topics,
        "topic_counts": counts.astype(int).tolist(),
        "real_props": props,
        "avg_sentiment": float(part["sentiment"].mean()) if len(part) and "sentiment" in part.columns else 0.0,
        "sentiment_intensity": float(part["sentiment_intensity"].mean()) if len(part) and "sentiment_intensity" in part.columns else 0.0,
        "avg_influence": float(part["influence"].mean()) if len(part) and "influence" in part.columns else 0.0,
        "coverage": coverage,
        "emerging_topic_share": emerging_share,
        "top_tickers": ticker_data["top_tickers"],
        "ticker_total_weight": ticker_data["ticker_total_weight"],
        "ticker_concentration": ticker_data["ticker_concentration"],
        "ticker_summary": ticker_data["ticker_summary"],
        "top_authors": author_counts.head(10).to_dict(),
        "top_authors_by_influence": {str(k): float(v) for k, v in influence_by_author.items()},
        "author_concentration": float(author_counts.iloc[0] / len(part)) if len(part) and len(author_counts) else 0.0,
    }


def build_local_window_from_messages(messages_path: str | Path, start_time: Any, window_hours: int, strategy_topics: list[int]) -> dict[str, Any]:
    """Строит промежуточную структуру данных для вычислительного конвейера
    
    Args:
        messages_path: путь к таблице обработанных сообщений
        start_time: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
        strategy_topics: список тем, используемых как стратегии агентной модели
    
    Returns:
        построенная структура: промежуточную структуру данных для вычислительного конвейера
    """
    messages = pd.read_csv(messages_path)
    if "timestamp" not in messages.columns:
        raise ValueError("messages_enriched.csv must contain timestamp column")
    messages["timestamp"] = pd.to_datetime(messages["timestamp"], errors="coerce")
    start_ts = pd.Timestamp(pd.to_datetime(start_time, errors="coerce")).tz_localize(None)
    end_ts = start_ts + pd.Timedelta(hours=int(window_hours))
    part = messages[(messages["timestamp"] >= start_ts) & (messages["timestamp"] < end_ts)].copy()
    return _window_payload_from_part(part, start_ts, strategy_topics)
