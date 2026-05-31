"""Построение состояния локального шага и запуск симуляционного индикатора"""
from __future__ import annotations


from ..pipeline.common import *
from ..pipeline.configuration import AppConfig, _emit_progress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_ingestion.preparation import load_windows
    from ..discussion.influence_payoff import assign_dynamic_window_influence
    from ..indicators.discussion_indicators import (
        _align_arrays,
        _apply_empirical_ticker_statistics,
        _attach_ticker_activity_baseline_from_window_history,
        _combined_forward_indicator,
        _direction_label,
        _discussion_indicator_row,
        _empty_forward_indicator,
        _finalize_forward_indicator,
        _historical_context_statistics,
        _indicator_metadata_from_window,
        _indicator_reasons,
        _model_shift_fields,
        _step_to_arrays,
        _ticker_discussion_pvalues,
        _ticker_expected_topic_fields,
        _ticker_indicator_from_pvalue,
        _topic_shift_history_from_windows,
        _topic_weight_vector,
        _weighted_distribution_metrics,
        _window_payload_from_part,
    )
    from ..nlp.ticker_extraction import _ticker_activity_fields
    from ..simulation.calibration import detect_prolonged
    from ..simulation.modeling import (
        _nrmse_threshold_percent,
        compare_distributions,
        distribution_to_counts,
        load_payoff,
        reference_metric_summary,
    )

def _safe_json_key(value: Any) -> str:
    """Нормализует значение для использования в ключе JSON или кэша
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        строковый ключ для JSON или кэша
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:80]


def load_fixed_model_params(cfg: AppConfig, best_params_path: str | Path | None = None) -> dict[str, Any]:
    """Загружает параметры модели из файлов калибровки и конфигурации
    
    Args:
        cfg: конфигурация проекта или приложения
        best_params_path: путь к файлу лучших параметров калибровки
    
    Returns:
        словарь параметров фиксированной агентной модели
    """
    path = Path(best_params_path) if best_params_path is not None else cfg.output_dir / "calibration" / "best_params.json"
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
    def _alpha(v):
        """Преобразует значение alpha из конфигурации калибровки
        
        Args:
            v: значение словаря или последовательности
        
        Returns:
            значение alpha или None для автоматического режима
        """
        if v is None:
            return None
        if str(v).lower() in {"none", "null", "nan", "unweighted"}:
            return None
        try:
            return float(v)
        except Exception:
            return None
    return {
        "source": str(path) if path.exists() else "config_defaults",
        "is_calibrated": bool(path.exists()),
        "prob_revision": float(data.get("prob_revision", cfg.get("model", "prob_revision", default=0.03))),
        "alpha": _alpha(data.get("alpha", cfg.get("model", "alpha", default=0.2))),
        "noise": float(data.get("noise", cfg.get("model", "noise", default=0.4))),
        "w_sentiment": float(data.get("w_sentiment", cfg.get("payoff", "w_sentiment", default=0.5))),
        "w_influence": float(data.get("w_influence", cfg.get("payoff", "w_influence", default=0.5))),
        "sentiment_mode": str(data.get("sentiment_mode", cfg.get("payoff", "sentiment_mode", default="magnitude"))),
    }


def _local_state_cache_key(start_time: Any, window_hours: int, context_windows: int, strategy_topics: list[int], cfg: AppConfig) -> str:
    """Формирует ключ кэша локального состояния шага дискретизации
    
    Args:
        start_time: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
        context_windows: число предыдущих шагов дискретизации для исторического контекста
        strategy_topics: список тем, используемых как стратегии агентной модели
        cfg: конфигурация проекта или приложения
    
    Returns:
        ключ кэша локального состояния
    """
    payload = {
        "start_time": pd.Timestamp(pd.to_datetime(start_time, errors="coerce")).isoformat(),
        "window_hours": int(window_hours),
        "context_windows": int(context_windows),
        "strategy_topics": [int(x) for x in strategy_topics],
        "influence": cfg.get("influence", default={}) or {},
        "topic_state": cfg.get("topic_state", default={}) or {},
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def build_dynamic_window_state(
    cfg: AppConfig,
    messages_path: str | Path,
    start_time: Any,
    window_hours: int,
    strategy_topics: list[int],
    context_windows: int = 8,
    use_cache: bool = True,
    progress=None,
) -> dict[str, Any]:
    """Строит локальное состояние дискуссии для выбранного шага дискретизации
    
    Args:
        cfg: конфигурация проекта или приложения
        messages_path: путь к таблице обработанных сообщений
        start_time: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
        strategy_topics: список тем, используемых как стратегии агентной модели
        context_windows: число предыдущих шагов дискретизации для исторического контекста
        use_cache: признак использования сохранённого кэша расчётов
        progress: callback для передачи статуса выполнения
    
    Returns:
        словарь локального состояния выбранного шага
    """
    messages_path = Path(messages_path)
    if not messages_path.exists():
        raise FileNotFoundError(messages_path)
    start_ts = pd.Timestamp(pd.to_datetime(start_time, errors="coerce")).tz_localize(None)
    window_hours = int(window_hours)
    context_windows = max(0, int(context_windows))
    cache_dir = cfg.output_dir / "cache" / "window_state"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _local_state_cache_key(start_ts, window_hours, context_windows, strategy_topics, cfg)
    cache_file = cache_dir / f"{key}.json"
    if use_cache and cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    _emit_progress(progress, "Загружаю обогащённые сообщения для локального шага дискретизации", 0, 3, stage="window_state")
    df = pd.read_csv(messages_path)
    if "timestamp" not in df.columns:
        raise ValueError("messages_enriched.csv must contain timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    context_start = start_ts - pd.Timedelta(hours=window_hours * context_windows)
    end_ts = start_ts + pd.Timedelta(hours=window_hours)
    context = df[(df["timestamp"] >= context_start) & (df["timestamp"] < end_ts)].copy()
    current = context[(context["timestamp"] >= start_ts) & (context["timestamp"] < end_ts)].copy()

    if context.empty:
        payload = _window_payload_from_part(current, start_ts, strategy_topics)
        payload.update({"context_start": context_start.isoformat(), "window_end": end_ts.isoformat(), "context_windows": context_windows, "used_cache": False})
        return payload

    _emit_progress(progress, "Пересчитываю динамическую авторитетность для контекста шаги дискретизации", 1, 3, stage="window_state")
    # динамическую авторитетность пересчитываем только внутри разрешённого контекста
    # это дешевле полного NLP, потому что тексты уже обогащены признаками
    context_recalc, influence_meta = assign_dynamic_window_influence(context, cfg, window_hours=window_hours, progress=progress)
    current_recalc = context_recalc[(context_recalc["timestamp"] >= start_ts) & (context_recalc["timestamp"] < end_ts)].copy()

    _emit_progress(progress, "Собираю состояние выбранного шага дискретизации", 2, 3, stage="window_state")
    payload = _window_payload_from_part(current_recalc, start_ts, strategy_topics)

    # добавляем распределение тем контекста с экспоненциальным затуханием
    # это не замена текущего распределения, а сигнал устойчивости
    lam = float(cfg.get("topic_state", "decay_lambda", default=0.15))
    age_hours = (end_ts - context_recalc["timestamp"]).dt.total_seconds() / 3600.0
    context_recalc["_context_weight"] = np.exp(-lam * age_hours / max(window_hours, 1))
    topic_to_idx = {int(t): i for i, t in enumerate(strategy_topics)}
    ctx_counts = np.zeros(len(strategy_topics), dtype=float)
    for topic, group in context_recalc.groupby("topic"):
        try:
            t = int(topic)
        except Exception:
            continue
        if t in topic_to_idx:
            ctx_counts[topic_to_idx[t]] = float(group["_context_weight"].sum())
    ctx_props = (ctx_counts / ctx_counts.sum()).tolist() if ctx_counts.sum() > 0 else [0.0] * len(strategy_topics)
    payload.update({
        "context_start": context_start.isoformat(),
        "window_end": end_ts.isoformat(),
        "context_windows": context_windows,
        "context_n_posts": int(len(context_recalc)),
        "context_topic_props": ctx_props,
        "dynamic_state_mode": "current_window_plus_rolling_context",
        "no_future_leakage": True,
        "influence_metadata": {k: v for k, v in influence_meta.items() if k != "window_summaries"},
        "used_cache": False,
        "cache_key": key,
    })
    _emit_progress(progress, "Состояние шаги дискретизации рассчитано", 3, 3, stage="window_state")
    if use_cache:
        try:
            cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass
    return payload


def fixed_model_predict_from_window(
    cfg: AppConfig,
    start_window: dict[str, Any],
    payoff_path: str | Path,
    future_steps: int,
    fixed_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Строит смоделированную тематическую динамику из фиксированных параметров
    
    Args:
        cfg: конфигурация проекта или приложения
        start_window: строка или индекс стартового шага дискретизации
        payoff_path: путь к файлу выигрышей тем
        future_steps: число будущих шагов дискретизации для симуляции
        fixed_params: фиксированные параметры агентной модели
    
    Returns:
        таблица смоделированной тематической динамики
    """
    params = fixed_params or load_fixed_model_params(cfg)
    strategy_topics = [int(t) for t in start_window.get("strategy_topics", [])]
    if not strategy_topics:
        return pd.DataFrame()
    payoff = load_payoff(payoff_path, strategy_topics)
    runner = NetLogoRunner(NetLogoConfig(**(cfg.get("netlogo", default={}) or {})))
    pop_size = int(cfg.get("netlogo", "pop_size", default=1000))
    initial = distribution_to_counts(start_window.get("real_props", []), pop_size)
    return runner.run(
        payoff=payoff,
        initial_distribution=initial,
        n_steps=int(max(1, future_steps)),
        prob_revision=float(params["prob_revision"]),
        noise=float(params["noise"]),
        alpha=params.get("alpha"),
    )


def run_predictive_market_indicator(
    cfg: AppConfig,
    windows_path: str | Path,
    payoff_path: str | Path,
    start_index: int,
    future_steps: int,
    prob_revision: float,
    alpha: float | None,
    noise: float,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Рассчитывает дискуссионные индикаторы для шага истории
    
    Args:
        cfg: конфигурация проекта или приложения
        windows_path: путь к таблице шагов дискретизации
        payoff_path: путь к файлу выигрышей тем
        start_index: индекс начального шага дискретизации
        future_steps: число будущих шагов дискретизации для симуляции
        prob_revision: вероятность пересмотра стратегии агентом
        alpha: параметр баланса собственной и наблюдаемой стратегии
        noise: вероятность случайного выбора стратегии
        output_path: путь для сохранения результата
    
    Returns:
        таблица дискуссионных индикаторов выбранного шага
    """
    windows = load_windows(windows_path)
    if not windows:
        return _empty_forward_indicator(output_path)
    start_index = int(max(0, min(int(start_index), len(windows) - 1)))
    future_steps = int(max(1, min(int(future_steps), len(windows) - start_index - 1 if len(windows) > start_index + 1 else int(future_steps))))
    start = dict(windows[start_index])
    try:
        inferred_hours = int(cfg.get("processing", "window_hours", default=6))
    except Exception:
        inferred_hours = 6
    try:
        if start_index + 1 < len(windows):
            t0 = pd.to_datetime(start.get("time"), errors="coerce")
            t1 = pd.to_datetime(windows[start_index + 1].get("time"), errors="coerce")
            if not pd.isna(t0) and not pd.isna(t1):
                inferred_hours = max(1, int(round((t1 - t0).total_seconds() / 3600)))
    except Exception:
        pass
    st_time = pd.to_datetime(start.get("time"), errors="coerce")
    planned_end = pd.Timestamp(st_time).tz_localize(None) + pd.Timedelta(hours=inferred_hours) if not pd.isna(st_time) else None
    start.update({
        "analysis_mode": "forecast_analysis",
        "source_mode": "precomputed_windows",
        "window_status": "historical_or_loaded_window",
        "window_complete": True,
        "planned_end_time": planned_end.isoformat() if planned_end is not None else None,
        "observed_until": planned_end.isoformat() if planned_end is not None else None,
        "topic_rebuild_mode": "precomputed_topic_space",
        "no_future_leakage": True,
    })
    strategy_topics = [int(t) for t in start.get("strategy_topics", [])]
    if not strategy_topics:
        return _empty_forward_indicator(output_path)
    payoff = load_payoff(payoff_path, strategy_topics)
    runner = NetLogoRunner(NetLogoConfig(**(cfg.get("netlogo", default={}) or {})))
    pop_size = int(cfg.get("netlogo", "pop_size", default=1000))
    initial = distribution_to_counts(start.get("real_props", []), pop_size)
    sim = runner.run(payoff=payoff, initial_distribution=initial, n_steps=int(future_steps), prob_revision=prob_revision, noise=noise, alpha=alpha)
    prop_cols = [c for c in sim.columns if c.startswith("prop_")]
    start_props = np.asarray(start.get("real_props", []), dtype=float)
    topic_ids = [str(t) for t in strategy_topics]
    ticker_summary = start.get("ticker_summary", {}) or {}
    if not ticker_summary:
        total = sum((start.get("top_tickers", {}) or {}).values()) or 1.0
        ticker_summary = {
            t: {"share": float(v) / total, "sentiment_intensity": float(start.get("sentiment_intensity", 0.0)), "avg_sentiment": float(start.get("avg_sentiment", 0.0)), "avg_influence": float(start.get("avg_influence", 0.0)), "topic_shares": {}, "post_count": int(v), "unique_authors": 0, "activity_rate": 0.0, "author_rate": 0.0}
            for t, v in (start.get("top_tickers", {}) or {}).items()
        }
    context_n = int(start.get("context_windows") or cfg.get("topic_state", "context_windows", default=8) or 8)
    ticker_summary = _attach_ticker_activity_baseline_from_window_history(
        ticker_summary,
        windows,
        start_index,
        context_n,
    )
    topic_shift_history = _topic_shift_history_from_windows(windows, start_index, context_n)
    rows: list[dict[str, Any]] = []
    prev_pred = start_props.copy()
    for h in range(1, future_steps + 1):
        pred = _step_to_arrays(sim, prop_cols, h)
        if pred is None:
            continue
        pred, baseline = _align_arrays(pred, start_props)
        prev, pred_for_prev = _align_arrays(prev_pred, pred)
        cur_topic_ids = topic_ids[: len(pred)]
        step_shift_vec = np.abs(pred_for_prev - prev)
        from_start_vec = np.abs(pred - baseline)
        step_shift = float(np.sum(step_shift_vec))
        from_start_shift = float(np.sum(from_start_vec))
        concentration = float(np.max(pred)) if len(pred) else 0.0
        model_shift_fields = _model_shift_fields(from_start_shift, topic_shift_history, context_n)
        future_idx = start_index + h
        actual_available = future_idx < len(windows)
        realized = {"rmse": None, "nrmse": None, "l1": None, "cosine": None}
        future_time = None
        if actual_available:
            future_time = windows[future_idx].get("time")
            realized = compare_distributions(windows[future_idx].get("real_props", []), pred.tolist())
        rows.append(_discussion_indicator_row(
            start_index=start_index,
            start_time=start.get("time"),
            future_step=h,
            future_time=future_time,
            step_shift=step_shift,
            from_start_shift=from_start_shift,
            concentration=concentration,
            start_window=start,
            realized=realized,
            actual_available=actual_available,
            model_shift_fields=model_shift_fields,
        ))
        shift_by_topic_id = {tid: float(step_shift_vec[i]) for i, tid in enumerate(cur_topic_ids)}
        for ticker, info in ticker_summary.items():
            share = float(info.get("share", 0.0) or 0.0)
            raw_topic_shares = info.get("topic_shares", {}) or {}
            has_topic_profile = bool(raw_topic_shares)
            weights = _topic_weight_vector(raw_topic_shares, cur_topic_ids, len(pred))
            topic_exposure = float(np.sum(weights * step_shift_vec[: len(weights)])) if has_topic_profile else 0.0
            ticker_from_start = float(np.sum(weights * from_start_vec[: len(weights)])) if has_topic_profile else 0.0
            sent_int = float(info.get("sentiment_intensity", start.get("sentiment_intensity", 0.0)) or 0.0)
            avg_sent = float(info.get("avg_sentiment", start.get("avg_sentiment", 0.0)) or 0.0)
            avg_infl = float(info.get("avg_influence", start.get("avg_influence", 0.0)) or 0.0)
            activity = _ticker_activity_fields(info)
            row_context_windows = start.get("context_windows") if isinstance(locals().get("start"), dict) else start_window.get("context_windows")
            topic_model_fields = _ticker_expected_topic_fields(
                info, weights, baseline, pred, step_shift_vec, from_start_vec, has_topic_profile,
                context_windows=row_context_windows,
            )
            model_pvalue = topic_model_fields.get("ticker_expected_topic_pvalue")
            pvalues = _ticker_discussion_pvalues(activity, model_pvalue=model_pvalue)
            forward_score = _ticker_indicator_from_pvalue(pvalues.get("ticker_discussion_event_pvalue"), activity, context_windows=row_context_windows)
            combined_score = _combined_forward_indicator(forward_score, topic_model_fields.get("ticker_expected_topic_indicator"), 0.70)
            ticker_realized = {"rmse": None, "nrmse": None, "l1": None}
            if actual_available and has_topic_profile:
                actual_arr = np.asarray(windows[future_idx].get("real_props", []), dtype=float)
                actual_arr, pred_for_actual = _align_arrays(actual_arr, pred)
                ticker_realized = _weighted_distribution_metrics(actual_arr, pred_for_actual, weights)
            out = {
                "start_index": start_index,
                "start_time": start.get("time"),
                "future_step": h,
                "future_time": future_time,
                "ticker": str(ticker).upper(),
                "ticker_share": share,
                "ticker_sentiment": avg_sent,
                "ticker_sentiment_intensity": sent_int,
                "ticker_avg_influence": avg_infl,
                "predicted_topic_drift": step_shift,
                "predicted_topic_concentration": concentration,
                "ticker_topic_exposure": float(topic_exposure),
                **model_shift_fields,
                **topic_model_fields,
                "forward_discussion_indicator": forward_score,
                "combined_forward_discussion_indicator": combined_score,
                "direction_hint": _direction_label(avg_sent),
                "actual_available": bool(actual_available),
                "realized_nrmse": realized.get("nrmse"),
                "realized_l1": realized.get("l1"),
                "realized_rmse": realized.get("rmse"),
                "ticker_realized_nrmse": ticker_realized.get("nrmse"),
                "ticker_realized_l1": ticker_realized.get("l1"),
                "ticker_realized_rmse": ticker_realized.get("rmse"),
                "ticker_model_step_shift": float(topic_exposure),
                "ticker_model_from_start_shift": float(ticker_from_start),
                "ticker_has_topic_profile": bool(has_topic_profile),
                **activity,
                **pvalues,
            }
            out.update(_indicator_metadata_from_window(start_window if 'start_window' in locals() else start, actual_available))
            out["interpretation"] = _indicator_reasons(out)
            rows.append(out)
        prev_pred = pred.copy()
    if output_path is None:
        output_path = cfg.output_dir / "market" / "forward_discussion_indicators.csv"
    return _finalize_forward_indicator(rows, output_path)

def run_predictive_market_indicator_for_range(
    cfg: AppConfig,
    messages_path: str | Path,
    windows_path: str | Path,
    payoff_path: str | Path,
    start_time: Any,
    window_hours: int,
    future_steps: int,
    prob_revision: float,
    alpha: float | None,
    noise: float,
    output_path: str | Path | None = None,
    context_windows: int = 8,
    use_cache: bool = True,
    progress=None,
) -> pd.DataFrame:
    """Рассчитывает индикаторы для диапазона шагов дискретизации
    
    Args:
        cfg: конфигурация проекта или приложения
        messages_path: путь к таблице обработанных сообщений
        windows_path: путь к таблице шагов дискретизации
        payoff_path: путь к файлу выигрышей тем
        start_time: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
        future_steps: число будущих шагов дискретизации для симуляции
        prob_revision: вероятность пересмотра стратегии агентом
        alpha: параметр баланса собственной и наблюдаемой стратегии
        noise: вероятность случайного выбора стратегии
        output_path: путь для сохранения результата
        context_windows: число предыдущих шагов дискретизации для исторического контекста
        use_cache: признак использования сохранённого кэша расчётов
        progress: callback для передачи статуса выполнения
    
    Returns:
        таблица индикаторов диапазона шагов дискретизации
    """
    payoff_df = pd.read_csv(payoff_path)
    strategy_topics = [int(x) for x in payoff_df["topic"].tolist()]
    start_window = build_dynamic_window_state(
        cfg,
        messages_path,
        start_time,
        window_hours,
        strategy_topics,
        context_windows=int(context_windows),
        use_cache=bool(use_cache),
        progress=progress,
    )
    start_ts_meta = pd.Timestamp(pd.to_datetime(start_time, errors="coerce")).tz_localize(None)
    planned_end_meta = start_ts_meta + pd.Timedelta(hours=int(window_hours))
    start_window.update({
        "analysis_mode": "forecast_analysis",
        "source_mode": "enriched_messages_local_window",
        "window_status": "observed_window",
        "window_complete": bool(pd.Timestamp.now().tz_localize(None) >= planned_end_meta),
        "planned_end_time": planned_end_meta.isoformat(),
        "observed_until": planned_end_meta.isoformat(),
        "context_windows": int(context_windows),
        "topic_rebuild_mode": "precomputed_topic_space_with_dynamic_window_state",
        "no_future_leakage": True,
    })
    if int(start_window.get("n_posts", 0)) <= 0:
        return _empty_forward_indicator(output_path)
    payoff = load_payoff(payoff_path, strategy_topics)
    runner = NetLogoRunner(NetLogoConfig(**(cfg.get("netlogo", default={}) or {})))
    pop_size = int(cfg.get("netlogo", "pop_size", default=1000))
    initial = distribution_to_counts(start_window.get("real_props", []), pop_size)
    sim = runner.run(payoff=payoff, initial_distribution=initial, n_steps=int(future_steps), prob_revision=prob_revision, noise=noise, alpha=alpha)
    prop_cols = [c for c in sim.columns if c.startswith("prop_")]
    start_props = np.asarray(start_window.get("real_props", []), dtype=float)
    topic_ids = [str(t) for t in strategy_topics]
    ticker_summary = start_window.get("ticker_summary", {}) or {}
    messages = pd.read_csv(messages_path)
    messages["timestamp"] = pd.to_datetime(messages["timestamp"], errors="coerce")
    start_ts = pd.Timestamp(pd.to_datetime(start_time, errors="coerce")).tz_localize(None)
    historical = _historical_context_statistics(messages, start_ts, int(window_hours))
    ticker_summary = _apply_empirical_ticker_statistics(ticker_summary, historical)
    topic_shift_history = historical.get("topic_shift_history", [])
    rows: list[dict[str, Any]] = []
    prev_pred = start_props.copy()
    for h in range(1, int(future_steps) + 1):
        pred = _step_to_arrays(sim, prop_cols, h)
        if pred is None:
            continue
        pred, baseline = _align_arrays(pred, start_props)
        prev, pred_for_prev = _align_arrays(prev_pred, pred)
        cur_topic_ids = topic_ids[: len(pred)]
        step_shift_vec = np.abs(pred_for_prev - prev)
        from_start_vec = np.abs(pred - baseline)
        step_shift = float(np.sum(step_shift_vec))
        from_start_shift = float(np.sum(from_start_vec))
        concentration = float(np.max(pred)) if len(pred) else 0.0
        model_shift_fields = _model_shift_fields(from_start_shift, topic_shift_history, start_window.get("context_windows"))
        future_start = start_ts + pd.Timedelta(hours=int(window_hours) * h)
        future_part = messages[(messages["timestamp"] >= future_start) & (messages["timestamp"] < future_start + pd.Timedelta(hours=int(window_hours)))].copy()
        actual_available = not future_part.empty
        realized = {"rmse": None, "nrmse": None, "l1": None, "cosine": None}
        if actual_available:
            future_window = _window_payload_from_part(future_part, future_start, strategy_topics)
            realized = compare_distributions(future_window.get("real_props", []), pred.tolist())
        rows.append(_discussion_indicator_row(
            start_index=None,
            start_time=start_window.get("time"),
            future_step=h,
            future_time=future_start.isoformat(),
            step_shift=step_shift,
            from_start_shift=from_start_shift,
            concentration=concentration,
            start_window=start_window,
            realized=realized,
            actual_available=actual_available,
            model_shift_fields=model_shift_fields,
        ))
        shift_by_topic_id = {tid: float(step_shift_vec[i]) for i, tid in enumerate(cur_topic_ids)}
        for ticker, info in ticker_summary.items():
            share = float(info.get("share", 0.0) or 0.0)
            raw_topic_shares = info.get("topic_shares", {}) or {}
            has_topic_profile = bool(raw_topic_shares)
            weights = _topic_weight_vector(raw_topic_shares, cur_topic_ids, len(pred))
            topic_exposure = float(np.sum(weights * step_shift_vec[: len(weights)])) if has_topic_profile else 0.0
            ticker_from_start = float(np.sum(weights * from_start_vec[: len(weights)])) if has_topic_profile else 0.0
            sent_int = float(info.get("sentiment_intensity", start_window.get("sentiment_intensity", 0.0)) or 0.0)
            avg_sent = float(info.get("avg_sentiment", start_window.get("avg_sentiment", 0.0)) or 0.0)
            avg_infl = float(info.get("avg_influence", start_window.get("avg_influence", 0.0)) or 0.0)
            activity = _ticker_activity_fields(info)
            row_context_windows = start.get("context_windows") if isinstance(locals().get("start"), dict) else start_window.get("context_windows")
            topic_model_fields = _ticker_expected_topic_fields(
                info, weights, baseline, pred, step_shift_vec, from_start_vec, has_topic_profile,
                context_windows=row_context_windows,
            )
            model_pvalue = topic_model_fields.get("ticker_expected_topic_pvalue")
            pvalues = _ticker_discussion_pvalues(activity, model_pvalue=model_pvalue)
            forward_score = _ticker_indicator_from_pvalue(pvalues.get("ticker_discussion_event_pvalue"), activity, context_windows=row_context_windows)
            combined_score = _combined_forward_indicator(forward_score, topic_model_fields.get("ticker_expected_topic_indicator"), 0.70)
            ticker_realized = {"rmse": None, "nrmse": None, "l1": None}
            if actual_available and has_topic_profile:
                try:
                    future_actual_props = future_window.get("real_props", [])
                    actual_arr = np.asarray(future_actual_props, dtype=float)
                    actual_arr, pred_for_actual = _align_arrays(actual_arr, pred)
                    ticker_realized = _weighted_distribution_metrics(actual_arr, pred_for_actual, weights)
                except Exception:
                    ticker_realized = {"rmse": None, "nrmse": None, "l1": None}
            out = {
                "start_index": None,
                "start_time": start_window.get("time"),
                "future_step": h,
                "future_time": future_start.isoformat(),
                "ticker": str(ticker).upper(),
                "ticker_share": share,
                "ticker_sentiment": avg_sent,
                "ticker_sentiment_intensity": sent_int,
                "ticker_avg_influence": avg_infl,
                "predicted_topic_drift": step_shift,
                "predicted_topic_concentration": concentration,
                "ticker_topic_exposure": float(topic_exposure),
                **model_shift_fields,
                **topic_model_fields,
                "forward_discussion_indicator": forward_score,
                "combined_forward_discussion_indicator": combined_score,
                "direction_hint": _direction_label(avg_sent),
                "actual_available": bool(actual_available),
                "realized_nrmse": realized.get("nrmse"),
                "realized_l1": realized.get("l1"),
                "realized_rmse": realized.get("rmse"),
                "ticker_realized_nrmse": ticker_realized.get("nrmse"),
                "ticker_realized_l1": ticker_realized.get("l1"),
                "ticker_realized_rmse": ticker_realized.get("rmse"),
                "ticker_model_step_shift": float(topic_exposure),
                "ticker_model_from_start_shift": float(ticker_from_start),
                "ticker_has_topic_profile": bool(has_topic_profile),
                **activity,
                **pvalues,
            }
            out.update(_indicator_metadata_from_window(start_window if 'start_window' in locals() else start, actual_available))
            out["interpretation"] = _indicator_reasons(out)
            rows.append(out)
        prev_pred = pred.copy()
    if output_path is None:
        output_path = cfg.output_dir / "market" / "forward_discussion_indicators.csv"
    return _finalize_forward_indicator(rows, output_path)

def diagnose_forecasts(cfg: AppConfig, forecasts: pd.DataFrame) -> dict[str, Any]:
    """Формирует диагностику качества смоделированных тематических траекторий
    
    Args:
        cfg: конфигурация проекта или приложения
        forecasts: таблица результатов проверки моделирования
    
    Returns:
        таблица диагностики смоделированных траекторий
    """
    if forecasts.empty:
        return {"events": pd.DataFrame(), "rebuild_decision": {"action": "not_enough_data"}}
    nrmse_t = _nrmse_threshold_percent(cfg.get("modeling", "nrmse_threshold", default=35.0), default=35.0)
    rmse_t = float(cfg.get("modeling", "rmse_threshold", default=0.15))
    l1_t = float(cfg.get("modeling", "l1_threshold", default=0.35))
    rel_t = float(cfg.get("modeling", "reliability_threshold", default=0.65))
    em_t = float(cfg.get("modeling", "emerging_share_threshold", default=0.30))
    events = forecasts[(forecasts["nrmse"] >= nrmse_t) | (forecasts["rmse"] >= rmse_t) | (forecasts["l1"] >= l1_t)].copy()
    events["event_type"] = np.where(
        (events["coverage"] < rel_t) | (events["emerging_topic_share"] > em_t),
        "model_quality_issue",
        "local_state_deviation",
    ) if not events.empty else []
    prolonged = detect_prolonged(events, min_windows=int(cfg.get("modeling", "prolonged_min_windows", default=3)))
    if prolonged:
        action = "recalibrate_or_rebuild_model"
    elif not events.empty and (events["event_type"] == "model_quality_issue").mean() > 0.5:
        action = "refresh_topic_model"
    elif not events.empty:
        action = "inspect_local_anomalies"
    else:
        action = "model_is_stable"
    return {
        "events": events,
        "rebuild_decision": {
            "action": action,
            "prolonged_deviation": prolonged,
            "n_events": int(len(events)),
            "n_local_deviation_events": int((events["event_type"] == "local_state_deviation").sum()) if not events.empty else 0,
            "mean_nrmse": float(reference_metric_summary(forecasts).get("matrix_nrmse") or forecasts["nrmse"].mean()),
            "mean_rmse": float(reference_metric_summary(forecasts).get("matrix_rmse") or forecasts["rmse"].mean()),
            "mean_l1": float(forecasts["l1"].mean()),
        },
    }
