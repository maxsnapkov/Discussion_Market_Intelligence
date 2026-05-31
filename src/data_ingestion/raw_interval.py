"""Локальная обработка исходного диапазона сообщений"""
from __future__ import annotations


from ..pipeline.common import *
from ..pipeline.configuration import AppConfig, _emit_progress, _load_normalized_raw_data, clean_text
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..discussion.influence_payoff import assign_dynamic_window_influence, compute_payoff_from_messages
    from ..discussion.publications import _top_publications_from_frame, _top_publications_from_range_frame
    from ..discussion.user_graph import load_payoff_from_df
    from ..indicators.discussion_indicators import (
        DISCUSSION_LEVEL_TICKER,
        _align_arrays,
        _attach_ticker_activity_baseline,
        _combined_forward_indicator,
        _direction_label,
        _discussion_indicator_row,
        _empty_forward_indicator,
        _finalize_forward_indicator,
        _indicator_metadata_from_window,
        _indicator_reasons,
        _model_shift_fields,
        _step_to_arrays,
        _ticker_discussion_pvalues,
        _ticker_expected_topic_fields,
        _ticker_indicator_from_pvalue,
        _topic_shift_history_from_context_df,
        _topic_weight_vector,
        _weighted_distribution_metrics,
        _window_payload_from_part,
    )
    from ..nlp.sentiment import _score_sentiment_with_cache
    from ..nlp.ticker_extraction import (
        _accept_ticker_candidate_for_analytics,
        _ticker_activity_fields,
        attach_ticker_thread_participation,
        extract_ticker_candidates_many,
        load_ticker_universe,
    )
    from ..nlp.topic_modeling import TopicService
    from ..simulation.modeling import compare_distributions, distribution_to_counts
    from .preparation import _merge_context_ticker_summary, _ticker_payload_method_stats

def _normalize_weights(w_sentiment: float | None, w_influence: float | None) -> tuple[float, float]:
    """Нормирует веса функции выигрыша
    
    Args:
        w_sentiment: вес тональности в функции выигрыша
        w_influence: вес авторитетности в функции выигрыша
    
    Returns:
        пара нормированных весов функции выигрыша
    """
    try:
        ws = float(0.5 if w_sentiment is None else w_sentiment)
    except Exception:
        ws = 0.5
    try:
        wi = float(0.5 if w_influence is None else w_influence)
    except Exception:
        wi = 0.5
    total = ws + wi
    if total <= 0:
        return 0.5, 0.5
    return ws / total, wi / total


def _validate_raw_csv_path(raw_path: str | Path) -> Path:
    """Проверяет путь к исходному CSV-файлу локального анализа
    
    Args:
        raw_path: путь к исходной таблице сообщений
    
    Returns:
        bool: результат проверки условия: путь к исходному CSV-файлу локального анализа
    """
    path = Path(raw_path).expanduser()
    if not str(raw_path).strip():
        raise FileNotFoundError("Raw dataset path is empty. Select an existing CSV file for local range analysis.")
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset path must be a CSV file, not a directory: {path}")
    return path


def _enrich_raw_slice_for_window(
    cfg: AppConfig,
    raw_path: str | Path,
    start_time: Any,
    window_hours: int,
    context_windows: int,
    w_sentiment: float = 0.5,
    w_influence: float = 0.5,
    sentiment_mode: str | None = None,
    use_cache: bool = True,
    progress=None,
) -> dict[str, Any]:
    """Обогащает фрагмент исходных сообщений признаками для локального анализа
    
    Args:
        cfg: конфигурация проекта или приложения
        raw_path: путь к исходной таблице сообщений
        start_time: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
        context_windows: число предыдущих шагов дискретизации для исторического контекста
        w_sentiment: вес тональности в функции выигрыша
        w_influence: вес авторитетности в функции выигрыша
        sentiment_mode: режим преобразования тональности
        use_cache: признак использования сохранённого кэша расчётов
        progress: callback для передачи статуса выполнения
    
    Returns:
        таблица сообщений с добавленными признаками
    """
    raw_path = _validate_raw_csv_path(raw_path)
    start_ts = pd.Timestamp(pd.to_datetime(start_time, errors="coerce")).tz_localize(None)
    window_hours = int(window_hours)
    context_windows = max(0, int(context_windows))
    context_start = start_ts - pd.Timedelta(hours=window_hours * context_windows)
    end_ts = start_ts + pd.Timedelta(hours=window_hours)

    # локальные шаги дороги, потому что заново считают тональность, тикеры
    # локальное тематическое состояние и динамическую авторитетность
    # кэшируем полный payload по всем параметрам, влияющим на результат
    # если use_cache=False, кэш пропускается и вызывающий код может очистить директорию
    cache_payload = {
        "raw_path": str(raw_path.resolve()),
        "raw_mtime": raw_path.stat().st_mtime if raw_path.exists() else None,
        "raw_size": raw_path.stat().st_size if raw_path.exists() else None,
        "start_time": start_ts.isoformat(),
        "window_hours": int(window_hours),
        "context_windows": int(context_windows),
        "w_sentiment": float(w_sentiment),
        "w_influence": float(w_influence),
        "sentiment_mode": sentiment_mode or cfg.get("payoff", "sentiment_mode", default="magnitude"),
        "nlp": cfg.get("nlp", default={}) or {},
        "influence": cfg.get("influence", default={}) or {},
        "topic_state": cfg.get("topic_state", default={}) or {},
    }
    cache_key = hashlib.sha1(json.dumps(cache_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]
    cache_dir = cfg.output_dir / "cache" / "raw_window_state"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.pkl"
    if use_cache and cache_file.exists():
        try:
            _emit_progress(progress, "Локальный шаг дискретизации: загружаю состояние из кэша", 0, 1, stage="local_cache")
            return pd.read_pickle(cache_file)
        except Exception:
            pass

    _emit_progress(progress, "Локальный шаг дискретизации: читаю исходные данные", 0, 7, stage="local_raw")
    normalized_raw = _load_normalized_raw_data(raw_path, use_cache=bool(use_cache))
    max_source_ts = pd.to_datetime(normalized_raw["timestamp"], errors="coerce").dropna().max() if "timestamp" in normalized_raw else pd.NaT
    now_ts = pd.Timestamp.now().tz_localize(None)
    if pd.isna(max_source_ts):
        observed_until = start_ts
        data_complete = False
    else:
        max_source_ts = pd.Timestamp(max_source_ts).tz_localize(None)
        observed_until = min(max_source_ts, end_ts)
        data_complete = bool(max_source_ts >= end_ts)
    window_complete = bool(data_complete or now_ts >= end_ts)
    window_status = "complete_window" if window_complete else "active_observed_window"
    local_meta = {
        "analysis_mode": "forecast_analysis",
        "source_mode": "raw_csv_local_window",
        "window_status": window_status,
        "window_complete": window_complete,
        "planned_end_time": end_ts.isoformat(),
        "observed_until": observed_until.isoformat() if hasattr(observed_until, "isoformat") else None,
        "raw_source_path": str(raw_path.resolve()),
        "raw_source_mtime": raw_path.stat().st_mtime if raw_path.exists() else None,
        "raw_source_size": raw_path.stat().st_size if raw_path.exists() else None,
        "context_windows": int(context_windows),
        "topic_rebuild_mode": "fit_on_context_and_observed_window_without_future",
        "no_future_leakage": True,
    }
    df = normalized_raw[(normalized_raw["timestamp"] >= context_start) & (normalized_raw["timestamp"] < end_ts)].copy().sort_values("timestamp")
    if df.empty:
        strategy_topics = [0]
        payload = _window_payload_from_part(pd.DataFrame(columns=["timestamp", "topic", "sentiment", "sentiment_intensity", "influence", "author"]), start_ts, strategy_topics)
        payload.update(local_meta)
        return {"window": payload, "payoff": pd.DataFrame({"topic": [0], "payoff": [0.5]}), "context": df, "topic_service": None}

    _emit_progress(progress, f"Локальный шаг дискретизации: найдено сообщений {len(df)}", 1, 7, stage="local_raw")
    df["text_clean"] = df["text"].map(clean_text)
    texts = df["text_clean"].astype(str).tolist()
    ticker_texts = df.get("text_for_ticker", df["text"]).map(clean_text).astype(str).tolist()

    total = len(texts)
    scores, local_sentiment_meta = _score_sentiment_with_cache(cfg, texts, progress=progress, stage="local_sentiment", use_cache=bool(use_cache))
    df["sentiment"] = scores
    df["sentiment_intensity"] = df["sentiment"].map(abs)

    _emit_progress(progress, "Локальный шаг дискретизации: извлекаю тикеры и сущности", 3, 7, stage="local_tickers")
    ticker_enabled = str(cfg.get("nlp", "ticker_extraction", default="hybrid_fast")).lower() != "none"
    if ticker_enabled:
        universe = load_ticker_universe(cfg, refresh=False)
        payloads = extract_ticker_candidates_many(ticker_texts, universe=universe, cfg=cfg, progress=progress, stage="local_tickers")
        ticker_method_stats = _ticker_payload_method_stats(payloads)
        df["ticker_mentions"] = [json.dumps(xs, ensure_ascii=False) for xs in payloads]
        df["tickers"] = [",".join([item["ticker"] for item in xs if _accept_ticker_candidate_for_analytics(item)]) for xs in payloads]
    else:
        payloads = []
        ticker_method_stats = {}
        df["ticker_mentions"] = "[]"
        df["tickers"] = ""
    df = attach_ticker_thread_participation(df)

    _emit_progress(progress, "Локальный шаг дискретизации: строю темы на контексте без будущих данных", 4, 7, stage="local_topics")
    topic_service = TopicService(cfg)
    if len(texts) < max(3, int(cfg.get("nlp", "min_topic_size", default=20))):
        df["topic"] = 0
        df["topic_modeling"] = 0
        df["topic_detailed"] = 0
        df["topic_indicator"] = 0
        topic_info = pd.DataFrame({"Topic": [0], "Count": [len(df)]})
        topic_service.backend = "single_topic"
    else:
        labels, topic_info = topic_service.fit_transform(texts)
        df["topic"] = labels
        detailed_labels = getattr(topic_service, "detailed_labels", None)
        if detailed_labels is not None and len(detailed_labels) == len(df):
            df["topic_detailed"] = [int(x) for x in detailed_labels]
            df["topic_indicator"] = df["topic_detailed"]
        else:
            df["topic_detailed"] = df["topic"]
            df["topic_indicator"] = df["topic"]
        df["topic_modeling"] = df["topic"]

    _emit_progress(progress, "Локальный шаг дискретизации: пересчитываю динамическую авторитетность", 5, 7, stage="local_influence")
    df, influence_meta = assign_dynamic_window_influence(df, cfg, window_hours=window_hours, progress=progress)

    ws, wi = _normalize_weights(w_sentiment, w_influence)
    payoff = compute_payoff_from_messages(df, cfg, ws, wi, sentiment_mode or cfg.get("payoff", "sentiment_mode", default="magnitude"))
    strategy_topics = [int(x) for x in payoff["topic"].tolist()] or [0]
    current = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    payload = _window_payload_from_part(current, start_ts, strategy_topics)
    payload = _merge_context_ticker_summary(payload, df, start_ts, window_hours)
    payload = _attach_ticker_activity_baseline(payload, df, start_ts, window_hours)
    # добавляем метаданные контекста для интерпретации в интерфейсе
    payload.update(local_meta)
    payload.update({
        "context_start": context_start.isoformat(),
        "window_end": end_ts.isoformat(),
        "context_n_posts": int(len(df)),
        "dynamic_state_mode": "raw_data_local_enrichment_with_adaptive_topic_space",
        "influence_metadata": {k: v for k, v in influence_meta.items() if k != "window_summaries"},
        "used_pre_enriched_messages": False,
        "ticker_ner_mode": str(cfg.get("nlp", "ticker_extraction", default="hybrid_fast")),
        "ticker_ner_method_stats": ticker_method_stats,
        "sentiment_meta": local_sentiment_meta,
        "simulation_topic_mode": str(cfg.get("nlp", "simulation_topic_mode", default="detailed")),
        "detailed_topics_preserved": bool("topic_detailed" in df.columns),
    })
    _emit_progress(progress, "Локальный шаг дискретизации: состояние рассчитано", 7, 7, stage="local_raw")
    result = {"window": payload, "payoff": payoff, "context": df, "topic_service": topic_service, "topic_info": topic_info}
    if use_cache:
        try:
            pd.to_pickle(result, cache_file)
        except Exception:
            pass
    return result


def run_predictive_market_indicator_for_raw_range(
    cfg: AppConfig,
    raw_path: str | Path,
    start_time: Any,
    window_hours: int,
    future_steps: int,
    prob_revision: float,
    alpha: float | None,
    noise: float,
    w_sentiment: float = 0.5,
    w_influence: float = 0.5,
    sentiment_mode: str | None = None,
    output_path: str | Path | None = None,
    context_windows: int = 8,
    use_cache: bool = True,
    progress=None,
) -> pd.DataFrame:
    """Запускает полный локальный анализ выбранного диапазона исходных сообщений
    
    Args:
        cfg: конфигурация проекта или приложения
        raw_path: путь к исходной таблице сообщений
        start_time: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
        future_steps: число будущих шагов дискретизации для симуляции
        prob_revision: вероятность пересмотра стратегии агентом
        alpha: параметр баланса собственной и наблюдаемой стратегии
        noise: вероятность случайного выбора стратегии
        w_sentiment: вес тональности в функции выигрыша
        w_influence: вес авторитетности в функции выигрыша
        sentiment_mode: режим преобразования тональности
        output_path: путь для сохранения результата
        context_windows: число предыдущих шагов дискретизации для исторического контекста
        use_cache: признак использования сохранённого кэша расчётов
        progress: callback для передачи статуса выполнения
    
    Returns:
        словарь результатов локального анализа исходного диапазона
    """
    built = _enrich_raw_slice_for_window(
        cfg, raw_path, start_time, window_hours, context_windows,
        w_sentiment=w_sentiment, w_influence=w_influence,
        sentiment_mode=sentiment_mode,
        use_cache=bool(use_cache),
        progress=progress,
    )
    start_window = built["window"]
    payoff_df = built["payoff"]
    if int(start_window.get("n_posts", 0)) <= 0:
        return _empty_forward_indicator(output_path)
    strategy_topics = [int(t) for t in payoff_df["topic"].tolist()] or [0]
    ticker_summary = start_window.get("ticker_summary", {}) or {}
    runner = NetLogoRunner(NetLogoConfig(**(cfg.get("netlogo", default={}) or {})))
    pop_size = int(cfg.get("netlogo", "pop_size", default=1000))
    payoff_array = load_payoff_from_df(payoff_df, strategy_topics)
    initial = distribution_to_counts(start_window.get("real_props", []), pop_size)
    sim = runner.run(payoff=payoff_array, initial_distribution=initial, n_steps=int(future_steps), prob_revision=float(prob_revision), noise=float(noise), alpha=alpha)
    prop_cols = [c for c in sim.columns if c.startswith("prop_")]
    raw_path = _validate_raw_csv_path(raw_path)
    raw = _load_normalized_raw_data(raw_path, use_cache=bool(use_cache))
    raw["text_clean"] = raw["text"].map(clean_text)
    topic_service = built.get("topic_service")
    start_ts = pd.Timestamp(pd.to_datetime(start_time, errors="coerce")).tz_localize(None)
    start_props = np.asarray(start_window.get("real_props", []), dtype=float)
    topic_ids = [str(t) for t in strategy_topics]
    topic_shift_history = start_window.get("model_topic_shift_history", []) or _topic_shift_history_from_context_df(built.get("context"), start_ts, int(window_hours))
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
        future_end = future_start + pd.Timedelta(hours=int(window_hours))
        future_raw = raw[(raw["timestamp"] >= future_start) & (raw["timestamp"] < future_end)].copy()
        actual_available = not future_raw.empty
        realized = {"rmse": None, "nrmse": None, "l1": None, "cosine": None}
        if actual_available:
            future_texts = future_raw["text_clean"].astype(str).tolist()
            if topic_service is not None and hasattr(topic_service, "transform"):
                future_raw["topic"] = topic_service.transform(future_texts)
            else:
                future_raw["topic"] = 0
            future_raw["sentiment"] = 0.0
            future_raw["sentiment_intensity"] = 0.0
            future_raw["influence"] = 0.0
            future_window = _window_payload_from_part(future_raw, future_start, strategy_topics)
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
    df = _finalize_forward_indicator(rows, output_path)
    output_path = Path(output_path)
    try:
        context_posts = built.get("context")
        if context_posts is not None and not pd.DataFrame(context_posts).empty:
            context_df = pd.DataFrame(context_posts).copy()
            context_df.to_csv(output_path.with_name(output_path.stem + "_context_messages.csv"), index=False)
            # экспорт симуляции содержит только модельные и наблюдаемые на старте поля
            # колонки фактической будущей проверки намеренно удалены
            # так файл можно использовать в текущем анализе без утечки будущих данных
            forecast_cols = [c for c in df.columns if c not in {"realized_nrmse", "realized_l1", "realized_rmse", "ticker_realized_nrmse", "ticker_realized_l1", "ticker_realized_rmse"}]
            df[forecast_cols].to_csv(output_path.with_name(output_path.stem + "_model_forecast.csv"), index=False)
            ticker_rows = df[df["ticker"].astype(str) != DISCUSSION_LEVEL_TICKER]
            top_tickers = (ticker_rows.groupby("ticker", as_index=False)["forward_discussion_indicator"].max()
                .sort_values("forward_discussion_indicator", ascending=False)
                .head(8)["ticker"].astype(str).tolist()) if not ticker_rows.empty else []
            # публикации текущего шага полезны
            # для адаптивного локального анализа также важен исторический контекст, сохраняем оба среза
            current_posts = _top_publications_from_frame(context_df, start_window.get("time"), int(window_hours), tickers=top_tickers, top_n=40)
            current_posts.to_csv(output_path.with_name(output_path.stem + "_publications.csv"), index=False)
            ctx_start = start_window.get("context_start") or start_window.get("time")
            ctx_end = start_window.get("window_end") or start_window.get("planned_end_time") or start_window.get("time")
            context_posts_export = _top_publications_from_range_frame(context_df, ctx_start, ctx_end, tickers=top_tickers, top_n=120, source_period="исторический контекст и текущий шаг дискретизации")
            context_posts_export.to_csv(output_path.with_name(output_path.stem + "_context_publications.csv"), index=False)
            topic_info = built.get("topic_info")
            if topic_info is not None and not pd.DataFrame(topic_info).empty:
                pd.DataFrame(topic_info).to_csv(output_path.with_name(output_path.stem + "_topic_info.csv"), index=False)
            ticker_summary = start_window.get("ticker_summary", {}) or {}
            if ticker_summary:
                rows = []
                for ticker, info in ticker_summary.items():
                    row = {"ticker": ticker}
                    row.update({k: v for k, v in info.items() if not isinstance(v, (dict, list, set))})
                    row["topic_shares"] = json.dumps(info.get("topic_shares", {}), ensure_ascii=False)
                    row["methods"] = json.dumps(info.get("methods", {}), ensure_ascii=False)
                    rows.append(row)
                pd.DataFrame(rows).to_csv(output_path.with_name(output_path.stem + "_ticker_summary.csv"), index=False)
    except Exception:
        pass
    return df
