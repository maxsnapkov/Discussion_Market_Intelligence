"""Инициализация проекта и построение шагов дискретизации"""
from __future__ import annotations


from ..pipeline.common import *
from ..pipeline.configuration import AppConfig, _emit_progress, ensure_dirs, load_raw_messages, clean_text
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..discussion.influence_payoff import assign_dynamic_window_influence, compute_payoff_from_messages
    from ..nlp.sentiment import _coerce_existing_sentiment, _existing_sentiment_column, _score_sentiment_with_cache
    from ..nlp.ticker_extraction import (
        _accept_ticker_candidate_for_analytics,
        _row_thread_root_id,
        _ticker_participation_mentions,
        attach_ticker_thread_participation,
        extract_ticker_candidates_many,
        load_ticker_universe,
    )
    from ..nlp.topic_modeling import TopicService

def initialize_project(
    cfg: AppConfig,
    raw_path: str | Path,
    limit: int | None = None,
    sample: int | None = None,
    random_state: int = 42,
    window_hours: int | None = None,
    w_sentiment: float | None = None,
    w_influence: float | None = None,
    sentiment_mode: str | None = None,
    use_cache: bool = True,
    progress=None,
) -> dict[str, Any]:
    """Выполняет полный подготовительный конвейер от исходных сообщений до файлов модели
    
    Args:
        cfg: конфигурация проекта или приложения
        raw_path: путь к исходной таблице сообщений
        limit: максимальное число записей или символов
        sample: размер случайной подвыборки
        random_state: случайное зерно для воспроизводимости
        window_hours: длительность шага дискретизации в часах
        w_sentiment: вес тональности в функции выигрыша
        w_influence: вес авторитетности в функции выигрыша
        sentiment_mode: режим преобразования тональности
        use_cache: признак использования сохранённого кэша расчётов
        progress: callback для передачи статуса выполнения
    
    Returns:
        словарь путей и таблиц, созданных подготовительным конвейером
    """
    ensure_dirs(cfg)
    processed_dir = cfg.data_dir / "processed"
    model_dir = cfg.model_dir

    _emit_progress(progress, "Чтение и нормализация данных", stage="load")
    df = load_raw_messages(raw_path, limit=limit, sample=sample, random_state=random_state)
    df["text_clean"] = df["text"].map(clean_text)

    stage_timings: dict[str, float] = {}
    texts = df["text_clean"].astype(str).tolist()
    ticker_texts = df.get("text_for_ticker", df["text"]).map(clean_text).astype(str).tolist()

    # тональность и связывание тикеров считаем отдельными этапами
    # тональности нужны небольшие батчи трансформера
    # а тикеры быстро ищутся на CPU большими фрагментами
    # объединение этапов мешало профилированию и замедляло быстрый путь тональности
    sent_t0 = time.perf_counter()
    sentiment_policy = str(cfg.get("nlp", "sentiment_policy", default="reuse_or_compute") or "reuse_or_compute").lower()
    existing_sentiment = _existing_sentiment_column(df)
    sentiment_meta: dict[str, Any] = {"policy": sentiment_policy, "existing_column": existing_sentiment}
    if existing_sentiment and sentiment_policy in {"reuse_or_compute", "reuse_only"}:
        _emit_progress(progress, f"Тональность: использую готовую колонку {existing_sentiment}", len(df), len(df), stage="sentiment")
        df["sentiment"] = _coerce_existing_sentiment(df[existing_sentiment])
        sentiment_meta.update({"source": "existing_column", "backend": "reused"})
    elif sentiment_policy == "reuse_only":
        _emit_progress(progress, "Тональность: готовая колонка не найдена, ставлю нейтральные значения", len(df), len(df), stage="sentiment")
        df["sentiment"] = 0.0
        sentiment_meta.update({"source": "neutral_no_existing_column", "backend": "none"})
    elif sentiment_policy == "skip":
        _emit_progress(progress, "Тональность: этап пропущен, используются нейтральные значения", len(df), len(df), stage="sentiment")
        df["sentiment"] = 0.0
        sentiment_meta.update({"source": "skipped", "backend": "none"})
    else:
        if use_cache:
            _emit_progress(progress, "Тональность: batch-обработка сообщений с кэшем", 0, len(df), stage="sentiment")
        else:
            _emit_progress(progress, "Тональность: batch-обработка сообщений без кэша", 0, len(df), stage="sentiment")
        scores, sentiment_meta = _score_sentiment_with_cache(cfg, texts, progress=progress, stage="sentiment", use_cache=bool(use_cache))
        sentiment_meta.update({"policy": sentiment_policy, "source": "computed_or_cached"})
        df["sentiment"] = scores
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0.0).astype(float).clip(-1.0, 1.0)
    df["sentiment_intensity"] = df["sentiment"].map(abs)
    stage_timings["sentiment_seconds"] = round(time.perf_counter() - sent_t0, 3)

    ticker_t0 = time.perf_counter()
    ticker_enabled = str(cfg.get("nlp", "ticker_extraction", default="hybrid_fast")).lower() != "none"
    if ticker_enabled:
        _emit_progress(progress, "Тикеры: загрузка справочника и быстрый entity linking", 0, len(df), stage="tickers")
        universe = load_ticker_universe(cfg, refresh=False)
        ticker_payloads = extract_ticker_candidates_many(ticker_texts, universe=universe, cfg=cfg, progress=progress, stage="tickers")
        df["ticker_mentions"] = [json.dumps(xs, ensure_ascii=False) for xs in ticker_payloads]
        df["tickers"] = [",".join([item["ticker"] for item in xs if _accept_ticker_candidate_for_analytics(item)]) for xs in ticker_payloads]
    else:
        df["ticker_mentions"] = "[]"
        df["tickers"] = ""
    df = attach_ticker_thread_participation(df)
    stage_timings["ticker_seconds"] = round(time.perf_counter() - ticker_t0, 3)

    _emit_progress(progress, "Обучение тематической модели", stage="topics")
    topics = TopicService(cfg)
    labels, topic_info = topics.fit_transform(texts)
    df["topic"] = labels
    # topic является активным пространством тем для агентной симуляции
    # при укрупнении тем сохраняем исходные детальные метки
    # они нужны для интерпретации индикаторов и проверки
    detailed_labels = getattr(topics, "detailed_labels", None)
    if detailed_labels is not None and len(detailed_labels) == len(df):
        df["topic_detailed"] = [int(x) for x in detailed_labels]
        df["topic_indicator"] = df["topic_detailed"]
    else:
        df["topic_detailed"] = df["topic"]
        df["topic_indicator"] = df["topic"]
    df["topic_modeling"] = df["topic"]
    topics.save(model_dir)
    topic_info_path = processed_dir / "topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False)
    detailed_info_path = processed_dir / "topic_info_detailed.csv"
    detailed_info = getattr(topics, "detailed_info", None)
    if detailed_info is not None:
        try:
            pd.DataFrame(detailed_info).to_csv(detailed_info_path, index=False)
        except Exception:
            pass

    topic_mapping_path = processed_dir / "topic_mapping.csv"
    try:
        topic_mapping = (
            df.groupby(["topic", "topic_detailed"], dropna=False)
            .size()
            .reset_index(name="message_count")
            .sort_values(["topic", "message_count", "topic_detailed"], ascending=[True, False, True])
        )
        topic_mapping["share_in_large_group"] = topic_mapping["message_count"] / topic_mapping.groupby("topic")["message_count"].transform("sum").replace(0, np.nan)
        topic_mapping["share_in_large_group"] = topic_mapping["share_in_large_group"].fillna(0.0)
        topic_mapping = topic_mapping.rename(columns={"topic": "large_topic", "topic_detailed": "detailed_topic"})
        topic_mapping.to_csv(topic_mapping_path, index=False)
    except Exception:
        pd.DataFrame(columns=["large_topic", "detailed_topic", "message_count", "share_in_large_group"]).to_csv(topic_mapping_path, index=False)

    _emit_progress(progress, "Расчёт payoff", stage="payoff")

    selected_window_hours = int(window_hours or cfg.window_hours)
    _emit_progress(progress, "Динамический пересчёт авторитетности по шагам дискретизации", stage="influence")
    df, influence_meta = assign_dynamic_window_influence(df, cfg, selected_window_hours, progress=progress)

    w_sent = float(cfg.get("payoff", "w_sentiment", default=0.4) if w_sentiment is None else w_sentiment)
    w_inf = float(cfg.get("payoff", "w_influence", default=0.6) if w_influence is None else w_influence)
    selected_sentiment_mode = sentiment_mode or cfg.get("payoff", "sentiment_mode", default="magnitude")
    payoff = compute_payoff_from_messages(df, cfg, w_sent, w_inf, selected_sentiment_mode)
    payoff.to_csv(processed_dir / "payoff.csv", index=False)

    windows = build_windows(df, selected_window_hours, strategy_topics=payoff["topic"].astype(int).tolist())
    windows_path = processed_dir / "windows.json"
    windows_path.write_text(json.dumps(windows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    df.to_csv(processed_dir / "messages_enriched.csv", index=False)
    with open(processed_dir / "author_influence.json", "w", encoding="utf-8") as f:
        json.dump(influence_meta, f, ensure_ascii=False, indent=2)

    state = {
        "raw_path": str(raw_path),
        "n_messages": int(len(df)),
        "n_windows": int(len(windows)),
        "n_topics": int(len(payoff)),
        "n_unique_tickers": int(len(set(t for value in df.get("tickers", pd.Series(dtype=str)).fillna("") for t in str(value).split(",") if t))),
        "window_hours": int(selected_window_hours),
        "w_sentiment": float(w_sent),
        "w_influence": float(w_inf),
        "sentiment_mode": selected_sentiment_mode,
        "stage_timings": stage_timings,
        "ticker_extraction_mode": str(cfg.get("nlp", "ticker_extraction", default="hybrid_fast")),
        "influence_mode": influence_meta.get("mode", "dynamic_decayed_pagerank"),
        "influence_decay_per_window": influence_meta.get("decay_per_window"),
        "influence_pagerank_weight": influence_meta.get("pagerank_weight"),
        "topic_backend": topics.backend,
        "simulation_topic_mode": str(cfg.get("nlp", "simulation_topic_mode", default="detailed")),
        "simulation_topic_count": int(cfg.get("nlp", "simulation_topic_count", default=6)),
        "detailed_topics_preserved": bool("topic_detailed" in df.columns),
        "n_detailed_topics": int(pd.Series(df.get("topic_detailed", df["topic"])).nunique()),
        "sentiment_meta": sentiment_meta,
        "processed_messages": str(processed_dir / "messages_enriched.csv"),
        "windows": str(windows_path),
        "payoff": str(processed_dir / "payoff.csv"),
        "topic_model_dir": str(model_dir),
        "topic_info": str(topic_info_path),
        "topic_info_detailed": str(detailed_info_path),
        "topic_mapping": str(topic_mapping_path),
    }
    (cfg.output_dir / "project_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state



def parse_ticker_mentions(value: Any) -> list[dict[str, Any]]:
    """Разбирает сохранённые тикерные упоминания из строки или структуры данных
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        список нормализованных тикерных упоминаний
    """
    if isinstance(value, list):
        raw = value
    elif value is None or (isinstance(value, float) and math.isnan(value)):
        raw = []
    else:
        text = str(value).strip()
        if not text:
            raw = []
        else:
            try:
                raw = json.loads(text)
            except Exception:
                try:
                    raw = ast.literal_eval(text)
                except Exception:
                    raw = []
    out: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        try:
            confidence = float(item.get("confidence", 1.0))
        except Exception:
            confidence = 1.0
        try:
            context_score = float(item.get("context_score", 0.5))
        except Exception:
            context_score = 0.5
        out.append({
            "ticker": ticker,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "method": str(item.get("method", "unknown")),
            "evidence": str(item.get("evidence", ""))[:120],
            "context_score": float(max(0.0, min(1.0, context_score))),
            "is_ambiguous": bool(item.get("is_ambiguous", False)),
        })
    return out


def _window_ticker_analytics(part: pd.DataFrame) -> dict[str, Any]:
    """Рассчитывает тикерную статистику внутри одного шага дискретизации
    
    Args:
        part: фрагмент таблицы сообщений внутри шага дискретизации
    
    Returns:
        словарь тикерной статистики текущего шага
    """
    ticker_weight: dict[str, float] = {}
    ticker_sent_sum: dict[str, float] = {}
    ticker_sent_int_sum: dict[str, float] = {}
    ticker_infl_sum: dict[str, float] = {}
    ticker_topic_weight: dict[str, dict[str, float]] = {}
    ticker_topic_weight_detailed: dict[str, dict[str, float]] = {}
    ticker_methods: dict[str, dict[str, float]] = {}
    ticker_conf_sum: dict[str, float] = {}
    ticker_ctx_sum: dict[str, float] = {}
    ticker_ambiguous: dict[str, bool] = {}
    ticker_post_count: dict[str, int] = {}
    ticker_authors: dict[str, set[str]] = {}
    ticker_root_threads: dict[str, set[str]] = {}

    n_posts = int(len(part))
    n_authors = int(part["author"].astype(str).nunique()) if n_posts and "author" in part.columns else 0

    participation_payloads = _ticker_participation_mentions(part)
    direct_post_count: dict[str, int] = {}
    inherited_post_count: dict[str, int] = {}

    for (_, row), participation_mentions in zip(part.iterrows(), participation_payloads):
        mentions = [m for m in participation_mentions if _accept_ticker_candidate_for_analytics(m)]
        if not mentions:
            continue
        sentiment = float(row.get("sentiment", 0.0) or 0.0)
        sent_int = abs(sentiment)
        influence = float(row.get("influence", 0.0) or 0.0)
        topic = str(int(row.get("topic", -1))) if pd.notna(row.get("topic", np.nan)) else "-1"
        detailed_topic_value = row.get("topic_indicator", row.get("topic_detailed", row.get("topic", -1)))
        topic_detailed = str(int(detailed_topic_value)) if pd.notna(detailed_topic_value) else topic
        author = str(row.get("author", row.get("author_name", "")) or "").strip()
        total_conf = sum(max(0.0, float(m.get("confidence", 0.0))) for m in mentions) or 1.0
        row_tickers: set[str] = set()
        for m in mentions:
            ticker = str(m.get("ticker", "")).upper().strip()
            if not ticker:
                continue
            conf = max(0.0, min(1.0, float(m.get("confidence", 1.0))))
            ctx = max(0.0, min(1.0, float(m.get("context_score", 0.5) or 0.0)))
            weight = conf / total_conf
            ticker_weight[ticker] = ticker_weight.get(ticker, 0.0) + weight
            ticker_sent_sum[ticker] = ticker_sent_sum.get(ticker, 0.0) + sentiment * weight
            ticker_sent_int_sum[ticker] = ticker_sent_int_sum.get(ticker, 0.0) + sent_int * weight
            ticker_infl_sum[ticker] = ticker_infl_sum.get(ticker, 0.0) + influence * weight
            ticker_conf_sum[ticker] = ticker_conf_sum.get(ticker, 0.0) + conf * weight
            ticker_ctx_sum[ticker] = ticker_ctx_sum.get(ticker, 0.0) + ctx * weight
            ticker_ambiguous[ticker] = ticker_ambiguous.get(ticker, False) or bool(m.get("is_ambiguous", False))
            ticker_topic_weight.setdefault(ticker, {})[topic] = ticker_topic_weight.setdefault(ticker, {}).get(topic, 0.0) + weight
            ticker_topic_weight_detailed.setdefault(ticker, {})[topic_detailed] = ticker_topic_weight_detailed.setdefault(ticker, {}).get(topic_detailed, 0.0) + weight
            method = str(m.get("method", "unknown"))
            ticker_methods.setdefault(ticker, {})[method] = ticker_methods.setdefault(ticker, {}).get(method, 0.0) + weight
            row_tickers.add(ticker)
            root_id = str(m.get("thread_root_id", "") or _row_thread_root_id(row)).strip()
            if root_id:
                ticker_root_threads.setdefault(ticker, set()).add(root_id)
            if str(m.get("participation_role", "direct")) == "direct":
                direct_post_count[ticker] = direct_post_count.get(ticker, 0) + 1
            else:
                inherited_post_count[ticker] = inherited_post_count.get(ticker, 0) + 1
        for ticker in row_tickers:
            ticker_post_count[ticker] = ticker_post_count.get(ticker, 0) + 1
            ticker_authors.setdefault(ticker, set()).add(author)

    total = sum(ticker_weight.values()) or 0.0
    top = dict(sorted(ticker_weight.items(), key=lambda kv: -kv[1])[:10])
    shares = {t: (w / total if total > 0 else 0.0) for t, w in ticker_weight.items()}
    summary: dict[str, dict[str, Any]] = {}
    for ticker, weight in ticker_weight.items():
        topics = ticker_topic_weight.get(ticker, {})
        main_topic = max(topics.items(), key=lambda kv: kv[1])[0] if topics else None
        topic_total = sum(topics.values()) or 1.0
        topic_shares = {k: v / topic_total for k, v in sorted(topics.items(), key=lambda kv: -kv[1])[:5]}
        detailed_topics = ticker_topic_weight_detailed.get(ticker, {})
        detailed_topic_total = sum(detailed_topics.values()) or 1.0
        topic_shares_detailed = {k: v / detailed_topic_total for k, v in sorted(detailed_topics.items(), key=lambda kv: -kv[1])[:10]}
        main_topic_detailed = max(detailed_topics.items(), key=lambda kv: kv[1])[0] if detailed_topics else main_topic
        posts = int(ticker_post_count.get(ticker, 0))
        authors = int(len(ticker_authors.get(ticker, set())))
        direct_posts = int(direct_post_count.get(ticker, 0))
        inherited_posts = int(inherited_post_count.get(ticker, 0))
        root_threads = int(len(ticker_root_threads.get(ticker, set())))
        confidence_avg = float(ticker_conf_sum.get(ticker, 0.0) / max(weight, 1e-12))
        context_avg = float(ticker_ctx_sum.get(ticker, 0.0) / max(weight, 1e-12))
        summary[ticker] = {
            "mention_weight": float(weight),
            "share": float(shares.get(ticker, 0.0)),
            "post_count": posts,
            "unique_authors": authors,
            "direct_post_count": direct_posts,
            "inherited_post_count": inherited_posts,
            "root_thread_count": root_threads,
            "participation_mode": "direct_and_descendant_thread",
            "activity_rate": float(posts / max(1, n_posts)),
            "author_rate": float(authors / max(1, n_authors)),
            "support_score": None,
            "avg_sentiment": float(ticker_sent_sum.get(ticker, 0.0) / max(weight, 1e-12)),
            "sentiment_intensity": float(ticker_sent_int_sum.get(ticker, 0.0) / max(weight, 1e-12)),
            "avg_influence": float(ticker_infl_sum.get(ticker, 0.0) / max(weight, 1e-12)),
            "confidence_avg": confidence_avg,
            "context_score_avg": context_avg,
            "is_ambiguous": bool(ticker_ambiguous.get(ticker, False)),
            "main_topic": main_topic,
            "topic_shares": topic_shares,
            "main_topic_detailed": main_topic_detailed,
            "topic_shares_detailed": topic_shares_detailed,
            "methods": dict(sorted(ticker_methods.get(ticker, {}).items(), key=lambda kv: -kv[1])[:4]),
        }
    return {
        "top_tickers": {k: float(v) for k, v in top.items()},
        "ticker_total_weight": float(total),
        "ticker_concentration": float(max(shares.values()) if shares else 0.0),
        "ticker_summary": summary,
    }




def _merge_context_ticker_summary(
    payload: dict[str, Any],
    context_df: pd.DataFrame,
    start_ts: pd.Timestamp,
    window_hours: int,
) -> dict[str, Any]:
    """Объединяет текущую тикерную статистику с историческим контекстом
    
    Args:
        payload: словарь промежуточных результатов шага дискретизации
        context_df: таблица сообщений исторического контекста
        start_ts: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
    
    Returns:
        payload с добавленной исторической сводкой по тикерам
    """
    if context_df is None or pd.DataFrame(context_df).empty:
        return payload
    data = pd.DataFrame(context_df).copy()
    if "timestamp" not in data.columns:
        return payload
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    end_ts = pd.Timestamp(start_ts) + pd.Timedelta(hours=int(window_hours))
    ctx = data[data["timestamp"] < end_ts].copy()
    if ctx.empty:
        return payload
    ctx_analytics = _window_ticker_analytics(ctx)
    ctx_summary = ctx_analytics.get("ticker_summary", {}) or {}
    current_summary = payload.get("ticker_summary", {}) or {}
    ctx_posts_total = max(1, int(len(ctx)))
    ctx_authors_total = max(1, int(ctx["author"].astype(str).nunique()) if "author" in ctx.columns else 1)
    for ticker, ctx_info in ctx_summary.items():
        ticker = str(ticker).upper().strip()
        if not ticker:
            continue
        ctx_posts = int(ctx_info.get("post_count", 0) or 0)
        ctx_authors = int(ctx_info.get("unique_authors", 0) or 0)
        ctx_share = float(ctx_posts / ctx_posts_total)
        if ticker in current_summary:
            info = current_summary[ticker]
            info.setdefault("source_scope", "current_and_context")
            info["current_post_count"] = int(info.get("post_count", 0) or 0)
            info["current_unique_authors"] = int(info.get("unique_authors", 0) or 0)
            info["context_post_count"] = ctx_posts
            info["context_unique_authors"] = ctx_authors
            info["context_direct_post_count"] = int(ctx_info.get("direct_post_count", 0) or 0)
            info["context_inherited_post_count"] = int(ctx_info.get("inherited_post_count", 0) or 0)
            info["context_root_thread_count"] = int(ctx_info.get("root_thread_count", 0) or 0)
            info["context_share"] = ctx_share
            info["historical_support_score"] = None
            # сохраняем тематический профиль текущего шага при наличии
            # профиль контекста используем как резерв для устойчивой модельной экспозиции
            info.setdefault("context_topic_shares", ctx_info.get("topic_shares", {}))
            if not info.get("topic_shares"):
                info["topic_shares"] = ctx_info.get("topic_shares", {})
            methods = dict(ctx_info.get("methods", {}) or {})
            methods.update(dict(info.get("methods", {}) or {}))
            info["methods"] = methods
        else:
            info = dict(ctx_info)
            info["source_scope"] = "historical_context"
            info["current_post_count"] = 0
            info["current_unique_authors"] = 0
            info["context_post_count"] = ctx_posts
            info["context_unique_authors"] = ctx_authors
            info["context_direct_post_count"] = int(ctx_info.get("direct_post_count", 0) or 0)
            info["context_inherited_post_count"] = int(ctx_info.get("inherited_post_count", 0) or 0)
            info["context_root_thread_count"] = int(ctx_info.get("root_thread_count", 0) or 0)
            info["context_share"] = ctx_share
            info["historical_support_score"] = None
            info["context_topic_shares"] = ctx_info.get("topic_shares", {})
            # текущая активность равна нулю
            # такие тикеры остаются кандидатами наблюдения по контексту и модельной экспозиции
            info["post_count"] = 0
            info["unique_authors"] = 0
            info["direct_post_count"] = 0
            info["inherited_post_count"] = 0
            info["root_thread_count"] = 0
            info["share"] = 0.0
            info["activity_rate"] = 0.0
            info["author_rate"] = 0.0
            info["support_score"] = None
            current_summary[ticker] = info
    payload["ticker_summary"] = current_summary
    # ведущие тикеры образуют смешанный список текущей и контекстной поддержки
    # сортируем сначала по текущей поддержке, затем по исторической
    ranked = sorted(
        current_summary.items(),
        key=lambda kv: (
            float(kv[1].get("post_count", 0.0) or 0.0),
            float(kv[1].get("unique_authors", 0.0) or 0.0),
            float(kv[1].get("context_share", 0.0) or 0.0),
        ),
        reverse=True,
    )
    payload["top_tickers"] = {ticker: float(info.get("mention_weight", 0.0) or info.get("context_share", 0.0) or 0.0) for ticker, info in ranked[:10]}
    return payload


def _ticker_payload_method_stats(payloads: list[list[dict[str, Any]]]) -> dict[str, int]:
    """Подсчитывает источники обнаружения тикеров внутри набора payload-записей
    
    Args:
        payloads: набор словарей промежуточных результатов
    
    Returns:
        счётчик методов обнаружения тикеров
    """
    counts: dict[str, int] = {}
    for items in payloads or []:
        for item in items or []:
            method = str(item.get("method", "unknown"))
            counts[method] = counts.get(method, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

def build_windows(df: pd.DataFrame, window_hours: int, strategy_topics: list[int]) -> list[dict[str, Any]]:
    """Агрегирует сообщения в шаги дискретизации и сохраняет тематические распределения
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
        window_hours: длительность шага дискретизации в часах
        strategy_topics: список тем, используемых как стратегии агентной модели
    
    Returns:
        таблица шагов дискретизации с распределениями тем и тикерной статистикой
    """
    data = df.copy()
    data["window_start"] = data["timestamp"].dt.floor(f"{int(window_hours)}h")
    topic_to_idx = {int(t): i for i, t in enumerate(strategy_topics)}
    windows: list[dict[str, Any]] = []
    for ts, part in data.groupby("window_start", sort=True):
        counts = np.zeros(len(strategy_topics), dtype=float)
        for topic, c in part["topic"].value_counts().items():
            if int(topic) in topic_to_idx:
                counts[topic_to_idx[int(topic)]] = float(c)
        total = float(counts.sum())
        props = (counts / total).tolist() if total > 0 else [0.0] * len(strategy_topics)
        ticker_data = _window_ticker_analytics(part)
        author_counts = part["author"].value_counts()
        if "influence" in part.columns and len(part):
            influence_by_author = part.groupby("author")["influence"].mean().sort_values(ascending=False).head(10).to_dict()
        else:
            influence_by_author = {}
        coverage = float((part["topic"] != -1).mean()) if len(part) else 0.0
        emerging_share = float((part["topic"] == -1).mean()) if len(part) else 0.0
        detailed_counts_payload: dict[str, int] = {}
        detailed_shares_payload: dict[str, float] = {}
        detailed_col = "topic_indicator" if "topic_indicator" in part.columns else "topic_detailed" if "topic_detailed" in part.columns else None
        if detailed_col:
            detailed_counts = part[detailed_col].value_counts().sort_index()
            detailed_total = float(detailed_counts.sum()) or 1.0
            detailed_counts_payload = {str(int(k)): int(v) for k, v in detailed_counts.items() if pd.notna(k)}
            detailed_shares_payload = {str(int(k)): float(v) / detailed_total for k, v in detailed_counts.items() if pd.notna(k)}
        windows.append({
            "time": pd.Timestamp(ts).isoformat(),
            "n_posts": int(len(part)),
            "strategy_topics": strategy_topics,
            "topic_counts": counts.astype(int).tolist(),
            "real_props": props,
            "detailed_topic_counts": detailed_counts_payload,
            "detailed_topic_shares": detailed_shares_payload,
            "avg_sentiment": float(part["sentiment"].mean()),
            "sentiment_intensity": float(part["sentiment_intensity"].mean()),
            "avg_influence": float(part["influence"].mean()),
            "coverage": coverage,
            "emerging_topic_share": emerging_share,
            "top_tickers": ticker_data["top_tickers"],
            "ticker_total_weight": ticker_data["ticker_total_weight"],
            "ticker_concentration": ticker_data["ticker_concentration"],
            "ticker_summary": ticker_data["ticker_summary"],
            "top_authors": author_counts.head(10).to_dict(),
            "top_authors_by_influence": {str(k): float(v) for k, v in influence_by_author.items()},
            "author_concentration": float(author_counts.iloc[0] / len(part)) if len(part) and len(author_counts) else 0.0,
        })
    return windows

def load_windows(path: str | Path) -> list[dict[str, Any]]:
    """Загружает таблицу шагов дискретизации
    
    Args:
        path: путь к файлу или директории
    
    Returns:
        таблица шагов дискретизации из CSV-файла
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))
