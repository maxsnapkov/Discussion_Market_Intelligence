"""Авторитетность участников и расчёт выигрышей тем"""
from __future__ import annotations


from ..pipeline.common import *
from ..pipeline.configuration import AppConfig, _emit_progress, is_service_author

def build_influence(df: pd.DataFrame) -> dict[str, float]:
    """Строит базовую авторитетность авторов по активности и оценкам сообщений
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        словарь авторитетности авторов
    """
    authors = df["author"].fillna("unknown").astype(str)
    edge_cols = [c for c in ["parent_author", "reply_to_author", "target_author", "quoted_author"] if c in df.columns]
    if edge_cols:
        g = nx.DiGraph()
        for _, row in df.iterrows():
            src = str(row.get("author", "unknown"))
            for col in edge_cols:
                dst = row.get(col)
                if pd.notna(dst) and str(dst).strip() and str(dst).lower() != "nan" and src != str(dst):
                    if g.has_edge(src, str(dst)):
                        g[src][str(dst)]["weight"] += 1.0
                    else:
                        g.add_edge(src, str(dst), weight=1.0)
        if g.number_of_edges() > 0:
            scores = nx.pagerank(g, weight="weight")
            return normalize_scores(scores)
    counts = authors.value_counts().to_dict()
    scores = {a: math.log1p(c) for a, c in counts.items()}
    return normalize_scores(scores)


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Нормирует словарь числовых значений в диапазон от нуля до единицы
    
    Args:
        scores: словарь числовых оценок
    
    Returns:
        словарь нормированных значений
    """
    if not scores:
        return {"unknown": 0.0}
    vals = np.array(list(scores.values()), dtype=float)
    mn, mx = float(vals.min()), float(vals.max())
    if abs(mx - mn) < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: float((v - mn) / (mx - mn)) for k, v in scores.items()}


def _id_variants(value: Any) -> set[str]:
    """Формирует варианты идентификатора для сопоставления сообщений и авторов
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        множество вариантов идентификатора
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return set()
    variants = {raw}
    if raw.endswith(".0"):
        variants.add(raw[:-2])
    if raw.startswith(("t1_", "t3_")):
        variants.add(raw[3:])
    else:
        variants.add("t1_" + raw)
        variants.add("t3_" + raw)
    return {v for v in variants if v}


def _register_author_ids(id_to_author: dict[str, str], row: pd.Series, author: str) -> None:
    """Связывает идентификаторы строки с каноническим автором
    
    Args:
        id_to_author: словарь соответствия идентификаторов и авторов
        row: строка таблицы с сообщением, инструментом или событием
        author: канонический идентификатор автора
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    for col in ["message_id", "comment_ref_id", "post_ref_id"]:
        for key in _id_variants(row.get(col, "")):
            id_to_author[key] = author


def _edge_weight(row: pd.Series, score_weight: float) -> float:
    """Рассчитывает вес ребра взаимодействия по оценке сообщения
    
    Args:
        row: строка таблицы с сообщением, инструментом или событием
        score_weight: вес оценки сообщения при построении графа
    
    Returns:
        вес ребра взаимодействия
    """
    base = 1.0
    score = max(0.0, float(row.get("score_value", 0.0) or 0.0))
    ups = max(0.0, float(row.get("ups_value", 0.0) or 0.0))
    engagement = max(score, ups)
    if engagement > 0 and score_weight > 0:
        base += float(score_weight) * math.log1p(engagement)
    return float(base)


def _decay_graph_edges(g: nx.DiGraph, decay: float, min_edge_weight: float) -> None:
    """Ослабляет веса старых рёбер социального графа
    
    Args:
        g: социальный граф участников
        decay: коэффициент затухания весов графа
        min_edge_weight: минимальный вес ребра после затухания
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    if g.number_of_edges() == 0:
        return
    for u, v, data in list(g.edges(data=True)):
        data["weight"] = float(data.get("weight", 1.0)) * decay
        if data["weight"] < min_edge_weight:
            g.remove_edge(u, v)
    isolates = [n for n in list(g.nodes()) if g.degree(n) == 0]
    if isolates:
        g.remove_nodes_from(isolates)


def _pagerank_nstart(g: nx.DiGraph, previous: dict[str, float]) -> dict[str, float] | None:
    """Формирует стартовые значения PageRank из предыдущего шага
    
    Args:
        g: социальный граф участников
        previous: значения предыдущего шага для сглаживания или инициализации
    
    Returns:
        словарь стартовых значений PageRank
    """
    if not previous or g.number_of_nodes() == 0:
        return None
    nstart = {node: max(0.0, float(previous.get(node, 0.0))) for node in g.nodes()}
    total = sum(nstart.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in nstart.items()}


def _window_activity_scores(part: pd.DataFrame) -> dict[str, float]:
    """Рассчитывает активность авторов внутри шага дискретизации
    
    Args:
        part: фрагмент таблицы сообщений внутри шага дискретизации
    
    Returns:
        словарь активности авторов в шаге дискретизации
    """
    if part.empty:
        return {}
    counts = part["author"].fillna("unknown").astype(str).value_counts().to_dict()
    score_sum = part.groupby("author")["score_value"].sum().to_dict() if "score_value" in part.columns else {}
    raw = {}
    for author, count in counts.items():
        author = str(author)
        if is_service_author(author):
            raw[author] = 0.0
        else:
            raw[author] = math.log1p(float(count)) + 0.15 * math.log1p(max(0.0, float(score_sum.get(author, 0.0))))
    return normalize_scores(raw)


def assign_dynamic_window_influence(
    df: pd.DataFrame,
    cfg: AppConfig,
    window_hours: int,
    progress=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Рассчитывает динамическую авторитетность авторов по шагам дискретизации
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
        cfg: конфигурация проекта или приложения
        window_hours: длительность шага дискретизации в часах
        progress: callback для передачи статуса выполнения
    
    Returns:
        таблица сообщений с авторитетностью и словарь итоговых оценок авторов
    """
    data = df.sort_values("timestamp").copy()
    data["window_start"] = data["timestamp"].dt.floor(f"{int(window_hours)}h")
    data["influence"] = 0.0
    data["influence_graph_score"] = 0.0
    data["influence_activity_score"] = 0.0

    decay = float(cfg.get("influence", "decay_per_window", default=0.95))
    decay = min(1.0, max(0.0, decay))
    pagerank_alpha = float(cfg.get("influence", "pagerank_alpha", default=0.85))
    pagerank_alpha = min(0.99, max(0.01, pagerank_alpha))
    pagerank_weight = float(cfg.get("influence", "pagerank_weight", default=0.70))
    pagerank_weight = min(1.0, max(0.0, pagerank_weight))
    min_edge_weight = float(cfg.get("influence", "min_edge_weight", default=0.0001))
    score_weight = float(cfg.get("influence", "edge_score_weight", default=0.10))

    g = nx.DiGraph()
    previous_pr: dict[str, float] = {}
    id_to_author: dict[str, str] = {}
    window_summaries: list[dict[str, Any]] = []

    grouped = list(data.groupby("window_start", sort=True))
    for window_idx, (ts, part) in enumerate(grouped):
        _emit_progress(progress, f"Авторитетность: шаг дискретизации {window_idx + 1}/{len(grouped)}", window_idx + 1, len(grouped), stage="influence")
        _decay_graph_edges(g, decay=decay, min_edge_weight=min_edge_weight)
        part = part.sort_values("timestamp")

        # добавляем взаимодействия в хронологическом порядке
        # ответ создаёт ребро от текущего автора к автору родительского сообщения или корня
        for idx, row in part.iterrows():
            src = str(row.get("author", "unknown"))
            target_author = None
            parent_candidates = []
            parent_candidates.extend(_id_variants(row.get("parent_ref_id", "")))
            # во многих выгрузках комментарии содержат корневой post id
            # даже если Parent id отсутствует или записан с другим префиксом
            parent_candidates.extend(_id_variants(row.get("post_ref_id", "")))
            for key in parent_candidates:
                if key in id_to_author:
                    target_author = id_to_author[key]
                    break
            if target_author and target_author != src:
                weight = _edge_weight(row, score_weight)
                old = float(g[src][target_author]["weight"]) if g.has_edge(src, target_author) else 0.0
                g.add_edge(src, target_author, weight=old + weight)
            _register_author_ids(id_to_author, row, src)

        # добавляем активных авторов в граф даже при отсутствии рёбер
        for author in part["author"].fillna("unknown").astype(str).unique():
            g.add_node(author)

        if g.number_of_nodes() and g.number_of_edges():
            try:
                pr_raw = nx.pagerank(
                    g,
                    alpha=pagerank_alpha,
                    weight="weight",
                    nstart=_pagerank_nstart(g, previous_pr),
                    max_iter=int(cfg.get("influence", "pagerank_max_iter", default=100)),
                    tol=float(cfg.get("influence", "pagerank_tol", default=1e-6)),
                )
            except Exception:
                pr_raw = previous_pr or {node: 1.0 for node in g.nodes()}
        else:
            pr_raw = {author: 1.0 for author in part["author"].fillna("unknown").astype(str).unique()}

        previous_pr = dict(pr_raw)
        # служебные, неизвестные и удалённые узлы остаются в графе
        # но исключаются из нормировки авторитетности
        # иначе старый корпус с большим числом удалённых аккаунтов занизит PageRank реальных авторов
        pr_scores = normalize_scores({str(k): float(v) for k, v in pr_raw.items() if not is_service_author(k)})
        activity_scores = _window_activity_scores(part)

        authors = part["author"].fillna("unknown").astype(str)
        influence_values = []
        graph_values = []
        activity_values = []
        for author in authors:
            graph_score = float(pr_scores.get(author, 0.0))
            activity_score = float(activity_scores.get(author, 0.0))
            if is_service_author(author):
                # удалённые, неизвестные и служебные авторы сохраняются как узлы и рёбра
                # но их реальная авторитетность не идентифицируема
                graph_score = 0.0
                activity_score = 0.0
                final_score = 0.0
            else:
                # если у автора нет содержательных рёбер в графе
                # активность сохраняет оценку пригодной вместо полного обнуления
                final_score = pagerank_weight * graph_score + (1.0 - pagerank_weight) * activity_score
            graph_values.append(graph_score)
            activity_values.append(activity_score)
            influence_values.append(final_score)

        data.loc[part.index, "influence_graph_score"] = graph_values
        data.loc[part.index, "influence_activity_score"] = activity_values
        data.loc[part.index, "influence"] = influence_values

        top = pd.DataFrame({"author": authors, "influence": influence_values})
        top = top[~top["author"].map(is_service_author)].copy()
        top = top.groupby("author", as_index=False)["influence"].mean().sort_values("influence", ascending=False).head(10)
        window_summaries.append({
            "time": pd.Timestamp(ts).isoformat(),
            "n_authors": int(authors.nunique()),
            "graph_nodes": int(g.number_of_nodes()),
            "graph_edges": int(g.number_of_edges()),
            "top_authors_by_influence": dict(zip(top["author"], top["influence"].astype(float))),
        })

    metadata = {
        "mode": "dynamic_decayed_pagerank",
        "decay_per_window": decay,
        "pagerank_alpha": pagerank_alpha,
        "pagerank_weight": pagerank_weight,
        "min_edge_weight": min_edge_weight,
        "edge_score_weight": score_weight,
        "n_windows": len(window_summaries),
        "window_summaries": window_summaries,
    }
    return data.drop(columns=["window_start"], errors="ignore"), metadata


def sentiment_component(s: float, cfg: AppConfig) -> float:
    """Преобразует тональность в компонент функции выигрыша
    
    Args:
        s: числовая оценка тональности
        cfg: конфигурация проекта или приложения
    
    Returns:
        числовой компонент тональности для функции выигрыша
    """
    mode = cfg.get("payoff", "sentiment_mode", default="magnitude")
    s = float(np.clip(s, -1, 1))
    if mode == "signed_shift":
        return (s + 1.0) / 2.0
    if mode == "hybrid":
        lam = float(cfg.get("payoff", "hybrid_lambda", default=0.7))
        return lam * abs(s) + (1 - lam) * ((s + 1.0) / 2.0)
    return abs(s)


def sentiment_component_with_mode(s: float, cfg: AppConfig, mode: str | None = None) -> float:
    """Преобразует тональность по выбранному режиму интерпретации
    
    Args:
        s: числовая оценка тональности
        cfg: конфигурация проекта или приложения
        mode: режим преобразования или отображения
    
    Returns:
        числовой компонент тональности в выбранном режиме
    """
    old_mode = cfg.raw.setdefault("payoff", {}).get("sentiment_mode")
    if mode is not None:
        cfg.raw.setdefault("payoff", {})["sentiment_mode"] = mode
    try:
        return sentiment_component(s, cfg)
    finally:
        if mode is not None:
            if old_mode is None:
                cfg.raw.setdefault("payoff", {}).pop("sentiment_mode", None)
            else:
                cfg.raw.setdefault("payoff", {})["sentiment_mode"] = old_mode


def compute_payoff_from_messages(
    messages: pd.DataFrame,
    cfg: AppConfig,
    w_sentiment: float,
    w_influence: float,
    sentiment_mode: str | None = None,
) -> pd.DataFrame:
    """Рассчитывает средние выигрыши тем по сообщениям
    
    Args:
        messages: таблица или список сообщений для обработки
        cfg: конфигурация проекта или приложения
        w_sentiment: вес тональности в функции выигрыша
        w_influence: вес авторитетности в функции выигрыша
        sentiment_mode: режим преобразования тональности
    
    Returns:
        таблица выигрышей тем
    """
    df = messages.copy()
    total_w = float(w_sentiment) + float(w_influence)
    if total_w <= 0:
        w_sentiment, w_influence = 0.5, 0.5
    else:
        w_sentiment, w_influence = float(w_sentiment) / total_w, float(w_influence) / total_w
    df["_sent_component"] = df["sentiment"].map(lambda x: sentiment_component_with_mode(float(x), cfg, sentiment_mode))
    df["_influence_component"] = df.get("influence", 0.0).fillna(0.0).astype(float)
    df["payoff_message"] = w_sentiment * df["_sent_component"] + w_influence * df["_influence_component"]
    payoff = df.groupby("topic", as_index=False)["payoff_message"].mean().rename(columns={"payoff_message": "payoff"})
    return payoff.sort_values("topic").reset_index(drop=True)


def recompute_payoff_file(
    cfg: AppConfig,
    messages_path: str | Path,
    output_path: str | Path,
    w_sentiment: float,
    w_influence: float,
    sentiment_mode: str | None = None,
) -> pd.DataFrame:
    """Пересчитывает payoff-файл для выбранных весов функции выигрыша
    
    Args:
        cfg: конфигурация проекта или приложения
        messages_path: путь к таблице обработанных сообщений
        output_path: путь для сохранения результата
        w_sentiment: вес тональности в функции выигрыша
        w_influence: вес авторитетности в функции выигрыша
        sentiment_mode: режим преобразования тональности
    
    Returns:
        путь к пересчитанному payoff-файлу
    """
    messages = pd.read_csv(messages_path)
    payoff = compute_payoff_from_messages(messages, cfg, w_sentiment, w_influence, sentiment_mode)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payoff.to_csv(output_path, index=False)
    return payoff
