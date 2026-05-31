"""Построение социального графа для выбранного шага дискретизации"""
from __future__ import annotations


from ..pipeline.common import *
from ..pipeline.configuration import AppConfig, normalize_columns, is_service_author, service_author_label
from ..discussion.influence_payoff import normalize_scores
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..discussion.influence_payoff import (
        _decay_graph_edges,
        _edge_weight,
        _id_variants,
        _pagerank_nstart,
        _register_author_ids,
        _window_activity_scores,
    )

def load_payoff_from_df(payoff_df: pd.DataFrame, strategy_topics: list[int]) -> np.ndarray:
    """Рассчитывает payoff тем напрямую из таблицы сообщений
    
    Args:
        payoff_df: таблица выигрышей тем
        strategy_topics: список тем, используемых как стратегии агентной модели
    
    Returns:
        таблица выигрышей тем из сообщений
    """
    mapping = {int(r["topic"]): float(r["payoff"]) for _, r in payoff_df.iterrows() if "topic" in r and "payoff" in r}
    vals = [mapping.get(int(t), 0.5) for t in strategy_topics]
    return np.asarray(vals, dtype=float)


def build_user_graph_snapshot_from_messages(
    messages: pd.DataFrame | str | Path,
    start_time: Any,
    window_hours: int,
    context_windows: int = 0,
    cfg: AppConfig | None = None,
    max_nodes: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Строит снимок социального графа для выбранного набора сообщений
    
    Args:
        messages: таблица или список сообщений для обработки
        start_time: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
        context_windows: число предыдущих шагов дискретизации для исторического контекста
        cfg: конфигурация проекта или приложения
        max_nodes: максимальное число узлов графа для визуализации
    
    Returns:
        таблицы узлов и рёбер графа участников
    """
    if isinstance(messages, (str, Path)):
        df = pd.read_csv(messages)
    else:
        df = pd.DataFrame(messages).copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = normalize_columns(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
    start_ts = pd.Timestamp(pd.to_datetime(start_time, errors="coerce")).tz_localize(None)
    ctx_start = start_ts - pd.Timedelta(hours=int(window_hours) * int(max(0, context_windows)))
    end_ts = start_ts + pd.Timedelta(hours=int(window_hours))
    data = df[(df["timestamp"] >= ctx_start) & (df["timestamp"] < end_ts)].sort_values("timestamp").copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    decay = float(cfg.get("influence", "decay_per_window", default=0.95) if cfg else 0.95)
    decay = min(1.0, max(0.0, decay))
    pagerank_alpha = float(cfg.get("influence", "pagerank_alpha", default=0.85) if cfg else 0.85)
    min_edge_weight = float(cfg.get("influence", "min_edge_weight", default=0.0001) if cfg else 0.0001)
    score_weight = float(cfg.get("influence", "edge_score_weight", default=0.10) if cfg else 0.10)

    data["window_start"] = data["timestamp"].dt.floor(f"{int(window_hours)}h")
    g = nx.DiGraph()
    id_to_author: dict[str, str] = {}
    previous_pr: dict[str, float] = {}
    for _, part in data.groupby("window_start", sort=True):
        _decay_graph_edges(g, decay=decay, min_edge_weight=min_edge_weight)
        part = part.sort_values("timestamp")
        for _, row in part.iterrows():
            src = str(row.get("author", "unknown"))
            parent_candidates = []
            parent_candidates.extend(_id_variants(row.get("parent_ref_id", "")))
            parent_candidates.extend(_id_variants(row.get("post_ref_id", "")))
            target_author = None
            for key in parent_candidates:
                if key in id_to_author:
                    target_author = id_to_author[key]
                    break
            if target_author and target_author != src:
                weight = _edge_weight(row, score_weight)
                old = float(g[src][target_author]["weight"]) if g.has_edge(src, target_author) else 0.0
                g.add_edge(src, target_author, weight=old + weight)
            _register_author_ids(id_to_author, row, src)
        for author in part["author"].fillna("unknown").astype(str).unique():
            g.add_node(author)
        if g.number_of_nodes() and g.number_of_edges():
            try:
                previous_pr = nx.pagerank(g, alpha=pagerank_alpha, weight="weight", nstart=_pagerank_nstart(g, previous_pr))
            except Exception:
                previous_pr = previous_pr or {node: 1.0 for node in g.nodes()}

    # ограничиваем визуализацию наиболее релевантными узлами, чтобы интерфейс оставался читаемым
    current = data[(data["timestamp"] >= start_ts) & (data["timestamp"] < end_ts)].copy()
    current_authors = set(current["author"].fillna("unknown").astype(str).unique())
    pr_scores = normalize_scores({str(k): float(v) for k, v in previous_pr.items() if not is_service_author(k)}) if previous_pr else {}
    activity_scores = _window_activity_scores(current) if not current.empty else {}
    node_rows = []
    for node in g.nodes():
        node = str(node)
        service = is_service_author(node)
        graph_score = 0.0 if service else float(pr_scores.get(node, 0.0))
        activity_score = 0.0 if service else float(activity_scores.get(node, 0.0))
        influence = 0.0 if service else 0.7 * graph_score + 0.3 * activity_score
        node_rows.append({
            "author": node,
            "label": service_author_label(node),
            "is_service_author": bool(service),
            "active_in_window": bool(node in current_authors),
            "influence": float(influence),
            "graph_score": float(graph_score),
            "activity_score": float(activity_score),
            "in_degree": int(g.in_degree(node)),
            "out_degree": int(g.out_degree(node)),
            "weighted_in_degree": float(g.in_degree(node, weight="weight")),
            "weighted_out_degree": float(g.out_degree(node, weight="weight")),
        })
    nodes = pd.DataFrame(node_rows)
    if not nodes.empty:
        nodes = nodes.sort_values(["active_in_window", "influence", "weighted_in_degree"], ascending=[False, False, False]).head(int(max_nodes)).reset_index(drop=True)
        keep = set(nodes["author"])
    else:
        keep = set()
    edge_rows = []
    for u, v, data_e in g.edges(data=True):
        if u in keep and v in keep:
            edge_rows.append({
                "source": str(u),
                "source_label": service_author_label(u),
                "target": str(v),
                "target_label": service_author_label(v),
                "weight": float(data_e.get("weight", 1.0)),
                "source_is_service": bool(is_service_author(u)),
                "target_is_service": bool(is_service_author(v)),
            })
    edges = pd.DataFrame(edge_rows).sort_values("weight", ascending=False).reset_index(drop=True) if edge_rows else pd.DataFrame(columns=["source", "target", "weight"])
    return nodes, edges
