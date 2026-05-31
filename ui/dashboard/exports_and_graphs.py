"""Экспорт экспериментов, социальный граф и графики качества моделирования"""
from __future__ import annotations


from .common import *

from .controls import fmt, metric_card

def build_experiment_package(
    fwd_current: pd.DataFrame,
    paths: dict[str, Path],
    price_panel: Any | None = None,
) -> bytes:
    """Формирует ZIP-пакет с результатами текущего эксперимента
    
    Args:
        fwd_current: таблица дискуссионных индикаторов текущего эксперимента
        paths: словарь путей к результатам, данным и временным файлам
        price_panel: таблица цен для рыночной панели
    
    Returns:
        bytes: байтовое содержимое ZIP-архива с результатами эксперимента
    """
    buffer = io.BytesIO()
    sidecar_base = Path(paths["forward_indicators"])
    files = [
        (sidecar_base, "event_indicators.csv"),
        (sidecar_base.with_name(sidecar_base.stem + "_model_forecast.csv"), "model_forecast_no_future.csv"),
        (sidecar_base.with_name(sidecar_base.stem + "_context_messages.csv"), "processed_context_messages.csv"),
        (sidecar_base.with_name(sidecar_base.stem + "_ticker_summary.csv"), "ticker_context_summary.csv"),
        (sidecar_base.with_name(sidecar_base.stem + "_topic_info.csv"), "local_topic_info.csv"),
        (sidecar_base.with_name(sidecar_base.stem + "_context_publications.csv"), "context_publications.csv"),
        (sidecar_base.with_name(sidecar_base.stem + "_publications.csv"), "window_publications.csv"),
    ]
    manifest = {
        "package_type": "discussion_market_experiment",
        "description": "Экспорт результатов одного запуска",
        "no_future_leakage_note": "model_forecast_no_future.csv не содержит фактические будущие данные, если такие данные не были доступны в момент расчёта",
        "files": [],
    }
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if fwd_current is not None and not pd.DataFrame(fwd_current).empty:
            data = pd.DataFrame(fwd_current).to_csv(index=False).encode("utf-8-sig")
            zf.writestr("event_indicators_from_session.csv", data)
            manifest["files"].append("event_indicators_from_session.csv")
        for path, name in files:
            try:
                path = Path(path)
                if path.exists() and path.is_file():
                    zf.write(path, arcname=name)
                    manifest["files"].append(name)
            except Exception:
                continue
        if price_panel is not None and not pd.DataFrame(price_panel).empty:
            zf.writestr("market_price_panel.csv", pd.DataFrame(price_panel).to_csv(index=False).encode("utf-8-sig"))
            manifest["files"].append("market_price_panel.csv")
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    return buffer.getvalue()


def render_user_graph(nodes: pd.DataFrame, edges: pd.DataFrame | None) -> None:
    """Показывает социальный граф в выбранном шаге
    
    Args:
        nodes: таблица узлов социального графа
        edges: таблица рёбер социального графа
    
    Returns:
        None
    """
    if nodes is None or pd.DataFrame(nodes).empty:
        st.info("Для выбранного шага не удалось построить социальный граф.")
        return
    nodes = pd.DataFrame(nodes).copy()
    edges = pd.DataFrame(edges).copy() if edges is not None else pd.DataFrame()
    if edges.empty:
        st.info("В выбранном диапазоне почти нет восстановимых связей ответов/родительских сообщений. Показаны активные авторы без рёбер.")
    g = nx.DiGraph()
    for _, row in nodes.iterrows():
        g.add_node(str(row.get("author")), **row.to_dict())
    for _, row in edges.iterrows():
        src, dst = str(row.get("source")), str(row.get("target"))
        if src in g.nodes and dst in g.nodes:
            g.add_edge(src, dst, weight=float(row.get("weight", 1.0) or 1.0))
    if g.number_of_nodes() == 0:
        st.info("Граф пуст.")
        return
    try:
        pos = nx.spring_layout(g, weight="weight", seed=42, k=0.65)
    except Exception:
        pos = nx.circular_layout(g)
    edge_x, edge_y = [], []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="rgba(148, 163, 184, 0.38)"),
        hoverinfo="none", name="связи"
    ))
    node_x, node_y, text, size, color = [], [], [], [], []
    for node in g.nodes():
        meta = g.nodes[node]
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
        influence = float(meta.get("influence", 0.0) or 0.0)
        is_service = bool(meta.get("is_service_author", False))
        label = str(meta.get("label", node))
        text.append(
            f"{label}<br>графовая авторитетность: {influence:.3f}<br>in-degree: {meta.get('in_degree', 0)}<br>out-degree: {meta.get('out_degree', 0)}"
        )
        size.append(10 + 28 * max(0.0, min(1.0, influence)))
        color.append("rgba(148, 163, 184, 0.78)" if is_service else "#6366F1")
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers", marker=dict(size=size, color=color, line=dict(width=1, color="rgba(148, 163, 184, 0.65)")),
        text=text, hoverinfo="text", name="авторы"
    ))
    fig.update_layout(
        height=560, showlegend=False, margin=dict(l=10, r=10, t=25, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    plotly_chart(fig)
    st.caption("Серые узлы - служебные/неопределённые авторы. Они остаются в графе, но их авторитетность принудительно равна 0, потому что реальный пользователь **не идентифицируется**.")

def compact_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Готовит компактную таблицу качества моделирования тематической динамики
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        pd.DataFrame: таблица качества моделирования с русскими названиями колонок
    """
    if df.empty:
        return df
    cols = [
        "start_index", "future_index", "horizon", "start_time", "future_time",
        "nrmse", "rmse", "l1", "cosine", "coverage", "emerging_topic_share", "event_type",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    rename = {
        "horizon": "шагов вперёд",
        "nrmse": "nRMSE, %",
        "rmse": "RMSE",
        "l1": "L1",
        "cosine": "cosine",
        "coverage": "coverage",
        "emerging_topic_share": "emerging share",
        "event_type": "тип события",
    }
    return out.rename(columns=rename)



def render_signal_feed(events: pd.DataFrame) -> None:
    """Показывает поток найденных дискуссионных событий
    
    Args:
        events: таблица дискуссионных событий
    
    Returns:
        None
    """
    if events is None or pd.DataFrame(events).empty:
        st.info("Пока нет аналитических событий. Обновите мониторинг или выберите более длинный период данных.")
        return
    df = pd.DataFrame(events).copy()
    for col in ["nrmse", "l1", "rmse", "coverage", "emerging_topic_share"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "nrmse" in df:
        df = df.sort_values("nrmse", ascending=False)
    st.markdown("### Лента аналитических событий")
    st.caption("Это рабочая лента для режима реального времени: сервис показывает шаги, где динамика дискуссии заметно отклонилась. Рыночная проверка может быть недоступна, если будущие данные ещё не появились.")
    for _, row in df.head(12).iterrows():
        nrmse = row.get("nrmse")
        l1 = row.get("l1")
        start_time = row.get("start_time", "-")
        future_time = row.get("future_time", "-")
        horizon = row.get("horizon", row.get("future_step", 1))
        coverage = row.get("coverage")
        emerging = row.get("emerging_topic_share")
        if pd.notna(nrmse) and float(nrmse) >= 35.0:
            level = "Сильный индикатор"
            tone = "red"
        elif pd.notna(nrmse) and float(nrmse) >= 15.0:
            level = "Умеренный индикатор"
            tone = "orange"
        else:
            level = "Наблюдать"
            tone = "blue"
        reasons = []
        if pd.notna(nrmse):
            reasons.append(f"nRMSE {float(nrmse):.2f}%")
        if pd.notna(l1):
            reasons.append(f"L1 {float(l1):.3f}")
        if pd.notna(coverage):
            reasons.append(f"coverage {float(coverage):.2f}")
        if pd.notna(emerging):
            reasons.append(f"новые темы {float(emerging):.2f}")
        reason_text = " · ".join(reasons) if reasons else "метрики требуют расчёта"
        html = f"""
        <div class="post-card" data-tone="{tone}">
          <div class="report-title">{level}</div>
          <div class="report-body">
            <b>Шаг:</b> {start_time}<br>
            <b>Сравнение траектории:</b> +{horizon} шаг, контрольное значение {future_time}<br>
            <b>Почему подсвечено:</b> {reason_text}<br>
            <b>Что делать:</b> открыть отчёт, посмотреть темы, тикеры, публикации и фактическое движение цены на горизонте сравнения.
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

def decision_view(decision: dict[str, Any] | None) -> None:
    """Показывает итоговую исследовательский отчет по выбранному событию
    
    Args:
        decision: словарь с итоговой интерпретацией события
    
    Returns:
        None
    """
    if not decision:
        st.info("Лента аналитических событий ещё не рассчитана. Обновите мониторинг индикаторов.")
        return
    action = decision.get("action", "unknown")
    labels = {
        "model_is_stable": ("Модель стабильна", "Сильных продолжительных отклонений нет.", "green"),
        "inspect_local_anomalies": ("Есть локальные отклонения", "Их стоит сопоставить с рыночными рядами.", "orange"),
        "refresh_topic_model": ("Обновить topic-space", "Высокая доля отклонений.", "red"),
        "recalibrate_or_rebuild_model": ("Нужен пересчёт", "Отклонения продолжаются несколько шагов подряд.", "red"),
        "not_enough_data": ("Недостаточно данных", "Нужно больше шагов для диагностики.", "gray"),
    }
    title, desc, tone = labels.get(action, ("Требуется проверка", f"Действие: {action}", "orange"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Решение", title, desc, tone)
    with c2:
        metric_card("Событий", str(decision.get("n_events", 0)), "Шаги с сильным расхождением", "blue")
    with c3:
        metric_card("Локальных отклонений", str(decision.get("n_local_deviation_events", 0)), "Без явной проблемы качества", "green")
    with c4:
        metric_card("nRMSE, %", fmt(decision.get("mean_nrmse"), 2) + "%", "по всем шагам и темам", "violet")
    st.caption(f"Продолжительное отклонение: {'да' if decision.get('prolonged_deviation') else 'нет'}")


def show_quality_cards(comp: pd.DataFrame, window_hours: int) -> None:
    """Показывает карточки качества моделирования
    
    Args:
        comp: таблица сравнения фактической и смоделированной динамики
        window_hours: длительность шага дискретизации в часах
    
    Returns:
        None
    """
    ordered = comp.sort_values("horizon")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("nRMSE, %", (fmt(ordered["matrix_nrmse"].iloc[0], 2) + "%") if "matrix_nrmse" in ordered and pd.notna(ordered["matrix_nrmse"].iloc[0]) else fmt(ordered["nrmse"].mean(), 2) + "%", "по всем шагам и темам", "green")
    with c2:
        metric_card("RMSE", fmt(ordered["matrix_rmse"].iloc[0], 4) if "matrix_rmse" in ordered and pd.notna(ordered["matrix_rmse"].iloc[0]) else fmt(ordered["rmse"].mean(), 4), "по всем шагам и темам", "blue")
    with c3:
        metric_card("Средний L1", fmt(ordered["l1"].mean()), "средний сдвиг распределения", "orange")
    with c4:
        max_h = int(ordered["horizon"].max())
        metric_card("Горизонт", f"{max_h * window_hours} ч", f"+1 ... +{max_h} шаг", "gray")
    st.caption("Метрики считаются по всей выбранной модельной симуляции, а не только по одному лучшему шагу.")


def plot_forecast_errors(comp: pd.DataFrame) -> None:
    """Строит график ошибок моделирования по шагам дискретизации
    
    Args:
        comp: таблица сравнения фактической и смоделированной динамики
    
    Returns:
        None
    """
    plot = comp.copy().sort_values("horizon")
    plot["future_step"] = plot["horizon"].astype(int)
    fig_n = px.line(
        plot,
        x="future_step",
        y="nrmse",
        markers=True,
        title="nRMSE по формуле: RMSE / размах фактического значения распределения тем × 100%",
        labels={"future_step": "следующий шаг (+N)", "nrmse": "nRMSE, %"},
    )
    fig_n.update_traces(mode="lines+markers")
    fig_n.update_xaxes(dtick=1)
    fig_n.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    plotly_chart(fig_n)

    fig_abs = px.line(
        plot,
        x="future_step",
        y=[c for c in ["rmse", "l1"] if c in plot.columns],
        markers=True,
        title="Абсолютные ошибки тематических распределений",
        labels={"future_step": "следующий шаг (+N)", "value": "значение", "variable": "метрика"},
    )
    fig_abs.update_traces(mode="lines+markers")
    fig_abs.update_xaxes(dtick=1)
    fig_abs.update_layout(legend_title_text="Метрика", margin=dict(l=20, r=20, t=55, b=20))
    plotly_chart(fig_abs)




def plot_diagnostic_errors(forecasts: pd.DataFrame) -> None:
    """Строит диагностический график ошибок по истории моделирования
    
    Args:
        forecasts: таблица результатов проверки моделирования
    
    Returns:
        None
    """
    if forecasts.empty:
        return
    plot_df = forecasts.copy()
    plot_df["horizon"] = plot_df["horizon"].astype(int)
    plot_df["start_index"] = plot_df["start_index"].astype(int)
    plot_df = plot_df.sort_values(["start_index", "horizon"])
    n_starts = int(plot_df["start_index"].nunique())
    max_horizon = int(plot_df["horizon"].max())

    if max_horizon > 1:
        if n_starts <= 12:
            plot_df["Стартовый шаг"] = plot_df["start_index"].map(lambda x: f"W{x}")
            fig = px.line(
                plot_df,
                x="horizon",
                y="nrmse",
                color="Стартовый шаг",
                markers=True,
                title="nRMSE: продолжение сравнения от стартового шага к будущим шагам",
                labels={"horizon": "следующий шаг (+N)", "nrmse": "nRMSE, %"},
                            )
        else:
            agg = plot_df.groupby("horizon", as_index=False).agg(
                mean_nrmse=("nrmse", "mean"),
                median_nrmse=("nrmse", "median"),
            )
            fig = px.line(
                agg,
                x="horizon",
                y=["mean_nrmse", "median_nrmse"],
                markers=True,
                title="nRMSE по будущим шагам дискретизации: среднее и медиана по стартовым шагам",
                labels={"horizon": "следующий шаг (+N)", "value": "nRMSE, %", "variable": "метрика"},
                            )
            fig.update_layout(legend_title_text="Агрегат")
        fig.update_traces(mode="lines+markers")
        fig.update_xaxes(dtick=1)
        fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
        plotly_chart(fig)
        return

    fig = px.line(
        plot_df,
        x="start_index",
        y="nrmse",
        markers=True,
        title="nRMSE по стартовым шагам дискретизации: проверка ближайшего будущего шага (+1)",
        labels={"start_index": "стартовый шаг", "nrmse": "nRMSE, %"}
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    plotly_chart(fig)
