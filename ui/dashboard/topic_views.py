"""Восстановление тематических названий и графики тематического распределения"""
from __future__ import annotations


from .common import *

def plot_topic_trajectories(comp: pd.DataFrame, max_topics: int = 5, topic_names: dict[int, str] | None = None) -> None:
    """Строит траектории фактических и смоделированных долей тем
    
    Args:
        comp: таблица сравнения фактической и смоделированной динамики
        max_topics: максимальное число тем для отображения
        topic_names: словарь отображаемых названий тем
    
    Returns:
        None
    """
    if comp.empty:
        return
    real_matrix = np.array(comp["real_props"].tolist(), dtype=float)
    pred_matrix = np.array(comp["predicted_props"].tolist(), dtype=float)
    if real_matrix.size == 0:
        return
    n_topics = real_matrix.shape[1]
    options = [topic_display_name(i, topic_names) for i in range(n_topics)]
    selected_label = st.selectbox(
        "Тема для детального сравнения траекторий",
        options,
        index=0,
        key="model_topic_bar_select",
        help="Можно выбрать любую тему, а не только крупнейшие. Для каждого будущего шага дискретизации показываются две траектории: фактическая доля темы и пунктирная модельная симуляция.",
    )
    topic_idx = options.index(selected_label)
    rows = []
    for _, row in comp.sort_values("horizon").iterrows():
        h = int(row["horizon"])
        rows.append({"future_step": h, "ряд": "факт", "доля": float(row["real_props"][topic_idx])})
        rows.append({"future_step": h, "ряд": "симуляция", "доля": float(row["predicted_props"][topic_idx])})
    plot = pd.DataFrame(rows)
    fig = go.Figure()
    for row_name, color, dash in [("факт", "#2563eb", "solid"), ("симуляция", "#14B89D", "dash")]:
        part = plot[plot["ряд"] == row_name].sort_values("future_step")
        fig.add_trace(
            go.Scatter(
                x=part["future_step"],
                y=part["доля"],
                mode="lines+markers",
                name=row_name,
                line=dict(color=color, width=2.6, dash=dash),
                marker=dict(size=6),
                hovertemplate="шаг +%{x}<br>доля темы: %{y:.4f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_xaxes(title_text="следующий шаг (+N)", dtick=1)
    fig.update_yaxes(title_text="доля темы")
    fig.update_layout(
        title=f"Факт и симуляция по теме: {selected_label}",
        legend_title_text="ряд",
        margin=dict(l=45, r=20, t=55, b=35),
    )
    plotly_chart(fig)


_TOPIC_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "you", "your", "have", "has",
    "had", "not", "but", "all", "can", "will", "just", "about", "into", "over", "under", "than", "then",
    "they", "them", "their", "there", "what", "when", "where", "who", "why", "how", "would", "could", "should",
    "это", "как", "что", "для", "или", "при", "над", "под", "уже", "ещё", "если", "там", "так", "его", "её", "они",
    "она", "оно", "без", "про", "все", "всё", "быть", "был", "была", "были", "будет", "которые", "который",
}


def _topic_column(df: pd.DataFrame) -> str | None:
    """Находит колонку с номером темы в таблице сообщений
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        str | None: название колонки темы или None, если колонка не найдена
    """
    for col in ["topic", "Topic", "topic_id", "topic_label", "тема"]:
        if col in df.columns:
            return col
    return None


def _app_column_key(value: Any) -> str:
    """Нормализует название колонки для сопоставления в интерфейсе
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        str: нормализованный ключ колонки для сопоставления алиасов
    """
    return re.sub(r"[^a-z0-9]+", "", str(value).replace("\ufeff", "").strip().lower())


def _first_app_col(df: pd.DataFrame, names: list[str]) -> str | None:
    """Возвращает первую доступную колонку из набора допустимых названий
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
        names: набор допустимых имён или словарь названий
    
    Returns:
        str | None: первое найденное название колонки или None
    """
    lookup = {_app_column_key(c): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        key = _app_column_key(name)
        if key in lookup:
            return lookup[key]
    return None


def _text_column(df: pd.DataFrame) -> str | None:
    """Находит колонку с текстом сообщения
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        str | None: название колонки с текстом сообщения или None
    """
    return _first_app_col(df, ["text", "text_clean", "body", "selftext", "title", "публикация", "message", "content", "comment_text", "post_text"])


def _time_column(df: pd.DataFrame) -> str | None:
    """Находит колонку с временной меткой сообщения
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        str | None: название колонки с временной меткой или None
    """
    return _first_app_col(df, ["timestamp", "created_utc", "created_at", "createdAt", "created", "datetime", "time", "date", "published_at", "publishedAt"])


@st.cache_data(show_spinner=False, max_entries=6)
def _raw_interval_metadata_cached(path_text: str, mtime: float, size: int) -> pd.DataFrame:
    """Собирает метаданные исходного CSV с кэшированием по состоянию файла
    
    Args:
        path_text: строковое представление пути к файлу
        mtime: время последнего изменения файла
        size: размер элемента интерфейса
    
    Returns:
        pd.DataFrame: таблица индексов и временных меток исходного CSV-файла
    """
    raw_path = Path(path_text).expanduser()
    raw_df = pd.read_csv(raw_path)
    normalized = normalize_columns(raw_df.copy())
    if "timestamp" not in normalized.columns:
        raise ValueError("Не найдена нормализованная колонка timestamp")
    ts = pd.to_datetime(normalized["timestamp"], errors="coerce")
    meta = pd.DataFrame({
        "source_index": normalized.index.to_numpy(),
        "timestamp": ts.to_numpy(),
    })
    meta = meta.dropna(subset=["timestamp"]).reset_index(drop=True)
    return meta


def _raw_interval_metadata(raw_path: Path) -> pd.DataFrame:
    """Возвращает метаданные исходного CSV-файла
    
    Args:
        raw_path: путь к исходной таблице сообщений
    
    Returns:
        pd.DataFrame: таблица метаданных исходного CSV-файла
    """
    raw_path = Path(raw_path).expanduser()
    stat = raw_path.stat()
    return _raw_interval_metadata_cached(str(raw_path), float(stat.st_mtime), int(stat.st_size)).copy()


def _default_interval_bounds_from_meta(meta: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Определяет начальные границы анализа по метаданным исходного файла
    
    Args:
        meta: метаданные исходного файла или анализа
    
    Returns:
        tuple[pd.Timestamp, pd.Timestamp]: начальная и конечная границы анализа по метаданным
    """
    if meta.empty:
        now = pd.Timestamp.now().floor("h")
        return now, now + pd.Timedelta(hours=1)
    ts = pd.to_datetime(meta["timestamp"], errors="coerce").dropna()
    start = pd.Timestamp(ts.min()).floor("h")
    # правая граница не включается, поэтому берём следующий полный час после последней публикации
    # иначе записи последнего часа пропадут из подсчёта
    # и не попадут в подготовленный фрагмент
    end = pd.Timestamp(ts.max()).floor("h") + pd.Timedelta(hours=1)
    if end <= start:
        end = start + pd.Timedelta(hours=1)
    return start, end


def _count_interval_rows(raw_path: Path, interval_start: pd.Timestamp, interval_end: pd.Timestamp) -> tuple[int, int]:
    """Подсчитывает строки исходного файла внутри выбранного интервала
    
    Args:
        raw_path: путь к исходной таблице сообщений
        interval_start: начало выбранного интервала анализа
        interval_end: конец выбранного интервала анализа
    
    Returns:
        tuple[int, int]: число строк выбранного интервала и число строк с корректной временной меткой
    """
    meta = _raw_interval_metadata(raw_path)
    if interval_end <= interval_start:
        return 0, int(len(meta))
    ts = pd.to_datetime(meta["timestamp"], errors="coerce")
    mask = (ts >= pd.Timestamp(interval_start)) & (ts < pd.Timestamp(interval_end))
    return int(mask.fillna(False).sum()), int(len(meta))


def _materialize_interval_raw_csv(raw_path: Path, interval_start: pd.Timestamp, interval_end: pd.Timestamp, cfg_obj: AppConfig) -> tuple[Path, int, int]:
    """Сохраняет фрагмент исходных сообщений для выбранного интервала анализа
    
    Args:
        raw_path: путь к исходной таблице сообщений
        interval_start: начало выбранного интервала анализа
        interval_end: конец выбранного интервала анализа
        cfg_obj: объект конфигурации приложения
    
    Returns:
        tuple[Path, int, int]: путь к сохранённому фрагменту, число выбранных строк и число валидных строк
    """
    raw_path = Path(raw_path).expanduser()
    if interval_end <= interval_start:
        raise ValueError("Конец интервала должен быть позже начала интервала")
    raw_df = pd.read_csv(raw_path)
    normalized = normalize_columns(raw_df.copy())
    ts = pd.to_datetime(normalized["timestamp"], errors="coerce")
    mask = ((ts >= pd.Timestamp(interval_start)) & (ts < pd.Timestamp(interval_end))).fillna(False)
    selected_source_index = normalized.index[mask.to_numpy()]
    filtered_raw = raw_df.loc[selected_source_index].copy()
    total_valid = int(len(normalized))
    if filtered_raw.empty:
        raise ValueError(
            "В выбранный интервал не попала ни одна публикация, пригодная для моделирования. "
            "Подготовка всего файла намеренно не запускается, чтобы не исказить эксперимент."
        )
    filtered_dir = cfg_obj.data_dir / "raw" / "uploads"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    digest_src = f"{raw_path.resolve()}|{pd.Timestamp(interval_start).isoformat()}|{pd.Timestamp(interval_end).isoformat()}|{len(filtered_raw)}"
    digest = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()[:12]
    target = filtered_dir / f"selected_interval_for_modeling_{digest}.csv"
    filtered_raw.to_csv(target, index=False, encoding="utf-8")
    return target, int(len(filtered_raw)), total_valid


def _topic_id(value: Any) -> int | None:
    """Преобразует значение темы в целочисленный идентификатор
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        int | None: целочисленный идентификатор темы или None при невозможности преобразования
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def _format_topic_words(value: Any) -> str:
    """Формирует читаемое название темы по ключевым словам
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        str: читаемая строка ключевых слов темы
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (list, tuple, set)):
        words = [str(x).strip() for x in value if str(x).strip()]
    else:
        raw = str(value).strip()
        words: list[str] = []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple, set)):
                words = [str(x).strip(" '\"[]") for x in parsed if str(x).strip()]
        except Exception:
            pass
        if not words:
            # BERTopic часто хранит имя темы в виде идентификатор плюс ключевые слова
            raw = re.sub(r"^-?\d+[_\s-]+", "", raw)
            words = [w.strip(" '\"[]") for w in re.split(r"[_;,/|]+", raw) if w.strip(" '\"[]")]
    cleaned = []
    for word in words:
        word = re.sub(r"\s+", " ", word).strip().lower()
        if not word or word in _TOPIC_STOPWORDS or len(word) < 2:
            continue
        if word not in cleaned:
            cleaned.append(word)
    return " / ".join(cleaned[:4])


def topic_name_map(cfg_obj: AppConfig, messages: pd.DataFrame | None = None) -> dict[int, str]:
    """Строит словарь отображаемых названий тем
    
    Args:
        cfg_obj: объект конфигурации приложения
        messages: таблица или список сообщений для обработки
    
    Returns:
        dict[int, str]: словарь названий тем по их числовым идентификаторам
    """
    names: dict[int, str] = {}
    topic_info_path = cfg_obj.data_dir / "processed" / "topic_info.csv"
    if topic_info_path.exists():
        try:
            info = pd.read_csv(topic_info_path)
            topic_col = "Topic" if "Topic" in info.columns else "topic" if "topic" in info.columns else None
            if topic_col:
                for _, row in info.iterrows():
                    topic_id = _topic_id(row.get(topic_col))
                    if topic_id is None or topic_id < 0:
                        continue
                    label = ""
                    for col in ["CustomName", "Name", "Representation", "KeyBERT", "keywords", "Words"]:
                        if col in info.columns:
                            label = _format_topic_words(row.get(col))
                            if label:
                                break
                    if label:
                        names[topic_id] = label
        except Exception:
            pass
    if messages is not None and not pd.DataFrame(messages).empty:
        df = pd.DataFrame(messages)
        topic_col = _topic_column(df)
        text_col = _text_column(df)
        if topic_col and text_col:
            for topic_value, part in df.groupby(topic_col):
                topic_id = _topic_id(topic_value)
                if topic_id is None or topic_id < 0 or topic_id in names:
                    continue
                counter: Counter[str] = Counter()
                for text_value in part[text_col].dropna().astype(str).head(250):
                    tokens = re.findall(r"[A-Za-zА-Яа-я][A-Za-zА-Яа-я0-9_]{2,}", text_value.lower())
                    counter.update(t for t in tokens if t not in _TOPIC_STOPWORDS)
                words = [w for w, _ in counter.most_common(4)]
                if words:
                    names[topic_id] = " / ".join(words)
    return names


def topic_display_name(topic_id: int | None, names: dict[int, str] | None = None) -> str:
    """Возвращает подпись темы для графиков и таблиц
    
    Args:
        topic_id: идентификатор темы
        names: набор допустимых имён или словарь названий
    
    Returns:
        str: читаемая подпись темы для графиков и таблиц
    """
    if topic_id is None:
        return "Тема не определена"
    if topic_id == -1:
        return "Шумовая тема"
    title = (names or {}).get(int(topic_id), "")
    return f"Тема {int(topic_id)} - {title}" if title else f"Тема {int(topic_id)}"


def _messages_for_window(source_path: Path, start_time: Any, window_hours: int) -> pd.DataFrame:
    """Загружает сообщения, попадающие в заданный шаг дискретизации
    
    Args:
        source_path: путь к исходной или обработанной таблице сообщений
        start_time: временная метка начала анализируемого шага
        window_hours: длительность шага дискретизации в часах
    
    Returns:
        pd.DataFrame: таблица сообщений, попавших в заданный шаг
    """
    if not source_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(source_path)
    time_col = _time_column(df)
    if time_col and start_time is not None:
        t0 = pd.to_datetime(start_time, errors="coerce")
        if not pd.isna(t0):
            t1 = t0 + pd.Timedelta(hours=int(window_hours))
            ts = pd.to_datetime(df[time_col], errors="coerce")
            df = df[(ts >= t0) & (ts < t1)].copy()
    return df


def current_window_topic_distribution(
    cfg_obj: AppConfig,
    messages_path: Path,
    forward_indicators_path: Path,
    start_time: Any,
    window_hours: int,
) -> tuple[pd.DataFrame, str]:
    """Рассчитывает тематическое распределение текущего шага дискретизации
    
    Args:
        cfg_obj: объект конфигурации приложения
        messages_path: путь к таблице обработанных сообщений
        forward_indicators_path: путь к таблице дискуссионных индикаторов
        start_time: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
    
    Returns:
        tuple[pd.DataFrame, str]: таблица распределения тем и описание источника данных
    """
    base = Path(forward_indicators_path)
    context_path = base.with_name(base.stem + "_context_messages.csv")
    sources = [(context_path, "контекст текущего расчёта"), (Path(messages_path), "сообщения последней инициализации")]
    for source, label in sources:
        df = _messages_for_window(source, start_time, int(window_hours))
        if df.empty:
            continue
        topic_col = _topic_column(df)
        if not topic_col:
            continue
        df = df.copy()
        df["_topic_id"] = df[topic_col].map(_topic_id)
        df = df[df["_topic_id"].notna() & (df["_topic_id"] >= 0)]
        if df.empty:
            continue
        names = topic_name_map(cfg_obj, df)
        counts = df["_topic_id"].astype(int).value_counts().sort_values(ascending=False)
        total = float(counts.sum()) if counts.sum() else 1.0
        out = pd.DataFrame({
            "topic_id": counts.index.astype(int),
            "messages": counts.values.astype(int),
            "share": counts.values.astype(float) / total,
        })
        out["topic_name"] = out["topic_id"].map(lambda x: names.get(int(x), ""))
        out["label"] = out["topic_id"].map(lambda x: topic_display_name(int(x), names))
        return out, label
    return pd.DataFrame(), "нет данных о темах выбранного шага дискретизации"


def render_current_window_topic_distribution(cfg_obj: AppConfig, messages_path: Path, forward_indicators_path: Path, start_time: Any, window_hours: int) -> None:
    """Показывает распределение тем текущего шага дискретизации
    
    Args:
        cfg_obj: объект конфигурации приложения
        messages_path: путь к таблице обработанных сообщений
        forward_indicators_path: путь к таблице дискуссионных индикаторов
        start_time: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
    
    Returns:
        None
    """
    dist, source = current_window_topic_distribution(cfg_obj, messages_path, forward_indicators_path, start_time, int(window_hours))
    if dist.empty:
        st.info("Для выбранного шага дискретизации не удалось восстановить распределение тем")
        return
    section_header(
        "Распределение тем в шаге дискретизации",
        "Показана доля сообщений по тематическим кластерам выбранного шага дискретизации. Названия тем сформированы по метаданным BERTopic или по наиболее частотным ключевым словам шага",
        "распределение тем",
    )
    plot = dist.head(18).sort_values("share", ascending=True)
    fig = px.bar(
        plot,
        x="share",
        y="label",
        orientation="h",
        title="Тематическая структура выбранного шага дискретизации",
        labels={"share": "доля сообщений", "label": "тематический кластер"},
        hover_data={"messages": True, "topic_id": True, "share": ":.2%", "label": False},
    )
    fig.update_layout(showlegend=False, margin=dict(l=220, r=36, t=62, b=62))
    fig.update_xaxes(tickformat=".0%")
    plotly_chart(fig)
    table = dist.copy()
    table["Доля"] = table["share"].map(lambda x: f"{x:.1%}")
    table = table.rename(columns={"topic_id": "Номер темы", "topic_name": "Название темы", "messages": "Сообщений"})
    st.caption(f"Источник тематического распределения: {source}")
    st.dataframe(table[["Номер темы", "Название темы", "Доля", "Сообщений"]], width="stretch", hide_index=True)

def plot_topic_distribution(row: pd.Series, title: str, topic_names: dict[int, str] | None = None) -> None:
    """Строит столбчатый график тематического распределения
    
    Args:
        row: строка таблицы с сообщением, инструментом или событием
        title: заголовок блока интерфейса
        topic_names: словарь отображаемых названий тем
    
    Returns:
        None
    """
    real = np.asarray(row["real_props"], dtype=float)
    pred = np.asarray(row["predicted_props"], dtype=float)
    n = min(len(real), len(pred))
    real, pred = real[:n], pred[:n]
    score = real + pred
    top_idx = np.argsort(-score)[: min(18, n)]
    dplot = pd.DataFrame({
        "topic_index": [topic_display_name(int(i), topic_names) for i in top_idx],
        "Реальность": real[top_idx],
        "Модель": pred[top_idx],
    })
    long = dplot.melt(id_vars="topic_index", var_name="ряд", value_name="доля")
    fig = px.bar(
        long,
        x="topic_index",
        y="доля",
        color="ряд",
        barmode="group",
        title=title,
        labels={"topic_index": "топ тем по суммарной доле", "доля": "доля темы"}
    )
    fig.update_layout(legend_title_text="", margin=dict(l=20, r=20, t=55, b=20))
    plotly_chart(fig)
    if np.allclose(pred, pred[0] if len(pred) else 0):
        st.caption("Модельное распределение выглядит почти плоским: проверьте payoff, alpha, prob_revision и noise. Это может означать, что NetLogo не видит различий между темами.")
