"""Загрузка пользовательских CSV и предпросмотр исходных данных"""
from __future__ import annotations


from .common import *

from .controls import info_panel, metric_card

def save_uploaded_raw_csv(uploaded_file: Any, cfg_obj: AppConfig) -> Path:
    """Сохраняет загруженный пользователем CSV-файл в директорию данных проекта
    
    Args:
        uploaded_file: файл, загруженный пользователем через интерфейс
        cfg_obj: объект конфигурации приложения
    
    Returns:
        Path: путь к сохранённому CSV-файлу пользователя
    """
    name = str(getattr(uploaded_file, "name", "uploaded_raw.csv") or "uploaded_raw.csv")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).name).strip("._") or "uploaded_raw.csv"
    stem = Path(safe_name).stem or "uploaded_raw"
    suffix = Path(safe_name).suffix.lower() or ".csv"
    if suffix != ".csv":
        suffix = ".csv"

    content = bytes(uploaded_file.getbuffer())
    digest = hashlib.sha256(content).hexdigest()[:16]
    signature = f"{safe_name}|{len(content)}|{digest}"

    target_dir = cfg_obj.data_dir / "raw" / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{stem}_{digest}{suffix}"

    previous_signature = st.session_state.get("market_local_raw_upload_signature")
    previous_path = Path(str(st.session_state.get("market_local_raw_upload_path", ""))).expanduser()
    if previous_signature == signature and previous_path.exists() and previous_path.is_file():
        target = previous_path
    else:
        if not target.exists():
            target.write_bytes(content)
        st.session_state["market_local_raw_upload_signature"] = signature
        st.session_state["market_local_raw_upload_path"] = str(target)

    st.session_state["connected_raw_path"] = str(target)
    st.session_state["market_local_raw_path"] = str(target)
    return target



def _preview_short(value: Any, limit: int = 110) -> str:
    """Сокращает длинное значение для компактного предпросмотра таблицы
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
        limit: максимальное число записей или символов
    
    Returns:
        str: сокращённая строка без переносов для табличного предпросмотра
    """
    text = str(value if value is not None else "").replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit] + ("..." if len(text) > limit else "")



@st.cache_data(show_spinner=False, max_entries=8)
def _load_raw_preview_sample_cached(path_text: str, mtime: float, size: int, nrows: int = 500) -> pd.DataFrame:
    """Загружает первые строки исходного CSV с кэшированием по состоянию файла
    
    Args:
        path_text: строковое представление пути к файлу
        mtime: время последнего изменения файла
        size: размер элемента интерфейса
        nrows: число строк для загрузки
    
    Returns:
        pd.DataFrame: таблица с первыми строками исходного CSV-файла
    """
    path = Path(path_text).expanduser()
    return pd.read_csv(path, nrows=int(nrows))

def render_raw_dataset_preview(raw_path: Path | None, *, key_prefix: str = "raw_preview") -> None:
    """Показывает сводку и предпросмотр исходного массива сообщений
    
    Args:
        raw_path: путь к исходной таблице сообщений
        key_prefix: префикс ключей Streamlit для независимых виджетов
    
    Returns:
        None
    """
    if raw_path is None:
        info_panel("Предпросмотр данных", "Подключите CSV, чтобы увидеть распознавание колонок", "gray")
        return
    raw_path = Path(raw_path).expanduser()
    if not raw_path.exists() or not raw_path.is_file():
        info_panel("Предпросмотр данных", "Файл не найден или путь указывает на папку", "gray")
        return
    try:
        stat = raw_path.stat()
        sample = _load_raw_preview_sample_cached(str(raw_path), float(stat.st_mtime), int(stat.st_size), 500)
    except Exception as exc:
        st.warning(f"Не удалось прочитать предпросмотр CSV: {exc}")
        return
    if sample.empty:
        info_panel("Предпросмотр данных", "CSV прочитан, но первые строки пустые", "gray")
        return
    schema = detect_raw_schema(sample)
    required_ok = bool(schema.get("timestamp")) and bool(schema.get("text") or schema.get("title"))
    try:
        normalized = normalize_columns(sample)
    except Exception as exc:
        normalized = pd.DataFrame()
        st.warning(f"Колонки найдены частично, но нормализация не выполнена: {exc}")

    st.markdown("#### Как распарсился источник")
    chips = []
    field_labels = {
        "timestamp": "дата",
        "text": "текст",
        "title": "заголовок",
        "author": "автор",
        "author_id": "id автора",
        "message_id": "id сообщения",
        "post_id": "id поста",
        "comment_id": "id комментария",
        "parent_id": "родитель",
        "score": "оценка",
        "children": "ответы",
    }
    for field in ["timestamp", "text", "title", "author", "author_id", "message_id", "post_id", "comment_id", "parent_id", "score", "children"]:
        source = schema.get(field)
        if source:
            chips.append((f"{field_labels.get(field, field)}: {source}", "blue"))
    if chips:
        chip_row(chips[:12])
    else:
        st.caption("Не удалось автоматически сопоставить колонки")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Строк в предпросмотре", str(len(sample)), "первые строки CSV", "blue")
    with c2:
        metric_card("Распознано полей", str(sum(1 for v in schema.values() if v)), "из известных алиасов", "violet")
    with c3:
        metric_card("Готовность", "ок" if required_ok else "нужна проверка", "дата и текст обязательны", "green" if required_ok else "orange")

    if not normalized.empty:
        view_cols = [c for c in ["timestamp", "author_display", "author", "text", "message_id", "post_ref_id", "comment_ref_id", "parent_ref_id", "score_value", "child_count"] if c in normalized.columns]
        preview = normalized[view_cols].head(6).copy()
        if "text" in preview.columns:
            preview["text"] = preview["text"].map(_preview_short)
        rename = {
            "timestamp": "Дата",
            "author_display": "Автор",
            "author": "Узел автора",
            "text": "Текст",
            "message_id": "ID сообщения",
            "post_ref_id": "ID поста",
            "comment_ref_id": "ID комментария",
            "parent_ref_id": "Родитель",
            "score_value": "Оценка",
            "child_count": "Ответы",
        }
        st.dataframe(
            preview.rename(columns=rename),
            width="stretch",
            hide_index=True,
            column_config={
                "Дата": st.column_config.DatetimeColumn("Дата", format="YYYY-MM-DD HH:mm"),
                "Текст": st.column_config.TextColumn("Текст", width="large"),
            },
        )
        time_min = pd.to_datetime(normalized["timestamp"], errors="coerce").dropna().min() if "timestamp" in normalized else pd.NaT
        time_max = pd.to_datetime(normalized["timestamp"], errors="coerce").dropna().max() if "timestamp" in normalized else pd.NaT
        if not pd.isna(time_min) and not pd.isna(time_max):
            st.caption(f"Диапазон времени в предпросмотре: {time_min} - {time_max}")
