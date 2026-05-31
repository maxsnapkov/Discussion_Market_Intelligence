"""Базовая конфигурация, загрузка данных и нормализация сообщений"""
from __future__ import annotations


from .common import *


@dataclass
class AppConfig:
    """Контейнер конфигурации приложения и путей проекта
    
    Класс хранит словарь настроек, загруженный из YAML-файла
    и предоставляет безопасный доступ к часто используемым директориям
    
    Attributes:
        raw: словарь параметров приложения, полученный из конфигурационного файла
    """
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        """Загружает конфигурацию приложения из YAML-файла
        
        Args:
            path: путь к файлу или директории
        
        Returns:
            объект AppConfig с параметрами из YAML-файла
        """
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {})

    def get(self, *keys: str, default: Any = None) -> Any:
        """Возвращает значение из вложенной секции конфигурации
        
        Args:
            *keys: позиционные параметры, передаваемые во внутренний вызов
            default: резервное значение
        
        Returns:
            значение вложенного параметра или default, если ключ не найден
        """
        value: Any = self.raw
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    @property
    def window_hours(self) -> int:
        """Возвращает длительность шага дискретизации в часах
        
        Returns:
            длительность шага дискретизации в часах
        """
        return int(self.get("project", "window_size_hours", default=6))

    @property
    def output_dir(self) -> Path:
        """Возвращает директорию выходных результатов
        
        Returns:
            путь к директории выходных результатов
        """
        return Path(self.get("project", "output_dir", default="results"))

    @property
    def data_dir(self) -> Path:
        """Возвращает директорию данных проекта
        
        Returns:
            путь к директории данных проекта
        """
        return Path(self.get("project", "data_dir", default="data"))

    @property
    def model_dir(self) -> Path:
        """Возвращает директорию сохранённых моделей
        
        Returns:
            путь к директории сохранённых моделей
        """
        return Path(self.get("project", "model_dir", default="models"))


def ensure_dirs(cfg: AppConfig) -> None:
    """Создаёт директории данных, моделей и результатов, указанные в конфигурации
    
    Args:
        cfg: конфигурация проекта или приложения
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    for p in [cfg.output_dir, cfg.data_dir / "processed", cfg.model_dir]:
        p.mkdir(parents=True, exist_ok=True)




_RAW_DATA_CACHE: dict[tuple[str, float, int], pd.DataFrame] = {}
_RAW_DATA_CACHE_ORDER: list[tuple[str, float, int]] = []
_RAW_DATA_CACHE_MAX = 2


def _validate_raw_csv_path(raw_path: str | Path) -> Path:
    """Проверяет путь к исходному CSV-файлу локального анализа
    
    Args:
        raw_path: путь к исходной таблице сообщений
    
    Returns:
        Path: проверенный путь к существующему CSV-файлу
    """
    path = Path(raw_path).expanduser()
    if not str(raw_path).strip():
        raise FileNotFoundError("Raw dataset path is empty. Select an existing CSV file for local range analysis.")
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset path must be a CSV file, not a directory: {path}")
    return path


def clear_raw_data_memory_cache() -> None:
    """Очищает оперативный кэш нормализованных исходных CSV
    
    Returns:
        None
    """
    _RAW_DATA_CACHE.clear()
    _RAW_DATA_CACHE_ORDER.clear()


def _load_normalized_raw_data(raw_path: str | Path, use_cache: bool = True) -> pd.DataFrame:
    """Загружает исходный CSV и приводит его к внутренней схеме сообщений
    
    Args:
        raw_path: путь к исходной таблице сообщений
        use_cache: разрешение использовать оперативный кэш CSV
    
    Returns:
        pd.DataFrame: нормализованная таблица исходных сообщений
    """
    path = _validate_raw_csv_path(raw_path)
    stat = path.stat()
    key = (str(path.resolve()), float(stat.st_mtime), int(stat.st_size))
    if use_cache:
        cached = _RAW_DATA_CACHE.get(key)
        if cached is not None:
            return cached.copy()
    else:
        _RAW_DATA_CACHE.pop(key, None)
        try:
            _RAW_DATA_CACHE_ORDER.remove(key)
        except ValueError:
            pass
    raw = pd.read_csv(path)
    normalized = normalize_columns(raw)
    if use_cache:
        _RAW_DATA_CACHE[key] = normalized
        _RAW_DATA_CACHE_ORDER.append(key)
        while len(_RAW_DATA_CACHE_ORDER) > _RAW_DATA_CACHE_MAX:
            old = _RAW_DATA_CACHE_ORDER.pop(0)
            _RAW_DATA_CACHE.pop(old, None)
    return normalized.copy()

def _emit_progress(progress, message: str, current: int | None = None, total: int | None = None, stage: str | None = None) -> None:
    """Передаёт сообщение о ходе выполнения в callback интерфейса
    
    Args:
        progress: callback для передачи статуса выполнения
        message: текст сообщения о ходе выполнения
        current: текущее значение
        total: общее число шагов выполнения
        stage: название этапа вычислительного конвейера
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    if progress is None:
        return
    if current is not None and total is not None:
        payload = {"message": message, "current": int(current), "total": int(total), "stage": stage or "process"}
        try:
            progress(payload)
            return
        except Exception:
            pass
    try:
        progress(message)
    except Exception:
        pass


def load_raw_messages(path: str | Path, limit: int | None = None, sample: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """Загружает исходные сообщения с поддержкой ограничения размера и случайной выборки
    
    Args:
        path: путь к файлу или директории
        limit: максимальное число записей или символов
        sample: размер случайной подвыборки
        random_state: случайное зерно для воспроизводимости
    
    Returns:
        загруженный результат: исходные сообщения с поддержкой ограничения размера и случайной выборки
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if sample and sample > 0 and sample < len(df):
        df = df.sample(sample, random_state=random_state)
    df = normalize_columns(df)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if limit and limit > 0:
        df = df.head(limit).copy()
    return df


def _clean_column_name(value: Any) -> str:
    """Нормализует название колонки для сопоставления с ожидаемой схемой
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        очищенное название колонки
    """
    return str(value).replace("\ufeff", "").strip()


SERVICE_AUTHOR_VALUES = {
    "unknown", "__unknown__", "неизвестен", "не определён", "not defined",
    "[deleted]", "deleted", "u/[deleted]", "[removed]", "removed",
    "automoderator", "auto moderator", "bot", "[bot]",
}

SERVICE_AUTHOR_DISPLAY = {
    "unknown": "автор не определён",
    "deleted": "автор удалён",
    "automoderator": "AutoModerator / служебный автор",
}

_SERVICE_AUTHOR_NODE_PREFIX = "__service_author__"
_SPLIT_SERVICE_AUTHOR_KINDS = {"unknown", "deleted"}


def _service_author_node_kind(value: Any) -> str | None:
    """Определяет тип служебного автора по исходному значению
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        определённое значение: тип служебного автора по исходному значению
    """
    text = str(value).strip().lower()
    prefix = _SERVICE_AUTHOR_NODE_PREFIX + ":"
    if not text.startswith(prefix):
        return None
    parts = text.split(":", 2)
    if len(parts) >= 2 and parts[1] in SERVICE_AUTHOR_DISPLAY:
        return parts[1]
    return None


def _canonical_service_author(text: str) -> str | None:
    """Возвращает каноническое имя служебного автора
    
    Args:
        text: текст сообщения или текстовое значение
    
    Returns:
        каноническое имя служебного автора
    """
    value = str(text).strip()
    node_kind = _service_author_node_kind(value)
    if node_kind is not None:
        return node_kind
    low = value.lower()
    if low in {"", "nan", "none", "null", "unknown", "__unknown__", "неизвестен", "не определён", "not defined"}:
        return "unknown"
    if low in {"[deleted]", "deleted", "u/[deleted]", "[removed]", "removed"}:
        return "deleted"
    if low in {"automoderator", "auto moderator", "[bot]", "bot"}:
        return "automoderator"
    return None


def is_service_author(value: Any) -> bool:
    """Проверяет, относится ли автор к удалённым, неизвестным или техническим записям
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        bool: автор является служебным, удалённым или неизвестным
    """
    return _canonical_service_author(str(value)) is not None


def _safe_author_node_suffix(message_id: Any, row_idx: Any) -> str:
    """Создаёт стабильный суффикс технического узла автора
    
    Args:
        message_id: идентификатор сообщения
        row_idx: номер строки для резервной идентификации
    
    Returns:
        созданная структура: стабильный суффикс технического узла автора
    """
    raw = str(message_id).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        raw = f"row_{row_idx}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:64]
    if not safe:
        safe = hashlib.sha1(str(row_idx).encode("utf-8")).hexdigest()[:12]
    return safe


def _service_author_node(canonical: str, message_id: Any, row_idx: Any) -> str:
    """Формирует идентификатор технического узла для служебного автора
    
    Args:
        canonical: каноническое значение автора или служебного узла
        message_id: идентификатор сообщения
        row_idx: номер строки для резервной идентификации
    
    Returns:
        сформированная структура: идентификатор технического узла для служебного автора
    """
    suffix = _safe_author_node_suffix(message_id, row_idx)
    return f"{_SERVICE_AUTHOR_NODE_PREFIX}:{canonical}:{suffix}"


def service_author_label(value: Any) -> str:
    """Возвращает читаемую подпись для служебного автора
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        читаемую подпись для служебного автора
    """
    text = str(value).strip()
    canonical = _canonical_service_author(text)
    if canonical is None:
        return text
    label = SERVICE_AUTHOR_DISPLAY.get(canonical, "служебный/неопределённый автор")
    if text.lower().startswith((_SERVICE_AUTHOR_NODE_PREFIX + ":")):
        suffix = text.split(":", 2)[-1] if ":" in text else ""
        suffix = suffix[:10] if suffix else ""
        return f"{label} #{suffix}" if suffix else label
    return label


def _clean_author_value(value: Any, message_id: Any | None = None, row_idx: Any | None = None) -> str:
    """Нормализует автора и заменяет служебные значения техническими узлами
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
        message_id: идентификатор сообщения
        row_idx: номер строки для резервной идентификации
    
    Returns:
        нормализованное значение: автора и заменяет служебные значения техническими узлами
    """
    text = str(value).strip()
    service = _canonical_service_author(text)
    if service in _SPLIT_SERVICE_AUTHOR_KINDS and message_id is not None:
        return _service_author_node(service, message_id, row_idx)
    if service is not None:
        return service
    return text


def _assign_author_nodes(df: pd.DataFrame, source_col: str | None) -> pd.DataFrame:
    """Добавляет в таблицу канонические идентификаторы и подписи авторов
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
        source_col: название исходной колонки автора
    
    Returns:
        обновлённая структура с добавленными полями: в таблицу канонические идентификаторы и подписи авторов
    """
    raw_values = df[source_col].fillna("").astype(str) if source_col else pd.Series(["unknown"] * len(df), index=df.index)
    authors = []
    labels = []
    for pos, (idx, raw) in enumerate(raw_values.items()):
        message_id = df.at[idx, "message_id"] if "message_id" in df.columns else idx
        author = _clean_author_value(raw, message_id=message_id, row_idx=idx)
        authors.append(author)
        labels.append(service_author_label(author))
    df["author"] = authors
    df["author_display"] = labels
    return df


def _column_lookup_key(value: Any) -> str:
    """Приводит название колонки к ключу для нечувствительного поиска
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        ключ колонки без регистра и разделителей
    """
    return re.sub(r"[^a-z0-9]+", "", _clean_column_name(value).lower())


def _first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    # сопоставляем колонки без учёта регистра и разделителей
    # например createdAt, created_at, post id и CommentID приводятся к единой форме
    """Находит первую колонку из списка возможных названий
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
        names: набор допустимых имён или словарь названий
    
    Returns:
        название первой найденной колонки или None
    """
    lookup = {_column_lookup_key(c): c for c in df.columns}
    exact = {_clean_column_name(c).lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        low = _clean_column_name(name).lower()
        key = _column_lookup_key(name)
        if low in exact:
            return exact[low]
        if key in lookup:
            return lookup[key]
    return None


RAW_COLUMN_ALIASES: dict[str, list[str]] = {
    "text": ["text", "text_clean", "selftext", "body", "content", "message", "comment_text", "post_text"],
    "title": ["title", "headline", "post_title"],
    "timestamp": ["timestamp", "created_utc", "created_at", "createdAt", "created", "date", "datetime", "published_at", "publishedAt", "time"],
    "author": ["author", "Author", "author_name", "authorName", "username", "user", "screen_name", "screenName", "source", "channel", "subreddit"],
    "author_id": ["author_id", "authorId", "user_id", "userId", "uid"],
    "message_id": ["id", "message_id", "messageId"],
    "post_id": ["post id", "post_id", "postId", "postid", "link_id", "linkId", "submission_id", "submissionId"],
    "comment_id": ["Comment id", "comment_id", "commentId", "commentid"],
    "parent_id": ["Parent id", "parent_id", "parentId", "parentid", "parent"],
    "subreddit": ["subreddit", "subreddit_id", "subredditId"],
    "depth": ["depth", "level", "comment_depth"],
    "type": ["type", "kind", "message_type", "messageType"],
    "score": ["score", "ups", "upvotes", "likes", "reactions", "rating"],
    "ups": ["ups", "upvotes", "likes"],
    "children": ["childCount", "child_count", "children", "num_comments", "comments", "comment_count", "replies"],
}


def detect_raw_schema(df: pd.DataFrame) -> dict[str, str | None]:
    """Определяет соответствие исходных колонок внутренним полям проекта
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        определённое значение: соответствие исходных колонок внутренним полям проекта
    """
    data = df.copy()
    data.columns = [_clean_column_name(c) for c in data.columns]
    return {field: _first_present(data, aliases) for field, aliases in RAW_COLUMN_ALIASES.items()}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит таблицу сообщений к внутренней схеме проекта
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        таблица сообщений во внутренней схеме проекта
    """
    df = df.copy()
    df.columns = [_clean_column_name(c) for c in df.columns]
    schema = detect_raw_schema(df)
    text_col = schema.get("text")
    title_col = schema.get("title")
    time_col = schema.get("timestamp")
    author_col = schema.get("author") or schema.get("author_id")
    id_col = schema.get("message_id")
    post_col = schema.get("post_id")
    comment_col = schema.get("comment_id")
    parent_col = schema.get("parent_id")
    type_col = schema.get("type")
    score_col = schema.get("score")
    ups_col = schema.get("ups")
    child_col = schema.get("children")

    if text_col is None and title_col is None:
        raise ValueError("Dataset must contain text/selftext/body/content/message or title/headline column")
    if time_col is None:
        raise ValueError("Dataset must contain timestamp/created_utc/date/datetime column")

    title = df[title_col].fillna("").astype(str) if title_col else ""
    body = df[text_col].fillna("").astype(str) if text_col else ""
    if title_col and text_col and title_col != text_col:
        df["text"] = (title + "\n" + body).str.strip()
    elif text_col:
        df["text"] = body.str.strip()
    else:
        df["text"] = title.astype(str).str.strip()

    if id_col:
        df["message_id"] = df[id_col].astype(str)
    elif comment_col:
        df["message_id"] = df[comment_col].astype(str)
    elif post_col:
        df["message_id"] = df[post_col].astype(str)
    else:
        df["message_id"] = [str(i) for i in range(len(df))]

    df["post_ref_id"] = df[post_col].astype(str) if post_col else ""
    df["comment_ref_id"] = df[comment_col].astype(str) if comment_col else ""
    df["parent_ref_id"] = df[parent_col].astype(str) if parent_col else ""
    df["message_type"] = df[type_col].fillna("").astype(str).str.lower() if type_col else ""
    df["score_value"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0) if score_col else 0.0
    df["ups_value"] = pd.to_numeric(df[ups_col], errors="coerce").fillna(0.0) if ups_col else 0.0
    df["child_count"] = pd.to_numeric(df[child_col], errors="coerce").fillna(0.0) if child_col else 0.0

    if schema.get("subreddit"):
        df["subreddit_ref_id"] = df[schema["subreddit"]].astype(str)
    if schema.get("depth"):
        df["comment_depth"] = pd.to_numeric(df[schema["depth"]], errors="coerce").fillna(0).astype(int)
    else:
        df["comment_depth"] = 0

    def _own_text_for_ticker(row: pd.Series) -> str:
        """Извлекает текст собственной строки для проверки тикерного контекста
        
        Args:
            row: строка таблицы с сообщением, инструментом или событием
        
        Returns:
            текст текущего сообщения для проверки упоминания тикера
        """
        text_value = str(row.get("text", "") or "")
        depth_value = int(row.get("comment_depth", 0) or 0)
        parent_value = str(row.get("parent_ref_id", "") or "").strip()
        if (depth_value > 0 or parent_value) and "\n" in text_value:
            tail = "\n".join(text_value.split("\n")[1:]).strip()
            if tail:
                return tail
        return text_value.strip()

    df["text_for_ticker"] = df.apply(_own_text_for_ticker, axis=1)

    df = _assign_author_nodes(df, author_col)

    ts = df[time_col]
    if np.issubdtype(ts.dtype, np.number):
        # created_utc обычно хранится в секундах, но отдельные выгрузки используют миллисекунды
        median = float(ts.dropna().median()) if ts.notna().any() else 0
        unit = "ms" if median > 10_000_000_000 else "s"
        df["timestamp"] = pd.to_datetime(ts, unit=unit, errors="coerce", utc=True).dt.tz_localize(None)
    else:
        df["timestamp"] = pd.to_datetime(ts, errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["timestamp"])
    df = df[df["text"].astype(str).str.len() > 0].copy()
    return df

def clean_text(text: str) -> str:
    """Очищает текст сообщения без удаления финансово значимых обозначений
    
    Args:
        text: текст сообщения или текстовое значение
    
    Returns:
        очищенный текст без лишних пробелов и завершающей точки при необходимости
    """
    text = re.sub(r"https?://\S+", " ", str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
