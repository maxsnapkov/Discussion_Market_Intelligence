"""Состояние интерфейса, конфигурация, выбор интервалов и служебные утилиты"""
from __future__ import annotations


from .common import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .controls import metric_card

def _set_widget_state_default(key: str, value: Any) -> None:
    """Записывает значение в состояние Streamlit только при отсутствии ключа
    
    Args:
        key: ключ параметра, виджета или словаря
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        None
    """
    if key not in st.session_state:
        st.session_state[key] = value

def _reset_widget_state_value(key: str, value: Any) -> None:
    """Принудительно обновляет значение в состоянии Streamlit по ключу виджета
    
    Args:
        key: ключ параметра, виджета или словаря
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        None
    """
    st.session_state[key] = value

def clamp_float(value: Any, min_value: float, max_value: float, default: float | None = None) -> float:
    """Ограничивает вещественное значение заданным диапазоном
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
        min_value: нижняя граница допустимого значения
        max_value: верхняя граница допустимого значения
        default: резервное значение
    
    Returns:
        float: вещественное значение, ограниченное заданными границами
    """
    try:
        v = float(value)
    except Exception:
        v = float(default if default is not None else min_value)
    if not np.isfinite(v):
        v = float(default if default is not None else min_value)
    return float(max(float(min_value), min(float(max_value), v)))


def parse_alpha_value(value: Any) -> float | None:
    """Преобразует значение параметра alpha из интерфейса в число или автоматический режим
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        float | None: числовое значение alpha или None для режима без взвешивания
    """
    text = str(value).strip().lower()
    if text in {"", "none", "null", "nan", "нет", "unweighted"}:
        return None
    return clamp_float(text, 0.0, 1.0, default=0.2)


def _load_env_file_safely(path: Path, override: bool = False) -> tuple[bool, bool]:
    """Загружает пары ключ значение из env-файла без перезаписи системных переменных при необходимости
    
    Args:
        path: путь к файлу или директории
        override: признак перезаписи существующих переменных окружения
    
    Returns:
        tuple[bool, bool]: первый флаг показывает наличие файла, второй флаг показывает факт загрузки переменных
    """
    path = Path(path).expanduser()
    if not path.exists() or not path.is_file():
        return False, False
    loaded = False
    try:
        for raw_line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                continue
            value = value.strip()
            for _ in range(3):
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                    value = value[1:-1].strip()
                else:
                    break
            if override or key not in os.environ:
                os.environ[key] = value
                loaded = True
    except Exception:
        return True, False
    return True, loaded


def _env_candidates() -> list[Path]:
    """Возвращает возможные расположения env-файлов в порядке приоритета
    
    Returns:
        list[Path]: список возможных путей к env-файлам в порядке приоритета
    """
    roots = [Path.cwd(), Path(__file__).resolve().parent]
    candidates: list[Path] = []
    explicit = os.getenv("APP_ENV_FILE")
    if explicit:
        explicit_path = Path(explicit).expanduser()
        candidates.append(explicit_path if explicit_path.is_absolute() else Path.cwd() / explicit_path)
    for root in roots:
        candidates.append(root / ".env")
    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate.absolute())
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def load_environment() -> tuple[Path | None, bool]:
    """Загружает переменные окружения перед чтением конфигурации приложения
    
    Returns:
        tuple[Path | None, bool]: путь к найденному env-файлу и признак его обнаружения
    """
    found_path: Path | None = None
    any_found = False
    for env_path in _env_candidates():
        file_found, _loaded = _load_env_file_safely(env_path, override=False)
        if file_found:
            found_path = env_path
            any_found = True
            break
    for root in [Path.cwd(), Path(__file__).resolve().parent]:
        local_env = root / ".env.local"
        file_found, _loaded = _load_env_file_safely(local_env, override=True)
        if file_found and found_path is None:
            found_path = local_env
            any_found = True
    return found_path, any_found

ENV_PATH, ENV_LOADED = load_environment()
CONFIG_PATH = Path(os.getenv("APP_CONFIG", "config/app_config.yaml"))


def cfg() -> AppConfig:
    """Создаёт объект конфигурации приложения из файла настроек
    
    Returns:
        AppConfig: объект конфигурации приложения, загруженный из YAML-файла
    """
    return AppConfig.load(CONFIG_PATH)


def state_paths(c: AppConfig) -> dict[str, Path]:
    """Формирует набор путей к файлам состояния, результатам и кэшу приложения
    
    Args:
        c: объект конфигурации приложения
    
    Returns:
        dict[str, Path]: словарь путей к файлам состояния, кэша и результатов
    """
    processed = c.data_dir / "processed"
    return {
        "messages": processed / "messages_enriched.csv",
        "windows": processed / "windows.json",
        "payoff": processed / "payoff.csv",
        "state": c.output_dir / "project_state.json",
        "grid_calibration": c.output_dir / "calibration" / "calibration_results.csv",
        "optuna_calibration": c.output_dir / "calibration" / "optuna_results.csv",
        "best_params": c.output_dir / "calibration" / "best_params.json",
        "forecasts": c.output_dir / "window_modeling" / "window_forecasts.csv",
        "events": c.output_dir / "window_modeling" / "local_deviation_events.csv",
        "rebuild": c.output_dir / "window_modeling" / "rebuild_decision.json",
        "forward_indicators": c.output_dir / "market" / "forward_discussion_indicators.csv",
        "topic_info": processed / "topic_info.csv",
        "topic_info_detailed": processed / "topic_info_detailed.csv",
        "topic_mapping": processed / "topic_mapping.csv",
    }




def _window_time_value(w: dict[str, Any]) -> pd.Timestamp | None:
    """Возвращает временную метку начала шага из строки истории
    
    Args:
        w: строка шага дискретизации или значение веса
    
    Returns:
        pd.Timestamp | None: временная метка начала шага дискретизации или None при ошибке преобразования
    """
    t = pd.to_datetime(w.get("time"), errors="coerce")
    return None if pd.isna(t) else pd.Timestamp(t)


def _select_discretization_interval(windows: list[dict[str, Any]], *, key_prefix: str, step_hours: int, require_future: bool = False) -> tuple[list[int], pd.Timestamp | None, pd.Timestamp | None, int]:
    """Показывает выбор интервала шагов для локального анализа
    
    Args:
        windows: таблица шагов моделирования
        key_prefix: префикс ключей Streamlit для независимых виджетов
        step_hours: длительность шага дискретизации в часах
        require_future: требование наличия будущих шагов для проверки
    
    Returns:
        tuple[list[int], pd.Timestamp | None, pd.Timestamp | None, int]: индексы выбранных шагов, начало интервала, конец интервала и число публикаций
    """
    times = [_window_time_value(w) for w in windows]
    valid_times = [t for t in times if t is not None]
    if not valid_times:
        st.warning("Нет подготовленных шагов моделирования с корректной датой")
        return [], None, None, 0
    min_t, max_t = min(valid_times), max(valid_times) + pd.Timedelta(hours=int(step_hours))
    if f"{key_prefix}_interval_init" not in st.session_state:
        st.session_state[f"{key_prefix}_start_date"] = min_t.date()
        st.session_state[f"{key_prefix}_start_hour"] = int(min_t.hour)
        st.session_state[f"{key_prefix}_end_date"] = max_t.date()
        st.session_state[f"{key_prefix}_end_hour"] = int(max_t.hour)
        st.session_state[f"{key_prefix}_interval_init"] = True
    st.markdown("##### Временной интервал")
    c1, c2 = st.columns(2)
    with c1:
        d0 = st.date_input("Начало интервала", key=f"{key_prefix}_start_date")
        h0 = st.number_input("Час начала", min_value=0, max_value=23, step=1, key=f"{key_prefix}_start_hour")
    with c2:
        d1 = st.date_input("Конец интервала", key=f"{key_prefix}_end_date")
        h1 = st.number_input("Час конца", min_value=0, max_value=23, step=1, key=f"{key_prefix}_end_hour")
    start_ts = pd.Timestamp(d0) + pd.Timedelta(hours=int(h0))
    end_ts = pd.Timestamp(d1) + pd.Timedelta(hours=int(h1))
    if end_ts <= start_ts:
        st.error("Конец интервала должен быть позже начала")
        return [], start_ts, end_ts, 0
    last_allowed = len(windows) - 2 if require_future else len(windows) - 1
    indices = []
    post_count = 0
    for i, (w, t) in enumerate(zip(windows, times)):
        if t is None or i > last_allowed:
            continue
        if start_ts <= t < end_ts:
            indices.append(i)
            post_count += int(w.get("n_posts", w.get("messages", 0)) or 0)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Шагов моделирования", str(len(indices)), f"по {int(step_hours)} ч", "blue")
    with c2: metric_card("Публикаций", str(post_count), "попали в интервал", "green")
    with c3: metric_card("Длина", f"{max(0, int((end_ts - start_ts).total_seconds() // 3600))} ч", "выбранный интервал", "gray")
    if not indices:
        st.warning("В выбранный интервал не попали подготовленные публикации")
    return indices, start_ts, end_ts, post_count


def _write_temp_windows_subset(windows: list[dict[str, Any]], indices: list[int], cfg_obj: AppConfig, name: str) -> Path:
    """Сохраняет выбранный фрагмент истории шагов моделирования во временный CSV-файл
    
    Args:
        windows: таблица шагов моделирования
        indices: индексы строк или шагов моделирования
        cfg_obj: объект конфигурации приложения
        name: имя файла, функции или элемента интерфейса
    
    Returns:
        Path: путь к временному JSON-файлу с выбранными шагами моделирования
    """
    subset = [windows[i] for i in indices if 0 <= i < len(windows)]
    out = cfg_obj.output_dir / "runtime" / f"{name}_windows_subset.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

def resolve_raw_dataset_path(cfg_obj: AppConfig, state_obj: dict[str, Any]) -> Path | None:
    """Определяет исходный CSV-файл для локального анализа выбранного диапазона
    
    Args:
        cfg_obj: объект конфигурации приложения
        state_obj: словарь состояния приложения с путями и выбранными файлами
    
    Returns:
        Path | None: путь к найденному исходному CSV-файлу или None, если файл не найден
    """
    candidates: list[Path] = []
    for key in ["connected_raw_path", "market_local_raw_path"]:
        value = str(st.session_state.get(key, "") or "").strip()
        if value:
            candidates.append(Path(value))
    live_env = str(os.getenv("APP_LIVE_RAW_CSV", "") or "").strip()
    if live_env:
        candidates.append(Path(live_env))
    raw_value = str(state_obj.get("raw_path", "") or "").strip()
    if raw_value:
        candidates.append(Path(raw_value))
    try:
        for cfg_key in ["live_raw_path", "raw_path"]:
            configured = cfg_obj.get("data", cfg_key, default=None)
            if configured:
                candidates.append(Path(str(configured)))
    except Exception:
        pass
    candidates.extend([
        cfg_obj.data_dir / "raw" / "2024wallstreetbets.csv",
        Path("data/raw/2024wallstreetbets.csv"),
    ])
    raw_dir = cfg_obj.data_dir / "raw"
    if raw_dir.exists():
        candidates.extend(sorted(raw_dir.glob("*.csv")))

    seen: set[str] = set()
    for candidate in candidates:
        if not str(candidate).strip():
            continue
        candidate = candidate.expanduser()
        key = str(candidate.resolve()) if candidate.exists() else str(candidate.absolute())
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None
