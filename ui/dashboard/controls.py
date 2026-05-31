"""Форматирование значений, элементы параметров и базовые панели интерфейса"""
from __future__ import annotations


from .common import *

from src.pipeline.configuration import clear_raw_data_memory_cache

from typing import TYPE_CHECKING, Callable, Any

if TYPE_CHECKING:
    from .session_state import clamp_float
    from .summaries import _discussion_rows, _truthy_series
    from .text_normalization import _clean_visible_text

def describe_result_availability(df: pd.DataFrame | None) -> dict[str, Any]:
    """Определяет, какие разделы отчёта доступны по колонкам таблицы результатов
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        dict[str, Any]: словарь со статусом доступности фактической проверки и текстами для интерфейса
    """
    if df is None or pd.DataFrame(df).empty:
        return {
            "title": "Результат не рассчитан",
            "label": "нет данных",
            "tone": "gray",
            "detail": "Запустите анализ",
            "available_steps": 0,
            "total_steps": 0,
            "has_any_fact": False,
            "has_full_fact": False,
        }
    x = pd.DataFrame(df).copy()
    disc = _discussion_rows(x)
    base = disc if not disc.empty else x.drop_duplicates(subset=["future_step"]) if "future_step" in x else x
    if "future_step" in base:
        total_steps = int(pd.to_numeric(base["future_step"], errors="coerce").dropna().nunique())
    else:
        total_steps = 0
    actual = _truthy_series(base["actual_available"]) if "actual_available" in base else pd.Series(False, index=base.index)
    if "future_step" in base:
        available_steps = int(pd.to_numeric(base.loc[actual, "future_step"], errors="coerce").dropna().nunique())
    else:
        available_steps = int(actual.any())
    has_any = available_steps > 0
    has_full = total_steps > 0 and available_steps >= total_steps
    start_time = None
    planned_end = None
    observed_until = None
    for col, target in [("start_time", "start_time"), ("planned_end_time", "planned_end"), ("observed_until", "observed_until")]:
        if col in x and x[col].notna().any():
            value = str(x[col].dropna().iloc[0])
            if target == "start_time": start_time = value
            if target == "planned_end": planned_end = value
            if target == "observed_until": observed_until = value
    if has_full:
        title = "Симуляция и проверка по данным"
        label = f"проверка доступна: {available_steps}/{total_steps}"
        tone = "green"
        detail = "Для всех будущих шагов горизонта симуляции найдены фактические данные, поэтому можно интерпретировать nRMSE и другие проверочные метрики"
    elif has_any:
        title = "Симуляция и частичная проверка"
        label = f"проверка частичная: {available_steps}/{total_steps}"
        tone = "orange"
        detail = "Часть будущих шагов уже есть в данных, остальные остаются симуляционной траекторией без фактической проверки"
    else:
        title = "Симуляционная траектория без проверки"
        label = f"ожидает будущих данных: 0/{total_steps}"
        tone = "blue"
        detail = "Модельная динамика рассчитана, но фактические будущие шаги дискретизации для проверки ещё отсутствуют в подключённом источнике"
    return {
        "title": title,
        "label": label,
        "tone": tone,
        "detail": detail,
        "available_steps": available_steps,
        "total_steps": total_steps,
        "has_any_fact": has_any,
        "has_full_fact": has_full,
        "start_time": start_time,
        "planned_end_time": planned_end,
        "observed_until": observed_until,
    }


def clear_current_analysis_outputs(paths: dict[str, Path], cfg_obj: AppConfig, clear_state_cache: bool = False) -> None:
    """Удаляет файлы и состояние предыдущего локального анализа
    
    Args:
        paths: словарь путей к результатам, данным и временным файлам
        cfg_obj: объект конфигурации приложения
        clear_state_cache: признак очистки кэша состояния интерфейса
    
    Returns:
        None
    """
    base = Path(paths["forward_indicators"])
    candidates = [
        base,
        base.with_name(base.stem + "_context_messages.csv"),
        base.with_name(base.stem + "_publications.csv"),
    ]
    for path in candidates:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass
    # удаляем рыночную панель прошлого эксперимента из состояния интерфейса
    for key in ["last_forward_indicators", "last_forward_price_panel"]:
        if key in st.session_state:
            del st.session_state[key]
    if clear_state_cache:
        clear_raw_data_memory_cache()
        for rel in [Path("cache") / "window_state", Path("cache") / "raw_window_state"]:
            cache_dir = cfg_obj.output_dir / rel
            try:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
            except Exception:
                pass
        sentiment_cache_dir = cfg_obj.data_dir / "cache"
        try:
            if sentiment_cache_dir.exists():
                for item in sentiment_cache_dir.glob("sentiment_*.jsonl"):
                    try:
                        item.unlink()
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            st.cache_data.clear()
        except Exception:
            pass

def read_json(path: Path) -> dict[str, Any] | None:
    """Читает JSON-файл и возвращает пустую структуру при отсутствии файла
    
    Args:
        path: путь к файлу или директории
    
    Returns:
        dict[str, Any] | None: словарь из JSON-файла или None при отсутствии файла
    """
    if not path.exists():
        return None
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def parse_float_list(text: str, allow_none: bool = False) -> list[float | None]:
    """Разбирает список числовых значений из строки интерфейса
    
    Args:
        text: текст сообщения или текстовое значение
        allow_none: разрешение значения None в списке параметров
    
    Returns:
        list[float | None]: список чисел и значений None, разобранных из строки
    """
    result = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if allow_none and item.lower() in {"none", "null", "нет"}:
            result.append(None)
        else:
            result.append(float(item))
    return result


def progress_writer() -> Callable[[Any], None]:
    """Создаёт callback для вывода прогресса длительных операций в Streamlit
    
    Returns:
        Callable[[Any], None]: функция обратного вызова для обновления текста и прогресса в интерфейсе
    """
    status_box = st.empty()
    bar_box = st.empty()
    caption_box = st.empty()

    def write(msg: Any) -> None:
        """Обновляет текстовый блок прогресса в интерфейсе
        
        Args:
            msg: текст сообщения для вывода прогресса
        
        Returns:
            None
        """
        if isinstance(msg, dict):
            message = str(msg.get("message", "Выполняется..."))
            current = msg.get("current")
            total = msg.get("total")
            stage = str(msg.get("stage", ""))
            status_box.info(message)
            try:
                if current is not None and total:
                    frac = max(0.0, min(1.0, float(current) / float(total)))
                    bar_box.progress(frac)
                    caption_box.caption(f"{stage}: {int(current)}/{int(total)} · {frac*100:.1f}%")
            except Exception:
                pass
        else:
            status_box.info(str(msg))

    return write


def future_steps(n: int) -> list[int]:
    """Формирует подписи будущих шагов дискретизации
    
    Args:
        n: число элементов, шагов или будущих горизонтов
    
    Returns:
        list[int]: список номеров будущих шагов дискретизации от 1 до n
    """
    return list(range(1, int(n) + 1))


def fmt(x: Any, nd: int = 4) -> str:
    """Форматирует число для компактного вывода в интерфейсе
    
    Args:
        x: числовое значение или элемент последовательности
        nd: число знаков после запятой
    
    Returns:
        str: строковое представление числа с заданной точностью или дефис для пустого значения
    """
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def pct(x: Any, nd: int = 1) -> str:
    """Форматирует долю как процентное значение
    
    Args:
        x: числовое значение или элемент последовательности
        nd: число знаков после запятой
    
    Returns:
        str: строковое процентное представление доли или дефис для пустого значения
    """
    if x is None:
        return "-"
    try:
        return f"{float(x) * 100:.{nd}f}%"
    except Exception:
        return str(x)


def format_seconds(value: Any) -> str:
    """Преобразует длительность в секундах в читаемую строку
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        str: читаемая строка длительности в секундах, минутах или часах
    """
    try:
        seconds = float(value)
    except Exception:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f} сек"
    if seconds < 3600:
        return f"{seconds/60:.1f} мин"
    return f"{seconds/3600:.2f} ч"


def nlp_runtime_options(c: AppConfig, key: str = "init") -> dict[str, Any]:
    """Собирает параметры NLP-компонентов из интерфейса
    
    Args:
        c: объект конфигурации приложения
        key: ключ параметра, виджета или словаря
    
    Returns:
        dict[str, Any]: словарь параметров NLP-компонентов для текущего запуска
    """
    st.markdown("**Производительность NLP**")
    st.caption("Тональность считается батчами через FinBERT/lexicon. Тикеры извлекаются отдельным быстрым локальным проходом; GLiNER включайте только для точечной проверки, иначе процесс извлечения тикеров может быть очень долгим.")
    col_a, col_b = st.columns(2)
    with col_a:
        ticker_modes = ["hybrid_fast", "gliner_hybrid", "gliner_full", "none"]
        current_ticker_mode = str(c.get("nlp", "ticker_extraction", default="hybrid_fast"))
        ticker_mode = st.selectbox(
            "Режим извлечения тикеров",
            ticker_modes,
            index=ticker_modes.index(current_ticker_mode) if current_ticker_mode in ticker_modes else 0,
            help="*hybrid_fast*: cashtag + биржевой символ + алиасы + названия компаний. *gliner_hybrid*: быстрый слой + локальный NER GLiNER по политике ниже. *gliner_full*: полноценный NER по всем сообщениям в пределах лимита. *none*: выключить тикеры.",
            key=f"{key}_ticker_mode",
        )
        sentiment_batch = st.number_input(
            "Размер batch-инференса тональности",
            min_value=1, max_value=256, value=int(c.get("nlp", "batch_size", default=32)), step=1,
            help="Для CPU обычно 16-64. Если есть память, можно увеличить. Это не влияет на качество, только на скорость/память.",
            key=f"{key}_sent_batch",
        )
    with col_b:
        ticker_batch = st.number_input(
            "Размер блока обработки тикеров",
            min_value=256, max_value=50000, value=int(c.get("nlp", "ticker_batch_size", default=4096)), step=256,
            help="Это не ML-batch, а технический chunk: сколько сообщений обрабатывать за один блок при извлечении тикеров и обновлении progress-bar. На качество не влияет.",
            key=f"{key}_ticker_batch",
        )
        gliner_policy = st.selectbox(
            "Политика GLiNER",
            ["missing_only", "all"],
            index=0 if str(c.get("nlp", "gliner_policy", default="missing_only")) != "all" else 1,
            help="*missing_only*: запускает NER только там, где быстрый слой не нашёл тикер. *all*: уточняет все сообщения. В режиме gliner_full всегда используется all.",
            key=f"{key}_gliner_policy",
        )
        gliner_max = st.number_input(
            "Лимит сообщений для GLiNER",
            min_value=0, max_value=1000000, value=int(c.get("nlp", "gliner_max_messages", default=2000)), step=500,
            help="0 означает без лимита. В gliner_full это может быть очень долго на больших CSV, но даёт полноценный NER-слой.",
            key=f"{key}_gliner_max",
        )
    return {
        "ticker_extraction": ticker_mode,
        "batch_size": int(sentiment_batch),
        "ticker_batch_size": int(ticker_batch),
        "gliner_policy": "all" if ticker_mode == "gliner_full" else gliner_policy,
        "gliner_max_messages": int(gliner_max),
    }



def default_w_sentiment(c: AppConfig, state: dict[str, Any] | None = None) -> float:
    """Определяет начальный вес тональности для функции выигрыша
    
    Args:
        c: объект конфигурации приложения
        state: состояние Streamlit или сохранённые параметры текущего запуска
    
    Returns:
        float: начальный вес тональности, ограниченный диапазоном от 0 до 1
    """
    state = state or {}
    try:
        value = float(state.get("w_sentiment", c.get("payoff", "w_sentiment", default=0.5)))
    except Exception:
        value = 0.5
    return max(0.0, min(1.0, value))



def _normalize_ui_weights(w_sent: float, w_inf: float) -> tuple[float, float]:
    """Нормирует веса тональности и авторитетности из пользовательского ввода
    
    Args:
        w_sent: вес тональности в функции выигрыша
        w_inf: вес авторитетности в функции выигрыша
    
    Returns:
        tuple[float, float]: нормированная пара весов тональности и авторитетности
    """
    try:
        ws = float(w_sent)
    except Exception:
        ws = 0.5
    try:
        wi = float(w_inf)
    except Exception:
        wi = 1.0 - ws
    total = ws + wi
    if total <= 0:
        return 0.5, 0.5
    return ws / total, wi / total

def weight_inputs(label_prefix: str, default: float, key: str, help_text: str = "") -> tuple[float, float]:
    """Показывает элементы выбора весов функции выигрыша
    
    Args:
        label_prefix: префикс подписи группы элементов интерфейса
        default: резервное значение
        key: ключ параметра, виджета или словаря
        help_text: текст подсказки для интерфейса
    
    Returns:
        tuple[float, float]: пара весов тональности и авторитетности, выбранная в интерфейсе
    """
    default = max(0.0, min(1.0, float(default)))
    st.markdown(f"**{label_prefix}**")
    col_a, col_b = st.columns([1.15, 1])
    with col_a:
        w_sent = st.number_input(
            "Вес тональности в payoff",
            min_value=0.0,
            max_value=1.0,
            value=default,
            step=0.001,
            format="%.3f",
            key=f"{key}_w_sentiment",
            help=help_text or "Единственный редактируемый параметр баланса payoff. 0.500 означает равный вклад тональности и авторитетности. Допустимый диапазон: от 0 до 1.",
        )
    w_sent = max(0.0, min(1.0, float(w_sent)))
    w_inf = 1.0 - w_sent
    with col_b:
        metric_card("Вес авторитетности", fmt(w_inf, 3), "Автоматически = 1 - вес тональности", "orange")
    st.caption(f"Итоговый баланс payoff: тональность {w_sent:.3f} / авторитетность {w_inf:.3f}. Эти значения будут переданы в расчёт payoff.")
    return w_sent, w_inf


def runtime_payoff_path(
    c: AppConfig,
    paths: dict[str, Path],
    w_sent: float,
    w_inf: float,
    sentiment_mode: str,
    tag: str,
) -> Path:
    """Возвращает путь к payoff-файлу для выбранных весов и режима тональности
    
    Args:
        c: объект конфигурации приложения
        paths: словарь путей к результатам, данным и временным файлам
        w_sent: вес тональности в функции выигрыша
        w_inf: вес авторитетности в функции выигрыша
        sentiment_mode: режим преобразования тональности
        tag: короткая метка варианта запуска или файла
    
    Returns:
        Path: путь к payoff-файлу для текущего набора весов и режима тональности
    """
    if not paths["messages"].exists():
        st.warning("Не найден messages_enriched.csv, поэтому используется payoff последней инициализации.")
        return paths["payoff"]
    out = c.output_dir / "runtime" / f"payoff_{tag}.csv"
    recompute_payoff_file(c, paths["messages"], out, w_sent, w_inf, sentiment_mode)
    return out


def range_slider(label: str, min_value: float, max_value: float, value: tuple[float, float], step: float, key: str, help_text: str) -> tuple[float, float]:
    """Создаёт ползунок диапазона с защитой от некорректных границ
    
    Args:
        label: подпись элемента интерфейса
        min_value: нижняя граница допустимого значения
        max_value: верхняя граница допустимого значения
        value: исходное значение, которое требуется нормализовать или преобразовать
        step: шаг изменения значения или номер шага анализа
        key: ключ параметра, виджета или словаря
        help_text: текст подсказки для интерфейса
    
    Returns:
        tuple[float, float]: пара выбранных границ диапазона
    """
    def _clamp_pair(pair: Any) -> tuple[float, float]:
        """Ограничивает пару значений диапазоном ползунка
        
        Args:
            pair: пара значений диапазона
        
        Returns:
            tuple[float, float]: пара границ, приведённая к допустимому диапазону
        """
        try:
            a, b = pair
        except Exception:
            a, b = value
        a = clamp_float(a, min_value, max_value, default=value[0])
        b = clamp_float(b, min_value, max_value, default=value[1])
        if a > b:
            a, b = b, a
        return (a, b)
    value = _clamp_pair(value)
    if key in st.session_state:
        st.session_state[key] = _clamp_pair(st.session_state[key])
    lo, hi = st.slider(label, min_value=min_value, max_value=max_value, value=value, step=step, key=key, help=help_text)
    if lo > hi:
        st.error(f"Для параметра {label} нижняя граница не может быть больше верхней.")
        st.stop()
    return float(lo), float(hi)

def sentiment_mode_index(value: Any) -> int:
    """Возвращает индекс режима преобразования тональности для элемента выбора
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        int: индекс режима тональности в списке вариантов интерфейса
    """
    options = ["magnitude", "signed_shift", "hybrid"]
    try:
        return options.index(str(value))
    except ValueError:
        return 0

def metric_card(title: str, value: str, caption: str = "", tone: str = "blue") -> None:
    """Показывает компактную карточку метрики в интерфейсе
    
    Args:
        title: заголовок блока интерфейса
        value: исходное значение, которое требуется нормализовать или преобразовать
        caption: поясняющая подпись блока интерфейса
        tone: визуальный тон карточки или панели
    
    Returns:
        None
    """
    ui_metric_card(_clean_visible_text(title), _clean_visible_text(value), _clean_visible_text(caption), tone)


def info_panel(title: str, body: str, tone: str = "blue") -> None:
    """Показывает информационный блок с пояснением результата
    
    Args:
        title: заголовок блока интерфейса
        body: основной текст блока интерфейса
        tone: визуальный тон карточки или панели
    
    Returns:
        None
    """
    ui_info_panel(_clean_visible_text(title), _clean_visible_text(body), tone)
