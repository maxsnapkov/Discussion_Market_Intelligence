"""Отбор и отображение публикаций, связанных с тикерными событиями"""
from __future__ import annotations


from .common import *

from .controls import fmt
from .topic_views import _topic_id, topic_display_name, topic_name_map

c: Any = None

def _display_author(row: Any) -> str:
    """Выбирает читаемое имя автора для карточки публикации
    
    Args:
        row: строка таблицы с сообщением, инструментом или событием
    
    Returns:
        str: читаемое имя автора с учётом служебных и удалённых пользователей
    """
    for key in ["author_display", "Автор"]:
        try:
            value = row.get(key)
        except Exception:
            value = None
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return text
    for key in ["author", "Author", "автор", "username", "user", "screen_name", "source", "channel"]:
        try:
            value = row.get(key)
        except Exception:
            value = None
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return service_author_label(text)
    return "автор не определён"


_BAD_TICKER_VALUES = {"", "NAN", "NONE", "NULL", "[]", "-", "__DISCUSSION__"}


def _clean_ticker_token(value: Any) -> str | None:
    """Нормализует строковое обозначение финансового инструмента
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        str | None: нормализованный тикер или None для некорректного значения
    """
    text = str(value).strip().upper().replace("$", "")
    text = text.strip(" [](){}'\"\n\r\t")
    if not text or text in _BAD_TICKER_VALUES:
        return None
    # сохраняем биржевые формы вроде BRK B и BRK-B, но отбрасываем фразы
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", text):
        return None
    return text


def _parse_ticker_value(value: Any) -> set[str]:
    """Извлекает множество тикеров из значения таблицы
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        set[str]: множество тикеров, извлечённых из строки, списка или словаря
    """
    out: set[str] = set()
    if value is None:
        return out
    try:
        if pd.isna(value):
            return out
    except Exception:
        pass

    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        raw = str(value).strip()
        if not raw or raw.upper() in _BAD_TICKER_VALUES:
            return out
        items = None
        if raw.startswith("[") or raw.startswith("{"):
            import json, ast
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(raw)
                    items = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]
                    break
                except Exception:
                    items = None
        if items is None:
            # поддерживаем списки тикеров через запятую, JSON-подобную строку и разделитель
            cleaned = raw.replace(";", ",").replace("|", ",").replace("[", ",").replace("]", ",")
            items = [x for x in cleaned.split(",") if x.strip()]

    for item in items:
        if isinstance(item, dict):
            for key in ("ticker", "symbol", "value"):
                token = _clean_ticker_token(item.get(key))
                if token:
                    out.add(token)
                    break
        else:
            token = _clean_ticker_token(item)
            if token:
                out.add(token)
    return out


def _post_mentions_tickers(row: Any) -> set[str]:
    """Определяет тикеры, явно связанные с публикацией
    
    Args:
        row: строка таблицы с сообщением, инструментом или событием
    
    Returns:
        set[str]: множество тикеров, связанных с публикацией по доступным полям
    """
    tickers: set[str] = set()
    for key in ["publication_tickers", "tickers", "ticker", "тикеры", "ticker_mentions", "mentions"]:
        try:
            tickers |= _parse_ticker_value(row.get(key))
        except Exception:
            continue
    return tickers


def _soft_text_has_ticker(row: Any, ticker: str) -> bool:
    """Проверяет наличие тикера в тексте без строгой привязки к извлечённым полям
    
    Args:
        row: строка таблицы с сообщением, инструментом или событием
        ticker: биржевой идентификатор финансового инструмента
    
    Returns:
        bool: True, если тикер найден в текстовых полях сообщения
    """
    ticker = str(ticker).upper().strip()
    haystack = " ".join(str(row.get(k, "")) for k in ["text", "text_clean", "title", "selftext", "публикация"]).upper()
    return bool(ticker and re.search(rf"(?<![A-Z0-9])\$?{re.escape(ticker)}(?![A-Z0-9])", haystack))




def load_publications_for_signal_window(
    messages_path: str | Path,
    forward_indicators_path: str | Path,
    start_time: Any,
    window_hours: int,
    tickers: list[str] | None = None,
    top_n: int = 18,
) -> tuple[pd.DataFrame, str]:
    """Загружает публикации, связанные с выбранным шагом дискретизации и тикерами
    
    Args:
        messages_path: путь к таблице обработанных сообщений
        forward_indicators_path: путь к таблице дискуссионных индикаторов
        start_time: временная метка начала анализируемого шага дискретизации
        window_hours: длительность шага дискретизации в часах
        tickers: набор биржевых идентификаторов финансовых инструментов
        top_n: число наиболее значимых записей для отбора
    
    Returns:
        tuple[pd.DataFrame, str]: таблица публикаций и текстовое описание источника данных
    """
    if start_time is None:
        return pd.DataFrame(), "start time is not defined"

    base = Path(forward_indicators_path)
    context_publications_path = base.with_name(base.stem + "_context_publications.csv")
    if context_publications_path.exists():
        try:
            posts = pd.read_csv(context_publications_path)
            if posts is not None and not posts.empty:
                return posts, "публикации исторического контекста и текущего шага дискретизации"
        except Exception:
            pass
    context_path = base.with_name(base.stem + "_context_messages.csv")
    sources: list[tuple[Path, str]] = []
    if context_path.exists():
        sources.append((context_path, "контекст текущего локального расчёта"))
    msg_path = Path(messages_path)
    if msg_path.exists():
        sources.append((msg_path, "обогащённые сообщения последней инициализации"))

    last_empty_source = "нет доступного источника сообщений"
    for source, label in sources:
        try:
            posts = top_publications_for_window(source, start_time, int(window_hours), tickers=tickers, top_n=int(top_n))
            if posts is not None and not pd.DataFrame(posts).empty:
                return posts, label
            last_empty_source = label
        except Exception as exc:
            last_empty_source = f"{label}: {exc}"

    return pd.DataFrame(), last_empty_source

def render_publication_cards(posts: pd.DataFrame, tickers: list[str] | None = None) -> None:
    """Показывает карточки публикаций, повлиявших на дискуссионное событие
    
    Args:
        posts: таблица или список публикаций
        tickers: набор биржевых идентификаторов финансовых инструментов
    
    Returns:
        None
    """
    if posts is None or pd.DataFrame(posts).empty:
        st.info("Для выбранного шага дискретизации не найдено публикаций. Проверьте, что анализ шага дискретизации был запущен после последней инициализации.")
        return

    original_df = pd.DataFrame(posts).copy()
    df = original_df.copy()
    requested_tickers = sorted({t for t in (_clean_ticker_token(x) for x in (tickers or [])) if t})
    available = sorted(set().union(*[_post_mentions_tickers(row) for _, row in df.iterrows()])) if not df.empty else []
    st.markdown("#### Публикации, которые объясняют аналитические события")
    st.caption("Список фильтра содержит тикеры, найденные в публикациях текущего шага дискретизации и выбранного исторического контекста")
    if requested_tickers and not set(requested_tickers).issubset(set(available)):
        missing = ", ".join(sorted(set(requested_tickers) - set(available)))
        if missing:
            st.caption(f"Тикеры без публикаций в текущем шаге дискретизации не добавлены в фильтр: {missing}")

    filter_options = ["Все публикации шага"] + available
    selected = st.selectbox("Тикер для фильтра публикаций", filter_options, index=0, key="publication_ticker_filter")

    if selected != "Все публикации шага":
        strict = df[df.apply(lambda r: selected in _post_mentions_tickers(r), axis=1)].copy()
        if strict.empty:
            soft = df[df.apply(lambda r: _soft_text_has_ticker(r, selected), axis=1)].copy()
            if not soft.empty:
                st.warning(f"Для {selected} не нашлось сохранённой тикерной разметки публикаций, но найдено {len(soft)} публикаций по текстовому упоминанию тикера.")
                df = soft
            else:
                st.info(f"Для {selected} в текущем шаге дискретизации не найдено публикаций с подтверждённой тикерной привязкой")
                return
        else:
            df = strict

    if "impact_score" in df.columns:
        df["impact_score"] = pd.to_numeric(df["impact_score"], errors="coerce")
        df = df.sort_values("impact_score", ascending=False)
    df = df.head(12)
    if df.empty:
        st.info("После фильтрации публикаций не осталось.")
        return

    topic_names_for_posts = topic_name_map(c, df)
    cols = st.columns(2)
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 2]:
            raw_text = str(row.get("text", row.get("публикация", "")))
            text = html_escape(raw_text[:1200], quote=False)
            author = _display_author(row)
            row_tickers = sorted(_post_mentions_tickers(row))
            tickers_text = ", ".join(row_tickers) if row_tickers else "тикер не определён"
            sent = fmt(row.get("sentiment", row.get("тональность", None)), 3)
            raw_author = str(row.get("author", row.get("Author", ""))).strip()
            service_author = bool(is_service_author(raw_author)) or author.startswith("автор не определён") or author.startswith("автор удалён") or author.startswith("AutoModerator")
            infl_value = 0.0 if service_author else row.get("influence", row.get("авторитетность", None))
            infl = fmt(infl_value, 3)
            impact = fmt(row.get("impact_score", None), 3)
            author_note = " · графовая авторитетность не оценивается" if service_author else ""
            topic = row.get("topic", row.get("тема", "-"))
            ts = row.get("timestamp", row.get("время", ""))
            source_period = row.get("source_period", "")
            source_text = f" · {source_period}" if str(source_period).strip() else ""
            card_html = (
                '<div class="post-card">'
                f'<div class="post-meta">{html_escape(str(ts), quote=False)}{html_escape(str(source_text), quote=False)} · автор: <b>{html_escape(str(author), quote=False)}</b>{html_escape(str(author_note), quote=False)} · {html_escape(topic_display_name(_topic_id(topic), topic_names_for_posts), quote=False)}</div>'
                f'<div class="post-meta">тикеры: <b>{html_escape(str(tickers_text), quote=False)}</b></div>'
                f'<div class="post-text dmi-publication-text">{text}</div>'
                f'<div class="post-stats">тональность: <b>{html_escape(str(sent), quote=False)}</b> · графовая авторитетность: <b>{html_escape(str(infl), quote=False)}</b> · вклад: <b>{html_escape(str(impact), quote=False)}</b></div>'
                '</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)
