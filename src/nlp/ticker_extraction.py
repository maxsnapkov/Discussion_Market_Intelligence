"""Извлечение финансовых инструментов и привязка тикеров к ветвям обсуждения"""
from __future__ import annotations


from ..pipeline.common import *
from ..pipeline.configuration import AppConfig, _emit_progress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_ingestion.preparation import parse_ticker_mentions
    from ..indicators.discussion_indicators import DISCUSSION_LEVEL_TICKER


_TICKER_RE = re.compile(r"(?<![A-Za-z])\$?([A-Z]{1,5})(?![A-Za-z])")
_CASHTAG_RE = re.compile(r"(?<![A-Za-z0-9])\$([A-Za-z]{1,5})(?![A-Za-z0-9])")
_STOP_TICKERS = {
    # общие английские и финансово-дискуссионные слова часто похожи на тикеры
    # они полезны как слова текста, но не должны автоматически становиться тикерами
    # принимаем их только если NER или сопоставление сущностей явно связывает слово
    # с названием компании
    "A", "I", "THE", "AND", "OR", "CEO", "USA", "USD", "AI", "IT", "ON", "IN", "TO", "FOR",
    "DD", "YOLO", "IMO", "LOL", "CFO", "SEC", "ETF", "IPO", "GDP", "CPI", "FOMC", "EPS",
    "WSB", "MORE", "PUMP", "EGGS", "ID", "YOY", "ATH", "ATM", "ITM", "OTM", "FD", "FDS",
    "CALL", "CALLS", "PUT", "PUTS", "BUY", "SELL", "HOLD", "MOON", "GAIN", "LOSS", "NEWS",
    "ER", "EPS", "PE", "IV", "OI", "TA", "CEO", "CFO", "IMO", "IMHO", "LOL", "LMAO",
    "YOU", "YOUR", "YOURS", "WHEN", "EDIT", "MAN", "BE", "DO", "ME", "MY", "WE", "US",
    "HE", "SHE", "HIM", "HER", "THEY", "THEM", "THIS", "THAT", "WHAT", "WHO", "WHY", "HOW",
}


# следующие тикеры являются биржевыми символами, но крайне неоднозначны в пользовательском тексте
# принимаем их только при сильном подтверждении: cashtag, заданный алиас или точное название компании
# это защищает от ложных совпадений вроде 3 PM
# и от превращения обычных фраз в события Philip Morris или Track Group
_AMBIGUOUS_TICKERS_REQUIRE_STRONG_EVIDENCE = {
    "PM", "ON", "ALL", "ARE", "CAN", "BIG", "LOW", "REAL", "OPEN", "PLAY", "LOVE", "RH", "NOW", "NEXT", "HAS", "EVER", "VERY", "TRCK",
    "CD", "HQ", "PC", "B", "BE", "RUN", "EDIT", "WHEN", "YOU", "MAN"
}
_STRONG_TICKER_EVIDENCE = {"cashtag", "alias", "company_name_indexed", "gliner_entity_symbol", "gliner_entity_name"}

_FALLBACK_TICKERS = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.", "GOOG": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.", "META": "Meta Platforms Inc.", "FB": "Meta Platforms Inc.", "TSLA": "Tesla Inc.", "NVDA": "NVIDIA Corporation",
    "AMD": "Advanced Micro Devices Inc.", "INTC": "Intel Corporation", "NFLX": "Netflix Inc.", "DIS": "Walt Disney Co.",
    "JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corporation", "WFC": "Wells Fargo & Company",
    "C": "Citigroup Inc.", "GS": "Goldman Sachs Group Inc.", "MS": "Morgan Stanley", "V": "Visa Inc.",
    "MA": "Mastercard Incorporated", "PYPL": "PayPal Holdings Inc.", "COIN": "Coinbase Global Inc.",
    "GME": "GameStop Corp.", "AMC": "AMC Entertainment Holdings Inc.", "BB": "BlackBerry Limited", "NOK": "Nokia Oyj",
    "PLTR": "Palantir Technologies Inc.", "SNAP": "Snap Inc.", "UBER": "Uber Technologies Inc.", "LYFT": "Lyft Inc.",
    "SPY": "SPDR S&P 500 ETF Trust", "QQQ": "Invesco QQQ Trust", "IWM": "iShares Russell 2000 ETF",
}
_NAME_ALIASES = {
    "apple": "AAPL", "iphone": "AAPL", "microsoft": "MSFT", "tesla": "TSLA", "nvidia": "NVDA",
    "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "facebook": "FB", "meta platforms": "META",
    "netflix": "NFLX", "disney": "DIS", "gamestop": "GME", "game stop": "GME", "amc": "AMC",
    "palantir": "PLTR", "coinbase": "COIN", "blackberry": "BB", "nokia": "NOK",
}

# приоритетные meme-stock тикеры должны находиться по явным упоминаниям
# они не принимаются по слабым фрагментам названия компании
# но не должны исчезать из январских шагов 2021 года
# где пользователи часто пишут GME без cashtag
_WSB_PRIORITY_TICKERS = {"GME", "AMC", "BB", "NOK", "PLTR", "TSLA", "BBBY", "KOSS", "EXPR", "NAKD"}
_AMBIGUOUS_VALIDATED_ALLOWED = _WSB_PRIORITY_TICKERS | {"SPY", "QQQ", "IWM"}

# финансовый контекст помогает отличать реальные тикеры от обычных слов
# для приоритетных meme-stock это не жёсткий фильтр
# для слабых символов он снижает уверенность
# и подавляет ложные срабатывания вроде PM, ON и ALL в обычном тексте
_FINANCE_CONTEXT_RE = re.compile(
    r"\b(stock|stocks|share|shares|option|options|call|calls|put|puts|buy|bought|sell|sold|"
    r"hold|holding|held|short|shorts|squeeze|gamma|delta|iv|oi|strike|expiry|expiration|"
    r"price|market|premarket|pre-market|aftermarket|dividend|divs|earnings|er|shares?|"
    r"portfolio|position|positions|broker|trade|trades|trading|yolo|dd|tendies|moon|rocket|"
    r"diamond|hands|bagholder|gain|gains|loss|losses|halt|suspended)\b",
    re.IGNORECASE,
)
_ACTION_CONTEXT_RE = re.compile(r"\b(hold|buy|bought|shares?|short|squeeze|calls?|puts?|moon|rocket|gains?|loss|halt)\b|🚀|💎|🙌", re.IGNORECASE)
_MIN_ANALYTIC_TICKER_CONFIDENCE = 0.42


def _row_ticker_set_for_activity(row: pd.Series | dict[str, Any]) -> set[str]:
    """Извлекает множество тикеров из строки активности
    
    Args:
        row: строка таблицы с сообщением, инструментом или событием
    
    Returns:
        множество тикеров, связанных со строкой
    """
    if hasattr(row, "get"):
        participation = parse_ticker_mentions(row.get("ticker_participation_mentions", "[]"))
        out = {str(m.get("ticker", "")).upper().strip() for m in participation if _accept_ticker_candidate_for_analytics(m)}
        if out:
            return {x for x in out if x and x != DISCUSSION_LEVEL_TICKER}
    mentions = parse_ticker_mentions(row.get("ticker_mentions", "[]")) if hasattr(row, "get") else []
    out = {str(m.get("ticker", "")).upper().strip() for m in mentions if _accept_ticker_candidate_for_analytics(m)}
    if not out and hasattr(row, "get"):
        raw = str(row.get("discussion_tickers", row.get("tickers", "")) or "")
        if raw.strip().lower() in {"nan", "none", "null"}:
            raw = ""
        out |= {x.strip().upper().replace("$", "") for x in raw.split(",") if x.strip() and x.strip().lower() not in {"nan", "none", "null"}}
    return {x for x in out if x and x != DISCUSSION_LEVEL_TICKER}



def _first_present_value(row: pd.Series | dict[str, Any], names: list[str]) -> str:
    """Возвращает первое непустое значение из набора колонок строки
    
    Args:
        row: строка таблицы с сообщением, инструментом или событием
        names: набор допустимых имён или словарь названий
    
    Returns:
        первое непустое значение из указанных колонок
    """
    if not hasattr(row, "get"):
        return ""
    for name in names:
        value = row.get(name, "")
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return text
    return ""



def _normalize_reddit_ref_id(value: Any) -> str:
    """Нормализует идентификатор сообщения или родительской записи
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        нормализованный идентификатор сообщения или пустая строка
    """
    raw = "" if value is None else str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return ""
    raw = raw.replace("\\", "/").strip()
    raw = raw.split("?")[0].split("#")[0]
    raw = raw.rstrip("/")
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    raw = raw.strip()
    lower = raw.lower()
    for prefix in ("t1_", "t3_", "t5_", "comment_", "post_", "submission_"):
        if lower.startswith(prefix):
            raw = raw[len(prefix):]
            lower = raw.lower()
            break
    raw = re.sub(r"[^A-Za-z0-9_\-]", "", raw)
    return raw.lower()


def _row_thread_root_id(row: pd.Series | dict[str, Any], fallback_pos: int | None = None) -> str:
    """Определяет корневую ветвь обсуждения для строки
    
    Args:
        row: строка таблицы с сообщением, инструментом или событием
        fallback_pos: резервная позиция строки для построения идентификатора
    
    Returns:
        определённое значение: корневую ветвь обсуждения для строки
    """
    root = _normalize_reddit_ref_id(_first_present_value(row, ["root_id", "thread_id", "link_id", "submission_id", "post_ref_id", "post_id"]))
    if root:
        return root
    parent = _message_parent_id(row)
    if parent:
        return parent
    node = _message_node_id(row)
    if node:
        return node
    return f"__row_{int(fallback_pos)}" if fallback_pos is not None else ""


def _message_node_id(row: pd.Series | dict[str, Any]) -> str:
    """Возвращает идентификатор узла сообщения
    
    Args:
        row: строка таблицы с сообщением, инструментом или событием
    
    Returns:
        идентификатор узла сообщения
    """
    return _normalize_reddit_ref_id(_first_present_value(row, ["message_id", "comment_ref_id", "comment_id", "id", "post_ref_id", "post_id"]))


def _message_parent_id(row: pd.Series | dict[str, Any]) -> str:
    """Возвращает идентификатор родительского сообщения
    
    Args:
        row: строка таблицы с сообщением, инструментом или событием
    
    Returns:
        идентификатор родительского сообщения
    """
    return _normalize_reddit_ref_id(_first_present_value(row, ["parent_ref_id", "parent_id", "parent", "reply_to", "in_reply_to"]))


def _direct_ticker_mentions_from_row(row: pd.Series | dict[str, Any]) -> list[dict[str, Any]]:
    """Извлекает прямые тикерные упоминания из строки сообщения
    
    Args:
        row: строка таблицы с сообщением, инструментом или событием
    
    Returns:
        извлечённые значения: прямые тикерные упоминания из строки сообщения
    """
    mentions = parse_ticker_mentions(row.get("ticker_mentions", "[]")) if hasattr(row, "get") else []
    if not mentions and hasattr(row, "get"):
        raw_tickers = str(row.get("tickers", "") or "")
        if raw_tickers.strip().lower() in {"nan", "none", "null"}:
            raw_tickers = ""
        mentions = [
            {
                "ticker": ticker.strip().upper().replace("$", ""),
                "confidence": 0.55,
                "method": "legacy_list",
                "evidence": ticker.strip(),
                "context_score": 0.50,
                "is_ambiguous": False,
            }
            for ticker in raw_tickers.split(",")
            if ticker.strip() and ticker.strip().lower() not in {"nan", "none", "null"}
        ]
    return [m for m in mentions if _accept_ticker_candidate_for_analytics(m)]


def _ticker_participation_mentions(part: pd.DataFrame, inherit_decay: float = 0.72, max_depth: int = 24) -> list[list[dict[str, Any]]]:
    """Восстанавливает участие тикеров в ветвях обсуждения с затуханием наследования
    
    Args:
        part: фрагмент таблицы сообщений внутри шага дискретизации
        inherit_decay: коэффициент затухания наследования тикера по ветви обсуждения
        max_depth: максимальная глубина наследования по ветви обсуждения
    
    Returns:
        список прямых и унаследованных тикерных связей
    """
    data = pd.DataFrame(part).reset_index(drop=True)
    if data.empty:
        return []
    ids: list[str] = []
    parent_ids: list[str] = []
    root_ids: list[str] = []
    id_to_pos: dict[str, int] = {}
    for pos, row in data.iterrows():
        node_id = _message_node_id(row)
        if not node_id:
            node_id = f"__row_{pos}"
        root_id = _row_thread_root_id(row, int(pos))
        ids.append(node_id)
        root_ids.append(root_id)
        comment_aliases: set[str] = set()
        post_aliases: set[str] = set()
        for alias_key in ["message_id", "comment_ref_id", "comment_id", "id"]:
            try:
                alias = _normalize_reddit_ref_id(row.get(alias_key, ""))
            except Exception:
                alias = ""
            if alias:
                comment_aliases.add(alias)
        for alias_key in ["post_ref_id", "post_id", "link_id", "submission_id"]:
            try:
                alias = _normalize_reddit_ref_id(row.get(alias_key, ""))
            except Exception:
                alias = ""
            if alias:
                post_aliases.add(alias)
        aliases = comment_aliases if comment_aliases else post_aliases
        if node_id:
            aliases.add(node_id)
        for alias in aliases:
            if alias:
                id_to_pos.setdefault(alias, int(pos))
        parent_ids.append(_message_parent_id(row))

    direct: list[list[dict[str, Any]]] = [_direct_ticker_mentions_from_row(row) for _, row in data.iterrows()]
    direct_by_ticker: list[dict[str, dict[str, Any]]] = []
    for items in direct:
        slot: dict[str, dict[str, Any]] = {}
        for item in items:
            ticker = str(item.get("ticker", "")).upper().strip()
            if ticker and (ticker not in slot or float(item.get("confidence", 0.0)) > float(slot[ticker].get("confidence", 0.0))):
                rec = dict(item)
                rec["participation_role"] = "direct"
                rec["inherit_distance"] = 0
                rec["inherited_from"] = ""
                rec["thread_root_id"] = root_ids[len(direct_by_ticker)] if len(direct_by_ticker) < len(root_ids) else ""
                slot[ticker] = rec
        direct_by_ticker.append(slot)

    cache: dict[int, dict[str, dict[str, Any]]] = {}

    def inherited_for(pos: int, trail: set[int] | None = None) -> dict[str, dict[str, Any]]:
        """Рассчитывает унаследованные тикеры для сообщения по цепочке родителей
        
        Args:
            pos: позиция строки или элемента
            trail: накопленная цепочка ключей или диагностических элементов
        
        Returns:
            список унаследованных тикерных связей для сообщения
        """
        if pos in cache:
            return cache[pos]
        trail = set() if trail is None else set(trail)
        if pos in trail:
            cache[pos] = {}
            return {}
        trail.add(pos)
        parent = parent_ids[pos]
        parent_pos = id_to_pos.get(parent)
        if parent_pos is None or parent_pos == pos:
            cache[pos] = {}
            return {}
        inherited: dict[str, dict[str, Any]] = {}
        for ticker, item in direct_by_ticker[parent_pos].items():
            rec = dict(item)
            rec["participation_role"] = "thread_descendant"
            rec["inherit_distance"] = 1
            rec["inherited_from"] = ids[parent_pos]
            rec["thread_root_id"] = root_ids[parent_pos] or rec.get("thread_root_id", "")
            inherited[ticker] = rec
        for ticker, item in inherited_for(parent_pos, trail).items():
            dist = int(item.get("inherit_distance", 1) or 1) + 1
            if dist > int(max_depth):
                continue
            rec = dict(item)
            rec["inherit_distance"] = dist
            rec["thread_root_id"] = root_ids[pos] or rec.get("thread_root_id", "")
            inherited[ticker] = rec
        cache[pos] = inherited
        return inherited

    output: list[list[dict[str, Any]]] = []
    for pos in range(len(data)):
        combined: dict[str, dict[str, Any]] = {ticker: dict(item) for ticker, item in direct_by_ticker[pos].items()}
        for ticker, item in inherited_for(pos).items():
            if ticker in combined:
                continue
            dist = int(item.get("inherit_distance", 1) or 1)
            decay = float(max(0.08, min(1.0, inherit_decay)) ** max(1, dist))
            rec = dict(item)
            rec["confidence"] = float(max(0.12, min(1.0, float(rec.get("confidence", 0.0) or 0.0) * decay)))
            rec["context_score"] = float(max(0.10, min(1.0, float(rec.get("context_score", 0.0) or 0.0) * decay)))
            rec["method"] = "thread_descendant"
            rec["evidence"] = str(rec.get("evidence", ticker))[:90]
            rec["is_ambiguous"] = bool(rec.get("is_ambiguous", False))
            rec["thread_root_id"] = root_ids[pos] or rec.get("thread_root_id", "")
            combined[ticker] = rec
        output.append(list(combined.values()))
    return output


def attach_ticker_thread_participation(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет к сообщениям прямые и унаследованные связи с финансовыми инструментами
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        обновлённая структура с добавленными полями: к сообщениям прямые и унаследованные связи с финансовыми инструментами
    """
    data = pd.DataFrame(df).copy()
    if data.empty:
        data["direct_tickers"] = ""
        data["inherited_tickers"] = ""
        data["discussion_tickers"] = ""
        data["ticker_participation_mentions"] = "[]"
        return data
    payloads = _ticker_participation_mentions(data)
    direct_values: list[str] = []
    inherited_values: list[str] = []
    discussion_values: list[str] = []
    serialized: list[str] = []
    for items in payloads:
        direct = sorted({str(x.get("ticker", "")).upper() for x in items if str(x.get("participation_role", "direct")) == "direct" and x.get("ticker")})
        inherited = sorted({str(x.get("ticker", "")).upper() for x in items if str(x.get("participation_role", "direct")) != "direct" and x.get("ticker")})
        all_tickers = sorted(set(direct) | set(inherited))
        direct_values.append(",".join(direct))
        inherited_values.append(",".join(inherited))
        discussion_values.append(",".join(all_tickers))
        serialized.append(json.dumps(items, ensure_ascii=False))
    data["direct_tickers"] = direct_values
    data["inherited_tickers"] = inherited_values
    data["discussion_tickers"] = discussion_values
    data["ticker_participation_mentions"] = serialized
    return data


def _ticker_activity_fields(info: dict[str, Any]) -> dict[str, Any]:
    """Преобразует сведения о тикере в поля активности для таблицы результатов
    
    Args:
        info: словарь признаков финансового инструмента
    
    Returns:
        словарь полей активности финансового инструмента
    """
    return {
        "ticker_post_count": int(float(info.get("post_count", 0) or 0)),
        "ticker_unique_authors": int(float(info.get("unique_authors", 0) or 0)),
        "ticker_direct_post_count": int(float(info.get("direct_post_count", 0) or 0)),
        "ticker_inherited_post_count": int(float(info.get("inherited_post_count", 0) or 0)),
        "ticker_root_thread_count": int(float(info.get("root_thread_count", 0) or 0)),
        "ticker_participation_mode": str(info.get("participation_mode", "direct_mentions")),
        "ticker_context_direct_post_count": int(float(info.get("context_direct_post_count", 0) or 0)),
        "ticker_context_inherited_post_count": int(float(info.get("context_inherited_post_count", 0) or 0)),
        "ticker_context_root_thread_count": int(float(info.get("context_root_thread_count", 0) or 0)),
        "ticker_activity_rate": float(info.get("activity_rate", 0.0) or 0.0),
        "ticker_author_rate": float(info.get("author_rate", 0.0) or 0.0),
        "ticker_activity_lift": info.get("activity_lift"),
        "ticker_activity_zscore": info.get("activity_zscore"),
        "ticker_activity_robust_zscore": info.get("activity_robust_zscore"),
        "ticker_baseline_posts": info.get("baseline_posts"),
        "ticker_baseline_median_posts": info.get("baseline_median_posts"),
        "ticker_support_score": float(info.get("support_score", 0.0) or 0.0),
        "ticker_activity_anomaly_score": float(info.get("activity_anomaly_score", 0.0) or 0.0),
        "ticker_attention_score": float(info.get("attention_score", 0.0) or 0.0),
        "ticker_confidence_avg": float(info.get("confidence_avg", 0.0) or 0.0),
        "ticker_context_score_avg": float(info.get("context_score_avg", 0.0) or 0.0),
        "ticker_current_post_count": int(float(info.get("current_post_count", info.get("post_count", 0)) or 0)),
        "ticker_current_unique_authors": int(float(info.get("current_unique_authors", info.get("unique_authors", 0)) or 0)),
        "ticker_context_post_count": int(float(info.get("context_post_count", info.get("post_count", 0)) or 0)),
        "ticker_context_unique_authors": int(float(info.get("context_unique_authors", info.get("unique_authors", 0)) or 0)),
        "ticker_context_share": float(info.get("context_share", info.get("share", 0.0)) or 0.0),
        "ticker_historical_support_score": info.get("historical_support_score"),
        "ticker_posts_pvalue": info.get("p_posts"),
        "ticker_authors_pvalue": info.get("p_authors"),
        "ticker_thread_pvalue": info.get("p_thread"),
        "ticker_global_posts_pvalue": info.get("p_global_posts"),
        "ticker_global_authors_pvalue": info.get("p_global_authors"),
        "ticker_global_thread_pvalue": info.get("p_global_thread"),
        "ticker_global_activity_rate_pvalue": info.get("p_global_activity_rate"),
        "ticker_observed_support_pvalue": info.get("p_observed_support"),
        "ticker_sentiment_pvalue": info.get("p_sentiment"),
        "ticker_influence_pvalue": info.get("p_influence"),
        "ticker_growth_pvalue": info.get("p_growth"),
        "ticker_current_prominence_pvalue": info.get("p_current_prominence"),
        "ticker_context_prominence_pvalue": info.get("p_context_prominence"),
        "ticker_presence_pvalue": info.get("p_presence"),
        "ticker_activity_pvalue": info.get("p_activity"),
        "ticker_text_pvalue": info.get("p_text"),
        "ticker_source_scope": str(info.get("source_scope", "current_window")),
        "ticker_ner_methods": json.dumps(info.get("methods", {}), ensure_ascii=False) if isinstance(info.get("methods", {}), dict) else str(info.get("methods", "")),
    }



def _normalize_company_name(value: Any) -> str:
    """Нормализует название компании для словарного сопоставления
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        нормализованное название компании
    """
    text = re.sub(r"[^a-z0-9 ]+", " ", str(value).lower())
    suffixes = [
        " inc ", " incorporated ", " corporation ", " corp ", " company ", " co ", " ltd ", " limited ",
        " plc ", " class a ", " class b ", " common stock ", " ordinary shares ", " adr ", " ads ",
        " holdings ", " holding ", " group ", " the ",
    ]
    text = f" {text} "
    for suf in suffixes:
        text = text.replace(suf, " ")
    return re.sub(r"\s+", " ", text).strip()


def _download_text(url: str, timeout: int = 20) -> str:
    """Загружает текстовый ресурс по URL для обновления справочников
    
    Args:
        url: URL внешнего ресурса
        timeout: таймаут сетевого запроса или внешнего процесса в секундах
    
    Returns:
        загруженный результат: текстовый ресурс по URL для обновления справочников
    """
    req = urllib.request.Request(url, headers={"User-Agent": "discussion-market-intelligence research app contact@example.com"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def load_ticker_universe(cfg: AppConfig, refresh: bool = False) -> pd.DataFrame:
    """Загружает справочник тикеров из кэша, внешних источников или встроенного набора
    
    Args:
        cfg: конфигурация проекта или приложения
        refresh: признак принудительного обновления внешних справочников
    
    Returns:
        загруженный результат: справочник тикеров из кэша, внешних источников или встроенного набора
    """
    market_dir = cfg.data_dir / "market"
    market_dir.mkdir(parents=True, exist_ok=True)
    cache = market_dir / "ticker_universe.csv"
    cached_df: pd.DataFrame | None = None
    if cache.exists():
        try:
            cached_df = pd.read_csv(cache).fillna("")
            if not refresh:
                return cached_df
        except Exception:
            cached_df = None
    rows: list[dict[str, Any]] = []

    # справочник SEC содержит тикер, название компании и биржу
    try:
        txt = _download_text("https://www.sec.gov/files/company_tickers_exchange.json")
        obj = json.loads(txt)
        fields = obj.get("fields", [])
        for item in obj.get("data", []):
            rec = dict(zip(fields, item))
            ticker = str(rec.get("ticker", "")).upper().strip()
            name = str(rec.get("name", "")).strip()
            exchange = str(rec.get("exchange", "")).strip()
            if ticker and name:
                rows.append({"ticker": ticker, "name": name, "exchange": exchange, "source": "sec"})
    except Exception:
        pass

    # каталог NASDAQ Trader покрывает NASDAQ и другие листинговые инструменты
    for url, source, sym_col, name_col in [
        ("https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt", "nasdaq", "Symbol", "Security Name"),
        ("https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt", "nasdaq_other", "ACT Symbol", "Security Name"),
    ]:
        try:
            from io import StringIO
            txt = _download_text(url)
            table = pd.read_csv(StringIO(txt), sep="|")
            for _, rec in table.iterrows():
                ticker = str(rec.get(sym_col, "")).upper().strip()
                name = str(rec.get(name_col, "")).strip()
                test_issue = str(rec.get("Test Issue", "N")).upper().strip()
                if ticker and name and test_issue != "Y" and "File Creation Time" not in ticker:
                    rows.append({"ticker": ticker, "name": name, "exchange": str(rec.get("Exchange", "")).strip(), "source": source})
        except Exception:
            pass

    if not rows:
        # не даём неудачному обновлению без интернета перезаписать полный локальный кэш
        # маленьким встроенным резервным справочником
        if cached_df is not None and not cached_df.empty:
            return cached_df
        rows = [{"ticker": t, "name": n, "exchange": "", "source": "fallback"} for t, n in _FALLBACK_TICKERS.items()]
    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "name"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["name_norm"] = df["name"].map(_normalize_company_name)
    df = df[df["ticker"].str.len().between(1, 6)].reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df



_GLINER_MODEL_CACHE: Any = None
_GLINER_LOAD_FAILED = False


def _get_gliner_model(cfg: AppConfig | None = None):
    """Загружает модель GLiNER для распознавания финансовых сущностей
    
    Args:
        cfg: конфигурация проекта или приложения
    
    Returns:
        загруженный результат: модель GLiNER для распознавания финансовых сущностей
    """
    global _GLINER_MODEL_CACHE, _GLINER_LOAD_FAILED
    if _GLINER_LOAD_FAILED:
        return None
    if _GLINER_MODEL_CACHE is not None:
        return _GLINER_MODEL_CACHE
    try:
        from gliner import GLiNER  # type: ignore
        model_name = "urchade/gliner_small-v2.1"
        if cfg is not None:
            model_name = str(cfg.get("nlp", "gliner_model", default=model_name))
        _GLINER_MODEL_CACHE = GLiNER.from_pretrained(model_name)
        return _GLINER_MODEL_CACHE
    except Exception:
        _GLINER_LOAD_FAILED = True
        return None


_TICKER_INDEX_CACHE: dict[int, dict[str, Any]] = {}
_TICKER_NAME_TOKEN_STOP = {
    "inc", "corp", "corporation", "company", "co", "ltd", "limited", "plc", "group", "holdings",
    "holding", "class", "common", "stock", "shares", "ordinary", "the", "and", "trust", "fund",
    "international", "technologies", "technology", "systems", "industries", "financial", "services",
}


def _ticker_index(universe: pd.DataFrame) -> dict[str, Any]:
    """Строит индекс тикеров, компаний и алиасов для быстрого поиска
    
    Args:
        universe: таблица справочника тикеров и компаний
    
    Returns:
        индекс для поиска тикеров, компаний и алиасов
    """
    fallback_universe = pd.DataFrame([{"ticker": t, "name": n, "name_norm": _normalize_company_name(n)} for t, n in _FALLBACK_TICKERS.items()])
    if universe is None or universe.empty:
        universe = fallback_universe
    else:
        universe = universe.copy()
        if "name_norm" not in universe.columns:
            universe["name_norm"] = universe["name"].map(_normalize_company_name)
        present = set(universe["ticker"].astype(str).str.upper().str.strip()) if "ticker" in universe.columns else set()
        extra = fallback_universe[~fallback_universe["ticker"].isin(present)]
        if not extra.empty:
            universe = pd.concat([universe, extra], ignore_index=True)
    if "name_norm" not in universe.columns:
        universe = universe.copy()
        universe["name_norm"] = universe["name"].map(_normalize_company_name)
    key = id(universe)
    cached = _TICKER_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    valid = set(universe["ticker"].astype(str).str.upper().str.strip())
    exact_name: dict[str, str] = {}
    token_index: dict[str, list[tuple[str, str]]] = {}
    for _, r in universe.iterrows():
        ticker = str(r.get("ticker", "")).upper().strip()
        name_norm = str(r.get("name_norm", "")).strip()
        if not ticker or not name_norm or ticker in _STOP_TICKERS:
            continue
        exact_name.setdefault(name_norm, ticker)
        tokens = [t for t in name_norm.split() if len(t) >= 4 and t not in _TICKER_NAME_TOKEN_STOP]
        # добавляем только несколько характерных токенов на компанию
        # так списки кандидатов остаются ограниченными и не требуют полного перебора справочника
        for tok in tokens[:3]:
            token_index.setdefault(tok, []).append((name_norm, ticker))

    alias_map = {str(k).lower(): str(v).upper() for k, v in _NAME_ALIASES.items()}
    # встроенные резервные названия точные и дешёвые для проверки
    for ticker, name in _FALLBACK_TICKERS.items():
        norm = _normalize_company_name(name)
        if norm:
            alias_map.setdefault(norm, ticker)

    out = {"valid": valid, "alias_map": alias_map, "exact_name": exact_name, "token_index": token_index}
    _TICKER_INDEX_CACHE[key] = out
    return out


def _ticker_context_score(raw: str, ticker: str, method: str, evidence: str = "") -> float:
    """Оценивает силу финансового контекста вокруг найденного кандидата
    
    Args:
        raw: исходное значение или исходная таблица до нормализации
        ticker: биржевой идентификатор финансового инструмента
        method: метод обнаружения кандидата
        evidence: текстовое или структурное подтверждение кандидата
    
    Returns:
        оценка финансового контекста кандидата
    """
    text = str(raw or "")
    ticker = str(ticker or "").upper().replace("$", "").strip()
    if not ticker:
        return 0.0
    if method == "cashtag":
        return 1.0
    if ticker in _WSB_PRIORITY_TICKERS:
        # в финансовых пользовательских корпусах эти символы значимы даже без cashtag
        # достаточно явного токена
        return 0.95
    score = 0.0
    if _FINANCE_CONTEXT_RE.search(text):
        score += 0.45
    if _ACTION_CONTEXT_RE.search(text):
        score += 0.20
    if re.search(rf"(?<![A-Za-z0-9])\${re.escape(ticker)}(?![A-Za-z0-9])", text):
        score += 0.35
    ev = str(evidence or "").strip().lower()
    if ev and len(ev.split()) >= 2:
        score += 0.25
    if method in {"alias", "company_name_indexed", "gliner_entity_symbol", "gliner_entity_name", "gliner_entity_alias", "gliner_entity_company"}:
        score += 0.30
    return float(max(0.0, min(1.0, score)))


def _ticker_candidate_record(ticker: str, confidence: float, method: str, evidence: str, context_score: float, is_ambiguous: bool) -> dict[str, Any]:
    """Создаёт унифицированную запись кандидата финансового инструмента
    
    Args:
        ticker: биржевой идентификатор финансового инструмента
        confidence: уверенность обнаружения кандидата
        method: метод обнаружения кандидата
        evidence: текстовое или структурное подтверждение кандидата
        context_score: оценка финансового контекста вокруг кандидата
        is_ambiguous: признак неоднозначного биржевого обозначения
    
    Returns:
        словарь с унифицированным описанием кандидата
    """
    return {
        "ticker": str(ticker).upper().strip(),
        "confidence": float(max(0.0, min(1.0, confidence))),
        "method": str(method),
        "evidence": str(evidence)[:80],
        "context_score": float(max(0.0, min(1.0, context_score))),
        "is_ambiguous": bool(is_ambiguous),
    }


def _collapse_derivative_ticker_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Убирает производные тикерные кандидаты при наличии базового инструмента
    
    Args:
        items: список элементов для обработки
    
    Returns:
        список кандидатов без лишних производных дублей
    """
    tickers = {str(x.get("ticker", "")).upper() for x in items}
    out = []
    for item in items:
        ticker = str(item.get("ticker", "")).upper()
        base = re.split(r"[-.]", ticker, maxsplit=1)[0]
        if base and base in tickers and base != ticker:
            # производный символ сохраняем только при явном cashtag или биржевом токене
            if str(item.get("method", "")) not in {"cashtag", "validated_symbol", "gliner_entity_symbol"}:
                continue
        out.append(item)
    return out


def _accept_ticker_candidate_for_analytics(candidate: dict[str, Any]) -> bool:
    """Проверяет, достаточно ли надёжен кандидат для аналитики
    
    Args:
        candidate: кандидат финансового инструмента
    
    Returns:
        bool: признак допуска кандидата в аналитику
    """
    ticker = str(candidate.get("ticker", "")).upper().strip()
    method = str(candidate.get("method", ""))
    confidence = float(candidate.get("confidence", 0.0) or 0.0)
    context = float(candidate.get("context_score", 0.0) or 0.0)
    if not ticker or ticker == DISCUSSION_LEVEL_TICKER:
        return False
    if method == "cashtag" or method.startswith("gliner_") or method in {"alias", "company_name_indexed"}:
        return confidence >= 0.50
    if ticker in _WSB_PRIORITY_TICKERS:
        return confidence >= 0.55
    if ticker in _AMBIGUOUS_TICKERS_REQUIRE_STRONG_EVIDENCE:
        return False
    return confidence >= _MIN_ANALYTIC_TICKER_CONFIDENCE and context >= 0.25


def _extract_ticker_candidates_fast(
    text: str,
    universe: pd.DataFrame | None = None,
    max_candidates: int = 8,
    ticker_index: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Быстро извлекает тикерных кандидатов регулярными выражениями и словарём
    
    Args:
        text: текст сообщения или текстовое значение
        universe: таблица справочника тикеров и компаний
        max_candidates: максимальное число кандидатов для возврата
        ticker_index: индекс тикеров, алиасов и названий компаний
    
    Returns:
        список кандидатов финансовых инструментов из быстрых правил
    """
    raw = str(text or "")
    universe = universe if universe is not None and not universe.empty else pd.DataFrame(
        [{"ticker": t, "name": n, "name_norm": _normalize_company_name(n)} for t, n in _FALLBACK_TICKERS.items()]
    )
    idx = ticker_index if ticker_index is not None else _ticker_index(universe)
    valid = idx["valid"]
    candidates: dict[str, dict[str, Any]] = {}

    def add(ticker: str, base_score: float, method: str, evidence: str):
        """Добавляет тикерного кандидата во временный список без дублирования
        
        Args:
            ticker: биржевой идентификатор финансового инструмента
            base_score: базовое значение скора до дополнительных поправок
            method: метод обнаружения кандидата
            evidence: текстовое или структурное подтверждение кандидата
        
        Returns:
            None: функция добавляет кандидата во временный список
        """
        ticker = str(ticker).upper().strip().replace("$", "")
        if not ticker or ticker in _STOP_TICKERS or ticker not in valid:
            return
        is_ambiguous = ticker in _AMBIGUOUS_TICKERS_REQUIRE_STRONG_EVIDENCE
        context_score = _ticker_context_score(raw, ticker, method, evidence)
        score = float(base_score)
        if method == "validated_symbol":
            if ticker in _WSB_PRIORITY_TICKERS:
                score = max(score, 0.92)
            elif is_ambiguous and ticker not in _AMBIGUOUS_VALIDATED_ALLOWED:
                # очень слабые подтверждения не добавляем в список кандидатов
                # NER или алиасы компаний всё ещё могут восстановить реальные случаи PM, ON и BIG
                return
            else:
                # обычным заглавным символам нужен финансовый контекст
                # без него они остаются с низкой уверенностью и не идут в аналитику
                score = min(score, 0.55 + 0.35 * context_score)
        elif method in {"alias", "company_name_indexed"}:
            score = max(score, 0.78 if method == "alias" else 0.70)
        elif method == "cashtag":
            score = 0.99
        # однобуквенные символы почти всегда неоднозначны без сильного подтверждения
        if len(ticker) == 1 and method not in {"cashtag", "alias", "company_name_indexed", "gliner_entity_symbol"}:
            return
        score = float(max(0.0, min(1.0, score)))
        rec = _ticker_candidate_record(ticker, score, method, evidence, context_score, is_ambiguous)
        old = candidates.get(ticker)
        if old is None or float(rec["confidence"]) > float(old.get("confidence", 0.0)):
            candidates[ticker] = rec

    # явные cashtag имеют высокую уверенность
    for t in _CASHTAG_RE.findall(raw):
        add(t, 0.99, "cashtag", "$" + t.upper())

    # заглавные токены проверяются по справочнику тикеров
    for t in _TICKER_RE.findall(raw):
        t = t.upper()
        if len(t) >= 2:
            add(t, 0.82, "validated_symbol", t)

    # алиасы компаний проверяются через индекс точных фраз
    text_norm = _normalize_company_name(raw)
    padded = f" {text_norm} "
    for alias, ticker in idx["alias_map"].items():
        if len(alias) >= 3 and f" {alias} " in padded:
            add(ticker, 0.86, "alias", alias)
            if len(candidates) >= max_candidates:
                break

    # кандидаты названий компаний выбираются по токенам сообщения
    seen_names: set[tuple[str, str]] = set()
    for tok in set(text_norm.split()):
        if len(tok) < 4:
            continue
        for name_norm, ticker in idx["token_index"].get(tok, [])[:40]:
            key = (name_norm, ticker)
            if key in seen_names:
                continue
            seen_names.add(key)
            name_tokens = name_norm.split()
            if len(name_tokens) < 2 and name_norm not in idx["alias_map"]:
                continue
            if len(name_norm) >= 5 and f" {name_norm} " in padded:
                add(ticker, 0.72, "company_name_indexed", name_norm)
                if len(candidates) >= max_candidates:
                    break
        if len(candidates) >= max_candidates:
            break

    return _collapse_derivative_ticker_candidates(sorted(candidates.values(), key=lambda r: (-float(r["confidence"]), r["ticker"])))[:max_candidates]


def extract_ticker_candidates_many(
    texts: list[str],
    universe: pd.DataFrame | None = None,
    cfg: AppConfig | None = None,
    progress=None,
    stage: str = "tickers",
) -> list[list[dict[str, Any]]]:
    """Извлекает кандидатов финансовых инструментов для набора текстов
    
    Args:
        texts: список текстов сообщений
        universe: таблица справочника тикеров и компаний
        cfg: конфигурация проекта или приложения
        progress: callback для передачи статуса выполнения
        stage: название этапа вычислительного конвейера
    
    Returns:
        извлечённые значения: кандидатов финансовых инструментов для набора текстов
    """
    total = len(texts)
    batch_size = int(cfg.get("nlp", "ticker_batch_size", default=4096) if cfg else 4096)
    mode = str(cfg.get("nlp", "ticker_extraction", default="hybrid_fast") if cfg else "hybrid_fast").lower()
    max_candidates = int(cfg.get("nlp", "max_ticker_candidates", default=8) if cfg else 8)
    fast_index = _ticker_index(universe if universe is not None else pd.DataFrame())
    out: list[list[dict[str, Any]]] = []
    for i in range(0, total, max(1, batch_size)):
        part = texts[i:i + batch_size]
        out.extend([_extract_ticker_candidates_fast(x, universe=universe, max_candidates=max_candidates, ticker_index=fast_index) for x in part])
        _emit_progress(progress, f"Тикеры: быстрый локальный слой {min(i + batch_size, total)}/{total}", min(i + batch_size, total), total, stage=stage)

    if mode in {"gliner", "gliner_hybrid", "hybrid_gliner", "gliner_full", "ner_full"}:
        policy = str(cfg.get("nlp", "gliner_policy", default="missing_only") if cfg else "missing_only").lower()
        if mode in {"gliner_full", "ner_full"}:
            policy = "all"
        max_gliner = int(cfg.get("nlp", "gliner_max_messages", default=2000) if cfg else 2000)
        _emit_progress(progress, "Тикеры: загружаю GLiNER NER", 0, 1, stage="ticker_ner")
        model = _get_gliner_model(cfg)
        if model is None:
            _emit_progress(progress, "Тикеры: GLiNER недоступен, используется быстрый слой", 1, 1, stage="ticker_ner")
        else:
            if policy == "all":
                candidates_idx = list(range(total)) if max_gliner <= 0 else list(range(total))[:max_gliner]
            else:
                base = [i for i, xs in enumerate(out) if not xs]
                candidates_idx = base if max_gliner <= 0 else base[:max_gliner]
            _emit_progress(progress, f"Тикеры: GLiNER будет применён к {len(candidates_idx)} сообщениям", 0, max(1, len(candidates_idx)), stage="ticker_ner")
            for n, idx_i in enumerate(candidates_idx, start=1):
                current = {x["ticker"]: x for x in out[idx_i]}
                for item in _extract_gliner_entities(texts[idx_i], universe if universe is not None else pd.DataFrame(), cfg):
                    ticker = str(item.get("ticker", "")).upper()
                    if ticker and (ticker not in current or float(item.get("confidence", 0)) > float(current[ticker].get("confidence", 0))):
                        current[ticker] = item
                out[idx_i] = _collapse_derivative_ticker_candidates(sorted(current.values(), key=lambda r: (-float(r["confidence"]), r["ticker"])))[:max_candidates]
                if n % 10 == 0 or n == len(candidates_idx):
                    _emit_progress(progress, f"Тикеры: GLiNER NER {n}/{len(candidates_idx)}", n, max(1, len(candidates_idx)), stage="ticker_ner")
    return out

def _company_match_score(entity_norm: str, company_norm: str) -> float:
    """Оценивает близость распознанной организации к названию компании
    
    Args:
        entity_norm: нормализованное название распознанной сущности
        company_norm: нормализованное название компании из справочника
    
    Returns:
        оценка близости названий компаний
    """
    if not entity_norm or not company_norm:
        return 0.0
    if entity_norm == company_norm:
        return 1.0
    if f" {entity_norm} " in f" {company_norm} " or f" {company_norm} " in f" {entity_norm} ":
        return 0.92
    entity_tokens = set(entity_norm.split())
    company_tokens = set(company_norm.split())
    if entity_tokens and company_tokens:
        jaccard = len(entity_tokens & company_tokens) / max(1, len(entity_tokens | company_tokens))
    else:
        jaccard = 0.0
    seq = SequenceMatcher(None, entity_norm, company_norm).ratio()
    return float(max(jaccard, seq * 0.85))


def _resolve_entity_to_ticker(entity: str, universe: pd.DataFrame, min_score: float = 0.72) -> tuple[str | None, float, str]:
    """Сопоставляет распознанную сущность с тикером справочника
    
    Args:
        entity: распознанная именованная сущность
        universe: таблица справочника тикеров и компаний
        min_score: минимальная оценка сопоставления
    
    Returns:
        пара из тикера и оценки сопоставления
    """
    raw = str(entity or "").strip()
    if not raw:
        return None, 0.0, "empty"
    valid = set(universe["ticker"].astype(str).str.upper())
    upper = raw.upper().replace("$", "").strip()
    if upper in valid and upper not in _STOP_TICKERS:
        return upper, 0.98, "entity_symbol"
    norm = _normalize_company_name(raw)
    if norm in _NAME_ALIASES:
        return _NAME_ALIASES[norm], 0.88, "entity_alias"
    best_ticker, best_score, best_name = None, 0.0, ""
    # сначала проверяем точные нормализованные названия и алиасы
    for _, r in universe.iterrows():
        name_norm = str(r.get("name_norm", ""))
        if len(name_norm) < 3:
            continue
        score = _company_match_score(norm, name_norm)
        if score > best_score:
            best_ticker = str(r.get("ticker", "")).upper()
            best_score = score
            best_name = name_norm
    if best_ticker and best_score >= min_score:
        return best_ticker, float(min(0.86, best_score)), f"entity_company:{best_name}"
    return None, float(best_score), "unresolved_entity"


def _extract_gliner_entities(text: str, universe: pd.DataFrame, cfg: AppConfig | None = None) -> list[dict[str, Any]]:
    """Извлекает финансовые сущности моделью GLiNER
    
    Args:
        text: текст сообщения или текстовое значение
        universe: таблица справочника тикеров и компаний
        cfg: конфигурация проекта или приложения
    
    Returns:
        список кандидатов, найденных GLiNER
    """
    model = _get_gliner_model(cfg)
    if model is None:
        return []
    labels = ["public company", "company", "organization", "stock ticker", "financial instrument", "brand", "product"]
    try:
        # GLiNER возвращает спаны с текстом, меткой и оценкой в зависимости от версии
        entities = model.predict_entities(str(text or "")[:2000], labels, threshold=float((cfg.get("nlp", "gliner_threshold", default=0.35) if cfg else 0.35)))
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for ent in entities or []:
        entity_text = str(ent.get("text", ent.get("span", ""))).strip()
        if not entity_text:
            continue
        ticker, score, reason = _resolve_entity_to_ticker(entity_text, universe)
        if ticker:
            method = "gliner_" + reason
            ctx = _ticker_context_score(text, ticker, method, entity_text)
            out.append(_ticker_candidate_record(
                ticker,
                float(max(float(ent.get("score", 0.5)) * score, 0.55)),
                method,
                entity_text[:80],
                max(ctx, 0.75),
                ticker in _AMBIGUOUS_TICKERS_REQUIRE_STRONG_EVIDENCE,
            ))
    return out


def extract_ticker_candidates(
    text: str,
    universe: pd.DataFrame | None = None,
    max_candidates: int = 8,
    cfg: AppConfig | None = None,
) -> list[dict[str, Any]]:
    """Объединяет быстрый поиск и GLiNER для одного текста
    
    Args:
        text: текст сообщения или текстовое значение
        universe: таблица справочника тикеров и компаний
        max_candidates: максимальное число кандидатов для возврата
        cfg: конфигурация проекта или приложения
    
    Returns:
        объединённая структура: быстрый поиск и GLiNER для одного текста
    """
    universe = universe if universe is not None and not universe.empty else pd.DataFrame(
        [{"ticker": t, "name": n, "name_norm": _normalize_company_name(n)} for t, n in _FALLBACK_TICKERS.items()]
    )
    candidates = _extract_ticker_candidates_fast(text, universe=universe, max_candidates=max_candidates)
    mode = str(cfg.get("nlp", "ticker_extraction", default="hybrid_fast") if cfg else "hybrid_fast").lower()
    if mode in {"gliner", "gliner_hybrid", "hybrid_gliner", "gliner_full", "ner_full"}:
        current = {x["ticker"]: x for x in candidates}
        for item in _extract_gliner_entities(str(text or ""), universe, cfg):
            ticker = str(item.get("ticker", "")).upper()
            if ticker and (ticker not in current or float(item.get("confidence", 0)) > float(current[ticker].get("confidence", 0))):
                current[ticker] = item
        candidates = _collapse_derivative_ticker_candidates(sorted(current.values(), key=lambda r: (-float(r["confidence"]), r["ticker"])))[:max_candidates]
    return candidates

def extract_tickers(text: str, universe: pd.DataFrame | None = None) -> list[str]:
    """Возвращает список тикеров, принятых для аналитики
    
    Args:
        text: текст сообщения или текстовое значение
        universe: таблица справочника тикеров и компаний
    
    Returns:
        список тикеров, принятых для аналитики
    """
    return [c["ticker"] for c in extract_ticker_candidates(text, universe=universe)]
