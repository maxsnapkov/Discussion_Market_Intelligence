"""Расчёт тональности сообщений и кэширование результатов"""
from __future__ import annotations


from ..pipeline.common import *
from ..pipeline.configuration import AppConfig, _emit_progress, clean_text

_POS = {
    "good", "great", "buy", "buying", "bought", "bull", "bullish", "moon", "gain", "gains", "profit",
    "profits", "up", "beat", "beating", "growth", "strong", "positive", "hold", "holding", "hodl",
    "shares", "calls", "call", "rocket", "rockets", "squeeze", "long", "win", "wins", "legendary",
    "worth", "green", "tendies", "diamond", "hands", "undervalued", "breakout"
}
_NEG = {
    "bad", "sell", "selling", "sold", "bear", "bearish", "down", "loss", "losses", "crash", "fraud",
    "weak", "miss", "negative", "risk", "drop", "dump", "dumping", "red", "puts", "put", "overvalued"
}


def _market_phrase_sentiment(text: str) -> float:
    """Оценивает тональность по словарю финансовых выражений
    
    Args:
        text: текст сообщения или текстовое значение
    
    Returns:
        словарная оценка финансовой тональности
    """
    raw = str(text or "")
    low = raw.lower()
    score = 0.0
    positive_patterns = [
        r"hold", r"hodl", r"buy", r"bought", r"buying",
        r"moon", r"rocket", r"🚀", r"short squeeze", r"squeeze",
        r"beat(ing)? the short", r"diamond hands", r"tendies", r"gains?",
        r"worth more", r"wins? money", r"all[- ]?in", r"bullish",
    ]
    negative_patterns = [
        r"sell", r"sold", r"puts?", r"bearish", r"crash", r"dump",
        r"loss(es)?", r"overvalued", r"worthless",
    ]
    for pat in positive_patterns:
        if re.search(pat, low):
            score += 0.18
    for pat in negative_patterns:
        if re.search(pat, low):
            score -= 0.18
    # восклицания и повторяющиеся ракеты отражают интенсивность, а не направление тональности
    if "!" in raw and score != 0:
        score *= min(1.35, 1.0 + raw.count("!") * 0.04)
    return float(np.clip(score, -1.0, 1.0))


class SentimentService:
    """Сервис расчёта тональности сообщений
    
    Класс загружает финансовую модель тональности при наличии зависимостей
    и использует словарный резервный режим, если модель недоступна
    
    Attributes:
        cfg: конфигурация проекта с параметрами модели тональности
        model: загруженная модель трансформера или значение None
        tokenizer: токенизатор модели тональности или значение None
    """
    def __init__(self, cfg: AppConfig):
        """Инициализирует сервис тональности и пытается загрузить модель
        
        Args:
            cfg: конфигурация проекта или приложения
        
        Returns:
            None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
        """
        self.backend = str(cfg.get("nlp", "sentiment_backend", default="finbert") or "finbert").lower()
        self.model_name = cfg.get("nlp", "finbert_model", default="ProsusAI/finbert")
        self.revision = cfg.get("nlp", "finbert_revision", default=None)
        self.batch_size = int(cfg.get("nlp", "batch_size", default=64))
        self.max_length = int(cfg.get("nlp", "sentiment_max_length", default=160))
        self.device = str(cfg.get("nlp", "sentiment_device", default="auto") or "auto").lower()
        self._pipe = None

    def _load(self):
        """Загружает трансформерную модель тональности и токенизатор
        
        Returns:
            pipeline модели тональности или None при переходе в резервный режим
        """
        if self.backend != "finbert":
            return None
        if self._pipe is None:
            try:
                from transformers import pipeline
                kwargs = {"model": self.model_name, "tokenizer": self.model_name, "truncation": True}
                if self.revision:
                    kwargs["revision"] = self.revision
                if self.device in {"cuda", "gpu", "0"}:
                    kwargs["device"] = 0
                elif self.device == "cpu":
                    kwargs["device"] = -1
                elif self.device == "auto":
                    try:
                        import torch
                        kwargs["device"] = 0 if torch.cuda.is_available() else -1
                    except Exception:
                        pass
                self._pipe = pipeline("sentiment-analysis", **kwargs)
            except Exception:
                self.backend = "lexicon"
                self._pipe = None
        return self._pipe

    def score_many(self, texts: list[str]) -> list[float]:
        """Рассчитывает тональность для списка текстов пакетами
        
        Args:
            texts: список текстов сообщений
        
        Returns:
            список числовых оценок тональности для входных текстов
        """
        fallback = [self._lexicon_score(t) for t in texts]
        if self.backend == "finbert" and self._load() is not None:
            try:
                out = self._pipe(texts, batch_size=self.batch_size, truncation=True, max_length=self.max_length)
                model_scores = [self._to_score(item) for item in out]
                return [self._blend_score(m, l) for m, l in zip(model_scores, fallback)]
            except Exception:
                self.backend = "lexicon"
        return fallback

    @staticmethod
    def _to_score(item: dict[str, Any]) -> float:
        """Преобразует выход модели тональности в числовую шкалу
        
        Args:
            item: выход одной записи модели или элемент входной структуры
        
        Returns:
            числовая оценка тональности в диапазоне от минус единицы до единицы
        """
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        if "positive" in label:
            return score
        if "negative" in label:
            return -score
        return 0.0

    @staticmethod
    def _blend_score(model_score: float, fallback_score: float) -> float:
        """Смешивает модельную тональность со словарной оценкой финансовых выражений
        
        Args:
            model_score: оценка тональности, полученная моделью
            fallback_score: резервная словарная оценка тональности
        
        Returns:
            смешанная оценка тональности после ограничения диапазона
        """
        model_score = float(np.clip(model_score, -1.0, 1.0))
        fallback_score = float(np.clip(fallback_score, -1.0, 1.0))
        if abs(model_score) < 0.08:
            return fallback_score
        if fallback_score and np.sign(model_score) == np.sign(fallback_score):
            return float(np.clip(0.70 * model_score + 0.30 * fallback_score, -1.0, 1.0))
        return model_score

    @staticmethod
    def _lexicon_score(text: str) -> float:
        """Рассчитывает словарную оценку тональности для резервного режима
        
        Args:
            text: текст сообщения или текстовое значение
        
        Returns:
            словарная оценка тональности в диапазоне от минус единицы до единицы
        """
        words = re.findall(r"[A-Za-z]+", str(text or "").lower())
        phrase_score = _market_phrase_sentiment(text)
        if not words:
            return phrase_score
        pos = sum(w in _POS for w in words)
        neg = sum(w in _NEG for w in words)
        lexical = float(np.clip((pos - neg) / max(1, pos + neg + 1), -1, 1))
        if abs(phrase_score) > abs(lexical):
            return phrase_score
        if phrase_score and np.sign(phrase_score) == np.sign(lexical):
            return float(np.clip(0.60 * lexical + 0.40 * phrase_score, -1.0, 1.0))
        return lexical



def _sentiment_cache_key(text: str, service: SentimentService) -> str:
    """Формирует ключ кэша тональности для текста и сервиса
    
    Args:
        text: текст сообщения или текстовое значение
        service: сервис или модель, использованная для расчёта признака
    
    Returns:
        хеш-ключ записи кэша тональности
    """
    raw = f"{service.backend}|{service.model_name}|{service.revision}|{service.max_length}|{clean_text(text)}"
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def _sentiment_cache_path(cfg: AppConfig, service: SentimentService) -> Path:
    """Возвращает путь к файлу кэша тональности
    
    Args:
        cfg: конфигурация проекта или приложения
        service: сервис или модель, использованная для расчёта признака
    
    Returns:
        путь к CSV-файлу кэша тональности
    """
    root = cfg.data_dir / "cache"
    root.mkdir(parents=True, exist_ok=True)
    model_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(service.model_name or service.backend))[:80]
    return root / f"sentiment_{service.backend}_{model_tag}_ml{service.max_length}.jsonl"


def _load_sentiment_cache(path: Path) -> dict[str, float]:
    """Загружает сохранённые оценки тональности из кэша
    
    Args:
        path: путь к файлу или директории
    
    Returns:
        загруженный результат: сохранённые оценки тональности из кэша
    """
    cache: dict[str, float] = {}
    if not path.exists():
        return cache
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                key = str(item.get("key", ""))
                if key:
                    cache[key] = float(item.get("score", 0.0))
    except Exception:
        return cache
    return cache


def _append_sentiment_cache(path: Path, rows: list[tuple[str, float]]) -> None:
    """Добавляет новые оценки тональности в кэш
    
    Args:
        path: путь к файлу или директории
        rows: строки, подлежащие сохранению или объединению
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for key, score in rows:
            f.write(json.dumps({"key": key, "score": float(score)}, ensure_ascii=False) + "\n")


def _existing_sentiment_column(df: pd.DataFrame) -> str | None:
    """Находит уже рассчитанную колонку тональности в таблице
    
    Args:
        df: таблица pandas с сообщениями, событиями или результатами
    
    Returns:
        название найденной колонки тональности или None
    """
    candidates = [
        "sentiment", "sentiment_score", "finbert_score", "polarity", "compound",
        "sentiment_value", "text_sentiment", "score_sentiment",
    ]
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _coerce_existing_sentiment(series: pd.Series) -> pd.Series:
    """Приводит существующие оценки тональности к рабочей числовой шкале
    
    Args:
        series: серия pandas с исходными значениями
    
    Returns:
        серия тональности, приведённая к рабочей шкале
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        out = numeric.fillna(0.0).astype(float)
        # некоторые датасеты хранят FinBERT-подобные оценки как позитивность от 0 до 1
        # знаковые оценки сохраняем, а экстремальные масштабы ограничиваем
        if out.abs().max() > 1.0:
            out = out / max(float(out.abs().max()), 1e-12)
        return out.clip(-1.0, 1.0)
    labels = series.astype(str).str.lower().str.strip()
    mapping = {
        "positive": 1.0, "pos": 1.0, "bullish": 1.0, "позитив": 1.0,
        "negative": -1.0, "neg": -1.0, "bearish": -1.0, "негатив": -1.0,
        "neutral": 0.0, "neu": 0.0, "нейтрально": 0.0,
    }
    return labels.map(mapping).fillna(0.0).astype(float)


def _score_sentiment_with_cache(
    cfg: AppConfig,
    texts: list[str],
    progress=None,
    stage: str = "sentiment",
    use_cache: bool | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Рассчитывает тональность сообщений с использованием кэша
    
    Args:
        cfg: конфигурация проекта или приложения
        texts: список текстов сообщений
        progress: callback для передачи статуса выполнения
        stage: название этапа вычислительного конвейера
        use_cache: явное разрешение использовать CSV-кэш тональности
    
    Returns:
        tuple[list[float], dict[str, Any]]: оценки тональности и технические сведения о расчёте
    """
    service = SentimentService(cfg)
    policy = str(cfg.get("nlp", "sentiment_policy", default="reuse_or_compute") or "reuse_or_compute").lower()
    if use_cache is None:
        cache_allowed = bool(cfg.get("nlp", "sentiment_cache_enabled", default=True))
    else:
        cache_allowed = bool(use_cache)
    cache_enabled = bool(cache_allowed and policy != "always_compute")
    cache_path = _sentiment_cache_path(cfg, service) if cache_enabled else None
    cache = _load_sentiment_cache(cache_path) if cache_enabled and cache_path is not None else {}
    scores: list[float | None] = [None] * len(texts)
    missing_idx: list[int] = []
    cache_hits = 0
    for i, text_value in enumerate(texts):
        key = _sentiment_cache_key(text_value, service)
        if key in cache:
            scores[i] = float(cache[key])
            cache_hits += 1
        else:
            missing_idx.append(i)
    batch_size = max(1, int(cfg.get("nlp", "batch_size", default=64)))
    cache_rows: list[tuple[str, float]] = []
    total_missing = len(missing_idx)
    if total_missing:
        if cache_enabled:
            _emit_progress(progress, f"Тональность: кэш {cache_hits}/{len(texts)}, расчёт {total_missing}", cache_hits, max(1, len(texts)), stage=stage)
        else:
            _emit_progress(progress, f"Тональность: расчёт 0/{len(texts)}", 0, max(1, len(texts)), stage=stage)
    for pos in range(0, total_missing, batch_size):
        idx_batch = missing_idx[pos:pos + batch_size]
        part_texts = [texts[i] for i in idx_batch]
        part_scores = service.score_many(part_texts)
        for row_idx, score in zip(idx_batch, part_scores):
            value = float(score)
            scores[row_idx] = value
            cache_rows.append((_sentiment_cache_key(texts[row_idx], service), value))
        done = cache_hits + min(pos + batch_size, total_missing)
        _emit_progress(progress, f"Тональность: {done}/{len(texts)}", done, len(texts), stage=stage)
    if cache_enabled and cache_path is not None:
        _append_sentiment_cache(cache_path, cache_rows)
    final_scores = [float(x if x is not None else 0.0) for x in scores]
    meta = {
        "backend": service.backend,
        "cache_enabled": bool(cache_enabled),
        "cache_hits": int(cache_hits),
        "cache_misses": int(total_missing),
        "cache_path": str(cache_path),
        "batch_size": int(batch_size),
        "max_length": int(service.max_length),
        "device": service.device,
    }
    return final_scores, meta
