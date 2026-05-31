"""Тематическое моделирование и укрупнение тематического пространства"""
from __future__ import annotations


from ..pipeline.common import *
from ..pipeline.configuration import AppConfig

def _aggregate_topic_labels_by_frequency(labels: list[int], target_n: int) -> list[int]:
    """Укрупняет детальные темы в группы по частоте сообщений
    
    Args:
        labels: метки тем или классов
        target_n: целевое число укрупнённых тематических групп
    
    Returns:
        список укрупнённых тематических меток
    """
    clean = [int(x) for x in labels]
    non_noise = [x for x in clean if x >= 0]
    if not non_noise:
        return [0 for _ in clean]
    target_n = max(2, int(target_n))
    top = [topic for topic, _ in Counter(non_noise).most_common(max(1, target_n - 1))]
    top_map = {topic: i for i, topic in enumerate(top)}
    other_bucket = len(top)
    return [top_map.get(x, other_bucket) if x >= 0 else other_bucket for x in clean]


class TopicService:
    """Сервис тематического моделирования сообщений
    
    Класс использует BERTopic при доступности зависимостей
    или резервный TF-IDF KMeans режим для исследовательского запуска
    
    Attributes:
        cfg: конфигурация тематического моделирования
        model: объект обученной тематической модели
        fallback_vectorizer: векторизатор резервного режима
        fallback_clusterer: кластеризатор резервного режима
    """
    def __init__(self, cfg: AppConfig):
        """Инициализирует сервис тематического моделирования
        
        Args:
            cfg: конфигурация проекта или приложения
        
        Returns:
            None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
        """
        self.cfg = cfg
        self.backend = cfg.get("nlp", "topic_backend", default="bertopic")
        self.model: Any = None
        self.vectorizer: Any = None

    def fit_transform(self, texts: list[str]) -> tuple[list[int], pd.DataFrame]:
        """Обучает тематическую модель и возвращает номера тем
        
        Args:
            texts: список текстов сообщений
        
        Returns:
            таблица сообщений с темами и список меток тем
        """
        self.detailed_labels: list[int] | None = None
        self.detailed_info: pd.DataFrame | None = None
        self.active_topic_mode = str(self.cfg.get("nlp", "simulation_topic_mode", default="detailed") or "detailed")
        self.simulation_topic_count = int(self.cfg.get("nlp", "simulation_topic_count", default=6))
        if self.backend == "bertopic":
            try:
                from bertopic import BERTopic
                embedding_model = self.cfg.get("nlp", "embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
                # для основного режима симуляции сначала строим детальное тематическое пространство
                # затем укрупняем его для агентной модели
                # детальные метки сохраняем в topic_indicator и topic_detailed
                # это позволяет индикаторам анализировать неукрупнённую тематическую экспозицию
                # без потери связи инструмента с исходными темами
                aggregate_for_simulation = self.active_topic_mode in {"aggregate", "aggregated", "coarse"}
                nr_topics = None if aggregate_for_simulation else self.cfg.get("nlp", "nr_topics", default="auto")
                model = BERTopic(
                    embedding_model=embedding_model,
                    min_topic_size=int(self.cfg.get("nlp", "min_topic_size", default=20)),
                    nr_topics=nr_topics,
                    calculate_probabilities=False,
                    verbose=False,
                )
                detailed_topics, _ = model.fit_transform(texts)
                self.detailed_labels = [int(t) for t in detailed_topics]
                self.detailed_info = model.get_topic_info()
                active_topics = self.detailed_labels
                if aggregate_for_simulation:
                    target_n = int(max(2, min(50, self.simulation_topic_count)))
                    try:
                        model.reduce_topics(texts, nr_topics=target_n)
                        active_topics = [int(t) for t in getattr(model, "topics_", self.detailed_labels)]
                    except Exception:
                        active_topics = _aggregate_topic_labels_by_frequency(self.detailed_labels, target_n)
                self.model = model
                info = model.get_topic_info()
                return [int(t) for t in active_topics], info
            except Exception:
                self.backend = "tfidf_kmeans"
        return self._fit_tfidf_kmeans(texts)

    def _fit_tfidf_kmeans(self, texts: list[str]) -> tuple[list[int], pd.DataFrame]:
        """Запускает резервное тематическое моделирование через TF-IDF и KMeans
        
        Args:
            texts: список текстов сообщений
        
        Returns:
            список тематических меток резервной модели
        """
        n = len(texts)
        if n <= 1:
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            try:
                self.vectorizer.fit(texts or ["empty"])
            except Exception:
                self.vectorizer.fit(["empty"])
            self.model = None
            labels = [0 for _ in texts]
            self.detailed_labels = labels
            self.detailed_info = pd.DataFrame({"Topic": [0], "Count": [len(texts)]})
            return labels, pd.DataFrame({"Topic": [0], "Count": [len(texts)]})
        if str(self.cfg.get("nlp", "simulation_topic_mode", default="detailed") or "detailed") in {"aggregate", "aggregated", "coarse"}:
            k = int(max(2, min(50, self.cfg.get("nlp", "simulation_topic_count", default=6))))
        else:
            k = max(2, min(20, int(math.sqrt(max(2, n / 2)))))
        k = min(k, n)
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        X = self.vectorizer.fit_transform(texts)
        self.model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = [int(x) for x in self.model.fit_predict(X)]
        self.detailed_labels = labels
        self.detailed_info = pd.DataFrame({"Topic": sorted(set(labels)), "Count": pd.Series(labels).value_counts().sort_index().values})
        info = pd.DataFrame({"Topic": sorted(set(labels)), "Count": pd.Series(labels).value_counts().sort_index().values})
        return labels, info

    def transform(self, texts: list[str]) -> list[int]:
        """Применяет обученную тематическую модель к новым текстам
        
        Args:
            texts: список текстов сообщений
        
        Returns:
            список номеров тем для новых текстов
        """
        if not texts:
            return []
        if self.backend == "bertopic" and self.model is not None and hasattr(self.model, "transform"):
            try:
                labels, _ = self.model.transform(texts)
                return [int(x) for x in labels]
            except Exception:
                pass
        if self.vectorizer is not None and self.model is not None and hasattr(self.model, "predict"):
            try:
                X = self.vectorizer.transform(texts)
                labels = self.model.predict(X)
                return [int(x) for x in labels]
            except Exception:
                pass
        return [0 for _ in texts]

    def save(self, model_dir: Path) -> None:
        """Сохраняет тематическую модель и описание тем
        
        Args:
            model_dir: директория сохранения модели
        
        Returns:
            None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
        """
        model_dir.mkdir(parents=True, exist_ok=True)
        if self.backend == "bertopic" and hasattr(self.model, "save"):
            try:
                self.model.save(str(model_dir / "topic_model"), serialization="safetensors", save_ctfidf=True)
                (model_dir / "topic_backend.txt").write_text("bertopic_safetensors", encoding="utf-8")
                return
            except Exception:
                pass
        joblib.dump({"backend": self.backend, "model": self.model, "vectorizer": self.vectorizer}, model_dir / "topic_model.joblib")
        (model_dir / "topic_backend.txt").write_text(self.backend, encoding="utf-8")
