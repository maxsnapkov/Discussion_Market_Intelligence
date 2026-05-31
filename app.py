"""Интерфейс исследовательского комплекса анализа дискуссионных событий

Модуль собирает элементы Streamlit-интерфейса, управляет пользовательскими параметрами
запускает вычислительный конвейер и показывает результаты моделирования
рыночная часть используется только для ассоциативной проверки
"""

from __future__ import annotations

import os
import json
import re
import shutil
import ast
import hashlib
import logging
import warnings
import io
import zipfile
import yaml
import copy
from collections import Counter
from html import escape as html_escape
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.utils.import_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r".*Accessing `__path__`.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*Behavior may be different.*", category=FutureWarning)

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import streamlit as st

from ui.theme import load_design_system, plotly_chart
from ui.components import app_header, page_header, author_footer, section_header, analysis_context_banner, executive_board, parameter_rail, sidebar_brand
from ui.components import metric_card as ui_metric_card, info_panel as ui_info_panel, chip_row

from src.pipeline.configuration import AppConfig, detect_raw_schema, is_service_author, normalize_columns, service_author_label
from src.data_ingestion.preparation import initialize_project, load_windows
from src.data_ingestion.raw_interval import run_predictive_market_indicator_for_raw_range
from src.nlp.ticker_extraction import load_ticker_universe
from src.discussion.influence_payoff import recompute_payoff_file
from src.discussion.publications import top_publications_for_window
from src.discussion.user_graph import build_user_graph_snapshot_from_messages
from src.simulation.calibration import calibrate_model, calibrate_model_optuna
from src.simulation.dynamic_state import (
    build_dynamic_window_state,
    load_fixed_model_params,
    run_predictive_market_indicator,
    run_predictive_market_indicator_for_range,
)
from src.simulation.modeling import run_backtest, run_window_modeling
from src.market.association import load_market_price_panel

PAGE_ICON = None
try:
    from PIL import Image
    icon_path = Path("ui/assets/favicon.png")
    if icon_path.exists():
        PAGE_ICON = Image.open(icon_path)
except Exception:
    PAGE_ICON = None

st.set_page_config(page_title="Discussion Market Intelligence", page_icon=PAGE_ICON, layout="wide")


if not hasattr(st, "_dmi_original_number_input"):
    st._dmi_original_number_input = st.number_input
_ORIGINAL_ST_NUMBER_INPUT = st._dmi_original_number_input


def _is_number_like(value: Any) -> bool:
    """Проверяет, можно ли значение безопасно использовать как число в виджете Streamlit
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        bool: True, если значение можно привести к числу для числового виджета Streamlit
    """
    try:
        if value is None:
            return False
        float(value)
        return True
    except Exception:
        return False


def _coerce_number_for_streamlit(value: Any, min_value: Any = None, max_value: Any = None, step: Any = None) -> Any:
    """Приводит значение виджета к числовому типу и границам, заданным в интерфейсе
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
        min_value: нижняя граница допустимого значения
        max_value: верхняя граница допустимого значения
        step: шаг изменения значения или номер шага анализа
    
    Returns:
        Any: значение, приведённое к допустимому типу и диапазону числового виджета
    """
    if not _is_number_like(value):
        return value
    # Streamlit определяет тип числового виджета по границам, значению и шагу
    # целочисленные виджеты оставляем целочисленными, остальные переводим в float
    int_widget = all(isinstance(x, int) and not isinstance(x, bool) for x in [min_value, max_value, step] if x is not None)
    try:
        v = float(value)
        if min_value is not None:
            v = max(v, float(min_value))
        if max_value is not None:
            v = min(v, float(max_value))
        return int(round(v)) if int_widget else float(v)
    except Exception:
        return value


def _safe_number_input(label: str, *args: Any, **kwargs: Any) -> Any:
    """Создаёт числовой ввод Streamlit с защитой от устаревших значений вне текущих границ
    
    Args:
        label: подпись элемента интерфейса
        *args: позиционные параметры, передаваемые во внутренний вызов
        **kwargs: дополнительные именованные параметры, передаваемые во внутренний вызов
    
    Returns:
        Any: результат вызова исходного Streamlit number_input
    """
    min_value = kwargs.get("min_value")
    max_value = kwargs.get("max_value")
    step = kwargs.get("step")
    if "value" in kwargs:
        kwargs["value"] = _coerce_number_for_streamlit(kwargs.get("value"), min_value, max_value, step)
    # здесь не изменяем session_state, чтобы Streamlit не ругался на двойную инициализацию
    # ключ уже может быть задан до создания виджета
    # значения вне границ обрабатываются ограничением явного default
    # несовместимые сохранённые значения нужно исправлять в месте создания состояния
    # общая обёртка не должна менять источник состояния
    return _ORIGINAL_ST_NUMBER_INPUT(label, *args, **kwargs)


st.number_input = _safe_number_input

from ui.dashboard import bind_context
from ui.dashboard.session_state import (
    CONFIG_PATH,
    ENV_LOADED,
    ENV_PATH,
    _default_interval_bounds_from_meta,
    _materialize_interval_raw_csv,
    _raw_interval_metadata,
    _reset_widget_state_value,
    _select_discretization_interval,
    _set_widget_state_default,
    _write_temp_windows_subset,
    cfg,
    clamp_float,
    parse_alpha_value,
    resolve_raw_dataset_path,
    state_paths,
)
from ui.dashboard.data_loading import render_raw_dataset_preview, save_uploaded_raw_csv
from ui.dashboard.controls import (
    _normalize_ui_weights,
    clear_current_analysis_outputs,
    default_w_sentiment,
    fmt,
    format_seconds,
    future_steps,
    info_panel,
    metric_card,
    parse_float_list,
    pct,
    progress_writer,
    range_slider,
    read_json,
    runtime_payoff_path,
    sentiment_mode_index,
    weight_inputs,
)
from ui.dashboard.summaries import _empty_ticker_summary, _forward_ticker_summary, _ticker_universe_caption, compact_forward_indicators
from ui.dashboard.reports import (
    render_analysis_context_header,
    render_discussion_state_overview,
    render_executive_report,
    render_forecast_check_text,
)
from ui.dashboard.publication_cards import load_publications_for_signal_window, render_publication_cards
from ui.dashboard.market_view import (
    download_dataframe_button,
    plot_forward_validation,
    plot_market_price_panel,
    plot_ticker_validation,
    render_benchmark_explanation,
    render_market_indicator_cards,
    render_price_interpretation,
)
from ui.dashboard.exports_and_graphs import (
    build_experiment_package,
    compact_forecasts,
    plot_forecast_errors,
    render_user_graph,
    show_quality_cards,
)
from ui.dashboard.topic_views import (
    plot_topic_distribution,
    plot_topic_trajectories,
    render_current_window_topic_distribution,
    topic_name_map,
)
from ui.dashboard.text_normalization import _patch_text_function, _patch_widget_text

load_design_system()

for _name in ["caption", "info", "warning", "success", "error"]:
    if hasattr(st, _name):
        _patch_text_function(_name)
if hasattr(st, "markdown"):
    _patch_text_function("markdown", skip_html=False)

for _name in ["radio", "selectbox", "checkbox", "button", "text_input", "date_input", "slider", "multiselect", "download_button", "expander"]:
    _patch_widget_text(_name)

c = cfg()
if isinstance(st.session_state.get("runtime_config_override"), dict):
    c = AppConfig(st.session_state["runtime_config_override"])
paths = state_paths(c)
state = read_json(paths["state"]) or {}
window_hours_state = int(state.get("window_hours", c.window_hours))
bind_context(c=c, paths=paths, state=state, window_hours_state=window_hours_state, CONFIG_PATH=CONFIG_PATH, ENV_PATH=ENV_PATH, ENV_LOADED=ENV_LOADED)

with st.sidebar:
    sidebar_brand(
        "Discussion Market Intelligence",
        "Анализ пользовательских дискуссий социальных сетей, модельной динамики и событийной проверки реакции рынка",
    )

status = "инициализирован" if paths["windows"].exists() and paths["payoff"].exists() else "требуется инициализация"
st.sidebar.markdown(f"**Статус проекта:** {status}")
if state:
    st.sidebar.caption(f"Шаг дискретизации: {state.get('window_hours', window_hours_state)} ч · сообщений: {state.get('n_messages', '-')} · тем: {state.get('n_topics', '-')}")
    st.sidebar.caption(f"Авторитетность: {state.get('influence_mode', 'dynamic_decayed_pagerank')}")
st.sidebar.divider()
page = st.sidebar.radio(
    "Навигация сервиса",
    ["Обзор", "Загрузка данных для моделирования", "Калибровка", "Моделирование дискуссии", "Расчёт предупреждающих индикаторов"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption("Параметры окружения можно задавать через .env, .env.local или APP_ENV_FILE")
if ENV_LOADED and ENV_PATH is not None:
    st.sidebar.success(f"env-файл найден: {ENV_PATH}")
else:
    st.sidebar.info("env-файл не найден - используются системные переменные и YAML-конфиг")

PAGE_HEADER_META = {
    "Загрузка данных для моделирования": ("Загрузка данных для моделирования", "", "Выбор CSV-файла, временного интервала и базового шага дискретизации"),
    "Калибровка": ("Калибровка модели", "", "Подбор параметров эволюционной модели и весов payoff по историческим шагам дискретизации"),
    "Моделирование дискуссии": ("Моделирование дискуссии", "", "Запуск NetLogo-модели для выбранного шага дискретизации и сравнение модельной симуляции с фактической динамикой тем"),
    "Расчёт предупреждающих индикаторов": ("Расчёт предупреждающих индикаторов", "", "Расчёт дискуссионных индикаторов, публикаций, социального графа и рыночной ассоциативной проверки"),
}

if page == "Обзор":
    app_header(
        "Discussion Market Intelligence",
        "Исследовательская среда для проведения экспериментов по моделированию динамики распределения тем пользовательских дискуссий социальных сетей",
        author={
            "Магистр": "Снапков Максим Юрьевич",
            "СПбГУ": "2026",
            "Тема исследования": "влияние мнений пользователей социальных сетей на ценообразование финансовых инструментов компаний",
        },
        show_logo=False,
        show_eyebrow=False,
    )
else:
    _title, _step, _desc = PAGE_HEADER_META.get(page, (page, "Рабочий блок", ""))
    page_header(_title, _desc, _step)

if page == "Обзор":
    section_header(
        "Обзор программного комплекса",
        "Исследовательский контур симуляции тематической динамики и расчёта предупреждающих дискуссионных индикаторов",
    )
    # chip_row([
    #     ("Режим 1: агентная симуляция", "green"),
    #     ("Крупные группы тем: 4-8", "blue"),
    #     ("Детальные темы сохранены для индикаторов", "violet"),
    #     ("Тональность: кэш / повторное использование", "orange"),
    # ])

    info_panel(
        "Основная идея",
        "Программный комплекс рассматривает поток публикаций социальных сетей как дискретную динамическую систему. На каждом шаге дискретизации строится тематическое распределение, рассчитываются признаки сообщений и запускается агентная симуляция распространения тем.",
        "blue",
    )
    info_panel(
        "Научная значимость",
        "Ключевой сценарий комплекса - построение правдоподобной тематической траектории на будущие шаги дискретизации без необходимости обучать тяжёлую supervised-модель на большом количестве размеченных примеров. Механизм основан на интерпретируемой цепочке: темы как стратегии агентов, payoff темы, правило imitate-if-better, ограниченная рациональность и сравнение модельной симуляции с историческим контекстом.",
        "green",
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Статус", status, "готовность обработанных данных", "green" if status == "инициализирован" else "orange")
    with k2:
        metric_card("Шаг дискретизации", f"{window_hours_state} ч", "базовая конфигурация - 6 ч", "blue")
    with k3:
        metric_card("Публикаций", str(state.get("n_messages", "-")), "последняя подготовка данных", "gray")
    with k4:
        metric_card("Тем", str(state.get("n_topics", "-")), "тематическое пространство", "violet")

    st.subheader("Сценарии использования")
    s1, s2 = st.columns(2)
    with s1:
        info_panel(
            "Сценарий 1 - основной",
            "Исследователь выбирает файл и временной интервал, выполняет калибровку на исторической части данных и запускает агентную симуляцию для выбранного стартового шага дискретизации. Система показывает, какие темы могут усилиться или ослабнуть в модельной симуляции, и насколько это перестроение отличается от обычной динамики дискуссии.",
            "green",
        )
    with s2:
        info_panel(
            "Сценарий 2 - дополнительный",
            "После симуляции система рассчитывает предупреждающие дискуссионные индикаторы для финансовых инструментов. Этот блок нужен, чтобы проверить, является ли модельный дискуссионный сигнал содержательно связанным с тикерными обсуждениями и последующей рыночной динамикой. Он не доказывает причинность и не является оценкой вероятности рыночного события.",
            "orange",
        )

    st.subheader("Методический контур")
    m1, m2, m3 = st.columns(3)
    with m1:
        info_panel(
            "1. Данные и темы",
            "Публикации очищаются, размечаются по времени, получают тематические метки, тональность, авторитетность автора и связи с финансовыми инструментами. Шаг дискретизации задаёт масштаб агрегирования, базовое значение - 6 часов.",
            "blue",
        )
    with m2:
        info_panel(
            "2. Агентная симуляция",
            "Темы интерпретируются как стратегии агентов. Агенты пересматривают стратегию с заданной вероятностью, могут выбирать случайно с уровнем шума или имитировать более успешную стратегию по правилу imitate-if-better.",
            "green",
        )
    with m3:
        info_panel(
            "3. Индикаторы",
            "Система рассчитывает текущую статистическую необычность тикерного обсуждения, модельно-тематическую поддержку и комбинированный предупреждающий индикатор. Значение индикатора - нормированный скор необычности, а не вероятность движения цены.",
            "violet",
        )

    st.subheader("Интерпретационные ограничения")
    info_panel(
        "Что можно утверждать",
        "Можно утверждать, что комплекс выявляет аномальные изменения структуры и интенсивности дискуссий, строит интерпретируемую агентную симуляцию тематической динамики и сопоставляет дискуссионные события с рыночными рядами в режиме ассоциативной проверки.",
        "green",
    )
    info_panel(
        "Что нельзя утверждать",
        "Нельзя утверждать, что дискуссия доказанно вызывает движение цены, что значение индикатора 0.8 означает 80% вероятность рыночного события или что выбранные пороги являются теоретически оптимальными без отдельных экспериментов чувствительности и out-of-sample проверки.",
        "orange",
    )

    st.subheader("Параметры текущего запуска")
    st.caption("В этом блоке параметры задаются через элементы интерфейса. Изменения можно применить только к текущему запуску или сохранить в файл конфигурации.")

    def _cfg_value(section: str, key: str, default: Any = None) -> Any:
        """Берёт значение из вложенной секции конфигурации интерфейса
        
        Args:
            section: секция конфигурации
            key: ключ параметра, виджета или словаря
            default: резервное значение
        
        Returns:
            Any: значение параметра конфигурации или резервное значение
        """
        value = c.get(section, key, default=default)
        return default if value is None else value

    def _select_idx(options: list[Any], value: Any, default_index: int = 0) -> int:
        """Определяет индекс выбранного значения в списке вариантов
        
        Args:
            options: список вариантов выбора
            value: исходное значение, которое требуется нормализовать или преобразовать
            default_index: индекс варианта по умолчанию
        
        Returns:
            int: индекс выбранного значения или индекс по умолчанию
        """
        value_s = str(value)
        for i, opt in enumerate(options):
            if str(opt) == value_s:
                return i
        return int(default_index)

    def _to_number_or_auto(value: Any) -> Any:
        """Преобразует значение параметра в число или маркер автоматического режима
        
        Args:
            value: исходное значение, которое требуется нормализовать или преобразовать
        
        Returns:
            Any: целое число или строковый маркер автоматического режима
        """
        text_value = str(value).strip()
        if text_value.lower() in {"", "auto", "none", "null"}:
            return "auto"
        try:
            return int(text_value)
        except Exception:
            return text_value

    tab_project, tab_nlp, tab_payoff, tab_agent, tab_market = st.tabs([
        "Проект и данные",
        "NLP и тикеры",
        "Payoff",
        "Агентная модель",
        "Рыночная проверка",
    ])

    with tab_project:
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            ui_project_name = st.text_input("Название проекта", value=str(_cfg_value("project", "name", "discussion-market-model")), key="cfg_project_name")
            ui_step_hours = st.number_input("Размер шага дискретизации, часов", min_value=1, max_value=168, value=int(_cfg_value("project", "window_size_hours", 6)), step=1, key="cfg_step_hours")
            ui_data_dir = st.text_input("Каталог данных", value=str(_cfg_value("project", "data_dir", "data")), key="cfg_data_dir")
        with p_col2:
            ui_output_dir = st.text_input("Каталог результатов", value=str(_cfg_value("project", "output_dir", "results")), key="cfg_output_dir")
            ui_model_dir = st.text_input("Каталог моделей", value=str(_cfg_value("project", "model_dir", "models")), key="cfg_model_dir")
            st.info("Базовый шаг дискретизации для экспериментов ВКР - 6 часов. Изменение параметра допустимо для анализа чувствительности.")

    with tab_nlp:
        n_col1, n_col2 = st.columns(2)
        with n_col1:
            topic_options = ["bertopic", "tfidf_kmeans"]
            ui_topic_backend = st.selectbox("Метод тематического моделирования", topic_options, index=_select_idx(topic_options, _cfg_value("nlp", "topic_backend", "bertopic")), key="cfg_topic_backend")
            ui_embedding_model = st.text_input("Модель эмбеддингов", value=str(_cfg_value("nlp", "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")), key="cfg_embedding_model")
            ui_min_topic_size = st.number_input("Минимальный размер темы", min_value=2, max_value=10000, value=int(_cfg_value("nlp", "min_topic_size", 20)), step=1, key="cfg_min_topic_size")
            ui_nr_topics = st.text_input("Детальное количество тем", value=str(_cfg_value("nlp", "nr_topics", "auto")), key="cfg_nr_topics", help="Для индикаторного режима можно оставить auto; для основного режима ниже задаётся укрупнение тем для симуляции")
            topic_mode_options = ["aggregate", "detailed"]
            ui_sim_topic_mode = st.selectbox("Темы для агентной симуляции", topic_mode_options, index=_select_idx(topic_mode_options, _cfg_value("nlp", "simulation_topic_mode", "aggregate")), key="cfg_sim_topic_mode", help="aggregate - укрупнить детальные BERTopic-темы до небольшого числа стратегий агентов; detailed - использовать детальные темы напрямую")
            ui_sim_topic_count = st.slider("Количество крупных групп для режима симуляции", min_value=4, max_value=8, value=int(_cfg_value("nlp", "simulation_topic_count", 6)), step=1, key="cfg_sim_topic_count")
            ui_keep_detailed_topics = st.checkbox("Сохранять исходные детальные темы для индикаторов", value=bool(_cfg_value("nlp", "indicator_keep_detailed_topics", True)), key="cfg_keep_detailed_topics")
            sentiment_options = ["finbert", "lexicon"]
            ui_sentiment_backend = st.selectbox("Метод оценки тональности", sentiment_options, index=_select_idx(sentiment_options, _cfg_value("nlp", "sentiment_backend", "finbert")), key="cfg_sentiment_backend")
            sentiment_policy_options = ["reuse_or_compute", "always_compute", "reuse_only", "skip"]
            ui_sentiment_policy = st.selectbox("Политика расчёта тональности", sentiment_policy_options, index=_select_idx(sentiment_policy_options, _cfg_value("nlp", "sentiment_policy", "reuse_or_compute")), key="cfg_sentiment_policy", help="reuse_or_compute использует готовую колонку sentiment, если она есть, иначе считает; always_compute всегда пересчитывает; reuse_only не запускает модель, если готовой колонки нет; skip ставит нейтральные значения")
            ui_finbert_model = st.text_input("FinBERT-модель", value=str(_cfg_value("nlp", "finbert_model", "ProsusAI/finbert")), key="cfg_finbert_model")
        with n_col2:
            ticker_options = ["hybrid_fast", "gliner_hybrid", "gliner_full", "none"]
            ui_ticker_mode = st.selectbox("Извлечение финансовых инструментов", ticker_options, index=_select_idx(ticker_options, _cfg_value("nlp", "ticker_extraction", "hybrid_fast")), key="cfg_ticker_mode")
            ui_batch_size = st.number_input("Batch тональности", min_value=1, max_value=512, value=int(_cfg_value("nlp", "batch_size", 64)), step=1, key="cfg_batch_size")
            ui_sentiment_max_length = st.number_input("Максимум токенов для тональности", min_value=32, max_value=512, value=int(_cfg_value("nlp", "sentiment_max_length", 160)), step=16, key="cfg_sentiment_max_length")
            sentiment_device_options = ["auto", "cpu", "cuda"]
            ui_sentiment_device = st.selectbox("Устройство для тональности", sentiment_device_options, index=_select_idx(sentiment_device_options, _cfg_value("nlp", "sentiment_device", "auto")), key="cfg_sentiment_device")
            ui_sentiment_cache = st.checkbox("Кэшировать тональность по тексту", value=bool(_cfg_value("nlp", "sentiment_cache_enabled", True)), key="cfg_sentiment_cache")
            ui_ticker_batch_size = st.number_input("Блок обработки тикеров", min_value=256, max_value=100000, value=int(_cfg_value("nlp", "ticker_batch_size", 4096)), step=256, key="cfg_ticker_batch_size")
            ui_gliner_model = st.text_input("GLiNER-модель", value=str(_cfg_value("nlp", "gliner_model", "urchade/gliner_small-v2.1")), key="cfg_gliner_model")
            ui_gliner_threshold = st.slider("Порог GLiNER", min_value=0.05, max_value=0.95, value=float(_cfg_value("nlp", "gliner_threshold", 0.35)), step=0.01, key="cfg_gliner_threshold")
            gliner_policy_options = ["missing_only", "all"]
            ui_gliner_policy = st.selectbox("Политика GLiNER", gliner_policy_options, index=_select_idx(gliner_policy_options, _cfg_value("nlp", "gliner_policy", "missing_only")), key="cfg_gliner_policy")
            ui_gliner_max = st.number_input("Лимит сообщений для GLiNER", min_value=0, max_value=1000000, value=int(_cfg_value("nlp", "gliner_max_messages", 2000)), step=500, key="cfg_gliner_max")

    with tab_payoff:
        pf_col1, pf_col2 = st.columns(2)
        with pf_col1:
            sentiment_mode_options = ["magnitude", "signed_shift", "hybrid"]
            ui_sentiment_mode = st.selectbox("Компонента тональности", sentiment_mode_options, index=_select_idx(sentiment_mode_options, _cfg_value("payoff", "sentiment_mode", "magnitude")), key="cfg_payoff_sentiment_mode")
            ui_hybrid_lambda = st.slider("Доля интенсивности в hybrid-режиме", min_value=0.0, max_value=1.0, value=float(_cfg_value("payoff", "hybrid_lambda", 0.7)), step=0.01, key="cfg_hybrid_lambda")
        with pf_col2:
            ui_w_sentiment = st.slider("Вес тональности в payoff", min_value=0.0, max_value=1.0, value=float(_cfg_value("payoff", "w_sentiment", 0.5)), step=0.01, key="cfg_w_sentiment")
            ui_w_influence = 1.0 - float(ui_w_sentiment)
            metric_card("Вес авторитетности", f"{ui_w_influence:.2f}", "нормируется как 1 - вес тональности", "blue")

    with tab_agent:
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            influence_mode_options = ["dynamic_decayed_pagerank"]
            ui_influence_mode = st.selectbox("Расчёт авторитетности", influence_mode_options, index=0, key="cfg_influence_mode")
            ui_decay = st.slider("Затухание графа за шаг", min_value=0.0, max_value=1.0, value=float(_cfg_value("influence", "decay_per_window", 0.95)), step=0.01, key="cfg_decay")
            ui_pr_alpha = st.slider("PageRank alpha", min_value=0.50, max_value=0.99, value=float(_cfg_value("influence", "pagerank_alpha", 0.85)), step=0.01, key="cfg_pr_alpha")
            ui_pr_weight = st.slider("Вес PageRank в авторитетности", min_value=0.0, max_value=1.0, value=float(_cfg_value("influence", "pagerank_weight", 0.70)), step=0.01, key="cfg_pr_weight")
            ui_pr_max_iter = st.number_input("Максимум итераций PageRank", min_value=10, max_value=1000, value=int(_cfg_value("influence", "pagerank_max_iter", 100)), step=10, key="cfg_pr_max_iter")
            ui_pr_tol = st.number_input("Точность PageRank", min_value=1e-12, max_value=1e-2, value=float(_cfg_value("influence", "pagerank_tol", 0.000001)), format="%.8f", key="cfg_pr_tol")
        with a_col2:
            ui_netlogo_executable = st.text_input("NetLogo Console", value="" if _cfg_value("netlogo", "executable", None) in {None, "None"} else str(_cfg_value("netlogo", "executable", "")), key="cfg_netlogo_executable", help="Можно оставить пустым, если путь определяется автоматически или используется fallback")
            ui_netlogo_model = st.text_input("Файл модели NetLogo", value=str(_cfg_value("netlogo", "model_path", "netlogo/evolutionary_discussion.nlogox")), key="cfg_netlogo_model")
            ui_timeout = st.number_input("Таймаут NetLogo, сек", min_value=10, max_value=3600, value=int(_cfg_value("netlogo", "timeout", 180)), step=10, key="cfg_netlogo_timeout")
            ui_seed = st.number_input("random seed", min_value=0, max_value=1000000, value=int(_cfg_value("netlogo", "random_seed", 42)), step=1, key="cfg_netlogo_seed")
            ui_pop_size = st.number_input("Размер агентной популяции", min_value=50, max_value=100000, value=int(_cfg_value("netlogo", "pop_size", 1000)), step=50, key="cfg_pop_size")
            ui_keep_files = st.checkbox("Сохранять временные файлы NetLogo", value=bool(_cfg_value("netlogo", "keep_files", False)), key="cfg_keep_files")
            ui_compare_steps = st.number_input("Сравнивать следующие шаги дискретизации", min_value=1, max_value=24, value=int(_cfg_value("modeling", "compare_future_windows", 1)), step=1, key="cfg_compare_steps")
            ui_max_continue = st.number_input("Максимальная длина модельной симуляции", min_value=1, max_value=48, value=int(_cfg_value("modeling", "max_continue_windows", 4)), step=1, key="cfg_max_continue")

    with tab_market:
        mk_col1, mk_col2 = st.columns(2)
        with mk_col1:
            ui_market_enabled = st.checkbox("Включить рыночную панель", value=bool(_cfg_value("market", "enabled", True)), key="cfg_market_enabled")
            provider_options = ["yfinance"]
            ui_market_provider = st.selectbox("Провайдер рыночных данных", provider_options, index=0, key="cfg_market_provider")
        with mk_col2:
            ui_market_index = st.text_input("Рыночный ориентир", value=str(_cfg_value("market", "market_index", "SPY")), key="cfg_market_index")
            ui_ticker_cache = st.text_input("Кэш справочника тикеров", value=str(_cfg_value("market", "ticker_universe_cache", "data/market/ticker_universe.csv")), key="cfg_ticker_cache")
            st.caption("Рыночная панель показывает фактическое движение цены вокруг анализируемого шага дискретизации.")

    apply_cfg = st.button("Применить для текущего запуска", type="primary", key="overview_apply_runtime_config")
    save_cfg = st.button("Сохранить в файл конфигурации", key="overview_save_config_file")

    edited_cfg = copy.deepcopy(c.raw)
    edited_cfg.setdefault("project", {})
    edited_cfg["project"].update({
        "name": ui_project_name,
        "window_size_hours": int(ui_step_hours),
        "output_dir": ui_output_dir,
        "data_dir": ui_data_dir,
        "model_dir": ui_model_dir,
    })
    edited_cfg.setdefault("nlp", {})
    edited_cfg["nlp"].update({
        "topic_backend": ui_topic_backend,
        "embedding_model": ui_embedding_model,
        "min_topic_size": int(ui_min_topic_size),
        "nr_topics": _to_number_or_auto(ui_nr_topics),
        "simulation_topic_mode": ui_sim_topic_mode,
        "simulation_topic_count": int(ui_sim_topic_count),
        "indicator_keep_detailed_topics": bool(ui_keep_detailed_topics),
        "sentiment_backend": ui_sentiment_backend,
        "sentiment_policy": ui_sentiment_policy,
        "finbert_model": ui_finbert_model,
        "batch_size": int(ui_batch_size),
        "sentiment_max_length": int(ui_sentiment_max_length),
        "sentiment_device": ui_sentiment_device,
        "sentiment_cache_enabled": bool(ui_sentiment_cache),
        "ticker_extraction": ui_ticker_mode,
        "ticker_batch_size": int(ui_ticker_batch_size),
        "gliner_model": ui_gliner_model,
        "gliner_threshold": float(ui_gliner_threshold),
        "gliner_policy": ui_gliner_policy,
        "gliner_max_messages": int(ui_gliner_max),
    })
    edited_cfg.setdefault("payoff", {})
    edited_cfg["payoff"].update({
        "sentiment_mode": ui_sentiment_mode,
        "hybrid_lambda": float(ui_hybrid_lambda),
        "w_sentiment": float(ui_w_sentiment),
        "w_influence": float(ui_w_influence),
    })
    edited_cfg.setdefault("influence", {})
    edited_cfg["influence"].update({
        "mode": ui_influence_mode,
        "decay_per_window": float(ui_decay),
        "pagerank_alpha": float(ui_pr_alpha),
        "pagerank_weight": float(ui_pr_weight),
        "pagerank_max_iter": int(ui_pr_max_iter),
        "pagerank_tol": float(ui_pr_tol),
        "min_edge_weight": float(_cfg_value("influence", "min_edge_weight", 0.0001)),
        "edge_score_weight": float(_cfg_value("influence", "edge_score_weight", 0.10)),
    })
    edited_cfg.setdefault("netlogo", {})
    edited_cfg["netlogo"].update({
        "executable": None if str(ui_netlogo_executable).strip() == "" else str(ui_netlogo_executable).strip(),
        "model_path": ui_netlogo_model,
        "timeout": int(ui_timeout),
        "random_seed": int(ui_seed),
        "pop_size": int(ui_pop_size),
        "keep_files": bool(ui_keep_files),
    })
    edited_cfg.setdefault("modeling", {})
    edited_cfg["modeling"].update({
        "compare_future_windows": int(ui_compare_steps),
        "max_continue_windows": int(ui_max_continue),
        "nrmse_threshold": float(_cfg_value("modeling", "nrmse_threshold", 35.0)),
        "rmse_threshold": float(_cfg_value("modeling", "rmse_threshold", 0.15)),
        "l1_threshold": float(_cfg_value("modeling", "l1_threshold", 0.35)),
        "reliability_threshold": float(_cfg_value("modeling", "reliability_threshold", 0.65)),
        "prolonged_min_windows": int(_cfg_value("modeling", "prolonged_min_windows", 3)),
        "coverage_threshold": float(_cfg_value("modeling", "coverage_threshold", 0.65)),
        "emerging_share_threshold": float(_cfg_value("modeling", "emerging_share_threshold", 0.30)),
    })
    edited_cfg.setdefault("market", {})
    edited_cfg["market"].update({
        "enabled": bool(ui_market_enabled),
        "provider": ui_market_provider,
        "market_index": ui_market_index,
        "ticker_universe_cache": ui_ticker_cache,
    })

    if apply_cfg:
        st.session_state["runtime_config_override"] = edited_cfg
        st.session_state["overview_config_message"] = "Параметры применены для текущего запуска интерфейса"
        st.rerun()
    if save_cfg:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(yaml.safe_dump(edited_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        st.session_state["runtime_config_override"] = edited_cfg
        st.session_state["overview_config_message"] = f"Файл конфигурации сохранён: {CONFIG_PATH}"
        st.rerun()
    if st.session_state.get("overview_config_message"):
        st.success(st.session_state.pop("overview_config_message"))

    with st.expander("Технический просмотр итоговой конфигурации", expanded=False):
        st.code(yaml.safe_dump(edited_cfg, allow_unicode=True, sort_keys=False), language="yaml")

elif page == "Загрузка данных для моделирования":
    section_header("Загрузка данных для моделирования", "Выбор CSV-файла, временного интервала и базового шага дискретизации", term="шаг дискретизации")
    info_panel(
        "Что происходит на этом шаге",
        "На этом шаге выбирается CSV-файл и временной интервал, который будет использоваться в дальнейшей калибровке, моделировании дискуссии и расчёте предупреждающих индикаторов. Базовый шаг дискретизации - 6 часов, но его можно изменить для текущего запуска.",
        "blue",
    )
    uploaded_model_file = st.file_uploader(
        "Выберите CSV-файл с публикациями",
        type=["csv"],
        key="model_data_upload",
        help="Файл сохраняется локально и используется как источник данных для текущего запуска комплекса.",
    )
    if uploaded_model_file is not None:
        try:
            saved_uploaded_path = save_uploaded_raw_csv(uploaded_model_file, c)
            st.session_state["model_data_raw_path"] = str(saved_uploaded_path)
            st.success(f"Файл загружен: {saved_uploaded_path}")
        except Exception as exc:
            st.error(f"Не удалось сохранить файл: {exc}")
    default_raw_path = st.session_state.get("model_data_raw_path") or str(resolve_raw_dataset_path(c, state) or state.get("raw_path", "") or "data/raw/2024wallstreetbets.csv")
    col1, col2 = st.columns([1.55, 1])
    with col1:
        raw_path = st.text_input("Путь к исходному CSV", default_raw_path, key="data_raw_path_input")
        st.caption("На этой вкладке задаются только источник данных и временной интервал. Модельные параметры изменяются во вкладке «Обзор». ")
    with col2:
        window_hours = st.number_input("Размер шага дискретизации, часов", min_value=1, max_value=168, value=window_hours_state, step=1)
    sentiment_mode = str(c.get("payoff", "sentiment_mode", default="magnitude"))
    w_sent = default_w_sentiment(c, state)
    w_inf = max(0.0, 1.0 - float(w_sent))
    random_state = int(c.get("netlogo", "random_seed", default=42))

    st.markdown("##### Тематическое пространство для текущей подготовки")
    st.caption(
        "Здесь задаётся, сколько крупных тематических групп будет использоваться как стратегии агентов "
        "в основном режиме симуляции. Детальные темы BERTopic при этом сохраняются отдельно для "
        "расчёта и интерпретации предупреждающих индикаторов."
    )
    _topic_mode_current = str(c.get("nlp", "simulation_topic_mode", default="aggregate") or "aggregate")
    _topic_mode_labels = {
        "aggregate": "Укрупнить темы для агентной симуляции",
        "detailed": "Использовать детальные темы напрямую",
    }
    _topic_mode_options = list(_topic_mode_labels.keys())
    _topic_mode_index = _topic_mode_options.index(_topic_mode_current) if _topic_mode_current in _topic_mode_options else 0
    tm1, tm2, tm3 = st.columns([1.25, 1, 1])
    with tm1:
        data_sim_topic_mode = st.selectbox(
            "Режим тем",
            _topic_mode_options,
            index=_topic_mode_index,
            format_func=lambda x: _topic_mode_labels.get(x, str(x)),
            key="data_sim_topic_mode",
            help="В основном режиме лучше использовать укрупнение до 4-8 групп, чтобы агентная модель не работала с сотнями мелких тем.",
        )
    with tm2:
        data_sim_topic_count = st.slider(
            "Крупных групп для симуляции",
            min_value=4,
            max_value=8,
            value=int(c.get("nlp", "simulation_topic_count", default=6)),
            step=1,
            key="data_sim_topic_count",
            disabled=(data_sim_topic_mode == "detailed"),
        )
    with tm3:
        data_min_topic_size = st.number_input(
            "Минимальный размер детальной темы",
            min_value=2,
            max_value=10000,
            value=int(c.get("nlp", "min_topic_size", default=20)),
            step=1,
            key="data_min_topic_size",
        )
    st.info(
        "После подготовки активные крупные группы будут видны в карточке состояния проекта, "
        "а соответствие крупных групп и детальных тем - в таблице «Связь крупных групп и детальных тем» ниже."
    )
    data_cfg_raw = copy.deepcopy(c.raw)
    data_cfg_raw.setdefault("nlp", {})
    data_cfg_raw["nlp"]["simulation_topic_mode"] = str(data_sim_topic_mode)
    data_cfg_raw["nlp"]["simulation_topic_count"] = int(data_sim_topic_count)
    data_cfg_raw["nlp"]["indicator_keep_detailed_topics"] = True
    data_cfg_raw["nlp"]["min_topic_size"] = int(data_min_topic_size)
    c_data = AppConfig(data_cfg_raw)

    raw_path_candidate = Path(str(raw_path).strip()).expanduser()
    if raw_path_candidate.exists() and raw_path_candidate.is_file():
        render_raw_dataset_preview(raw_path_candidate, key_prefix="model_data_preview")
        try:
            interval_meta_df = _raw_interval_metadata(raw_path_candidate)
            if not interval_meta_df.empty:
                min_ts, max_ts = _default_interval_bounds_from_meta(interval_meta_df)
                if st.session_state.get("model_data_interval_source") != str(raw_path_candidate):
                    st.session_state["model_data_start_date"] = min_ts.date()
                    st.session_state["model_data_start_hour"] = int(min_ts.hour)
                    st.session_state["model_data_end_date"] = max_ts.date()
                    st.session_state["model_data_end_hour"] = int(max_ts.hour)
                    st.session_state["model_data_interval_source"] = str(raw_path_candidate)
                st.markdown("##### Интервал данных для моделирования")
                i1, i2 = st.columns(2)
                with i1:
                    data_start_date = st.date_input("Начало интервала", key="model_data_start_date")
                    data_start_hour = st.number_input("Час начала", min_value=0, max_value=23, step=1, key="model_data_start_hour")
                with i2:
                    data_end_date = st.date_input("Конец интервала", key="model_data_end_date")
                    data_end_hour = st.number_input("Час конца", min_value=0, max_value=23, step=1, key="model_data_end_hour")
                st.caption("Правая граница интервала не включается. Значение по умолчанию ставится на следующий целый час после последней публикации, чтобы последняя публикация не терялась.")
                interval_start = pd.Timestamp(data_start_date) + pd.Timedelta(hours=int(data_start_hour))
                interval_end = pd.Timestamp(data_end_date) + pd.Timedelta(hours=int(data_end_hour))
                if interval_end <= interval_start:
                    exact_count, total_valid = 0, int(len(interval_meta_df))
                    st.error("Конец интервала должен быть позже начала интервала")
                else:
                    ts_interval = pd.to_datetime(interval_meta_df["timestamp"], errors="coerce")
                    in_interval_mask = ((ts_interval >= interval_start) & (ts_interval < interval_end)).fillna(False)
                    exact_count = int(in_interval_mask.sum())
                    total_valid = int(len(interval_meta_df))
                st.session_state["model_data_interval"] = {
                    "start": interval_start.isoformat(),
                    "end": interval_end.isoformat(),
                    "count": int(exact_count),
                    "total_valid": int(total_valid),
                    "raw_path": str(raw_path_candidate),
                }
                metric_card("Публикаций в выбранном интервале", str(exact_count), f"точный подсчёт по файлу; пригодных публикаций всего: {total_valid}", "green" if exact_count > 0 else "orange")
            else:
                st.warning("В файле не удалось распознать публикации с корректной датой и непустым текстом")
        except Exception as exc:
            st.warning(f"Не удалось построить интервал по файлу: {exc}")
    else:
        st.info("Выберите файл или укажите существующий путь к CSV")

    info_panel(
        "Подготовка выбранных данных",
        "После выбора файла и интервала можно запустить обработку: очистку текста, расчёт тональности, извлечение финансовых инструментов, тематическое моделирование, payoff и разбиение на шаги дискретизации. Настройки тематического пространства задаются прямо на этой вкладке, остальные параметры берутся из текущей конфигурации.",
        "green",
    )

    if st.button("Запустить подготовку данных", type="primary"):
        selected_limit = None
        selected_sample = None
        try:
            raw_for_initialize = Path(str(raw_path).strip()).expanduser()
            interval_meta = st.session_state.get("model_data_interval") or {}
            if interval_meta.get("raw_path") == str(raw_for_initialize) and interval_meta.get("start") and interval_meta.get("end"):
                try:
                    start_for_filter = pd.Timestamp(interval_meta["start"])
                    end_for_filter = pd.Timestamp(interval_meta["end"])
                    raw_for_initialize, filtered_count, total_valid = _materialize_interval_raw_csv(
                        raw_for_initialize,
                        start_for_filter,
                        end_for_filter,
                        c_data,
                    )
                    interval_meta["count"] = int(filtered_count)
                    interval_meta["total_valid"] = int(total_valid)
                    st.session_state["model_data_interval"] = interval_meta
                    st.info(
                        f"Для подготовки используется выбранный интервал: {filtered_count} публикаций "
                        f"из {total_valid} пригодных публикаций файла"
                    )
                except Exception as exc:
                    st.error(f"Не удалось применить выбранный интервал к подготовке данных: {exc}")
                    st.stop()
            else:
                st.error("Сначала выберите корректный файл и интервал данных. Подготовка всего файла без выбранного интервала не запускается.")
                st.stop()
            with st.spinner("Инициализация выполняется..."):
                result = initialize_project(
                    c_data,
                    raw_path=raw_for_initialize,
                    limit=selected_limit,
                    sample=selected_sample,
                    random_state=int(random_state),
                    window_hours=int(window_hours),
                    w_sentiment=float(w_sent),
                    w_influence=float(w_inf),
                    sentiment_mode=sentiment_mode,
                    progress=progress_writer(),
                )
            st.success("Инициализация завершена")
            a, b, d, e = st.columns(4)
            with a: metric_card("Сообщений", str(result.get("n_messages")), "После фильтрации", "blue")
            with b: metric_card("Шагов дискретизации", str(result.get("n_windows")), f"Размер шага дискретизации: {result.get('window_hours')} ч", "green")
            with d: metric_card("Тем", str(result.get("n_topics")), f"Backend: {result.get('topic_backend')}", "orange")
            with e: metric_card("Payoff", f"{result.get('w_sentiment'):.2f}/{result.get('w_influence'):.2f}", "тональность / графовая авторитетность", "gray")
            timings = result.get("stage_timings") or {}
            if timings:
                st.markdown("#### Время этапов")
                t1, t2, t3 = st.columns(3)
                sent_meta = result.get("sentiment_meta") or {}
                sent_caption = sent_meta.get("source", "batch/cache")
                if sent_meta.get("cache_enabled"):
                    sent_caption = f"{sent_caption}; кэш {sent_meta.get('cache_hits', 0)}/{int(sent_meta.get('cache_hits', 0)) + int(sent_meta.get('cache_misses', 0))}"
                with t1: metric_card("Тональность", format_seconds(timings.get("sentiment_seconds")), str(sent_caption), "blue")
                with t2: metric_card("Тикеры", format_seconds(timings.get("ticker_seconds")), f"mode: {result.get('ticker_extraction_mode', '-')}", "green")
                with t3: metric_card("Всего NLP", format_seconds(float(timings.get("sentiment_seconds", 0)) + float(timings.get("ticker_seconds", 0))), "sentiment + tickers", "gray")
        except Exception as exc:
            st.exception(exc)

    st.subheader("Состояние проекта")
    if state:
        a, b, d, e = st.columns(4)
        with a: metric_card("Сообщений", str(state.get("n_messages")), "В обработанном корпусе", "blue")
        with b: metric_card("Шагов дискретизации", str(state.get("n_windows")), f"По {state.get('window_hours', c.window_hours)} ч", "green")
        with d: metric_card("Крупных групп", str(state.get("n_topics")), f"режим: {state.get('simulation_topic_mode', '-')}", "orange")
        with e: metric_card("Детальных тем", str(state.get("n_detailed_topics", state.get("n_topics"))), "сохранены для индикаторов", "gray")
        st.caption(
            "Крупные группы используются в агентной симуляции как стратегии агентов. "
            "Детальные темы сохраняются отдельно и применяются для интерпретации тематической экспозиции и предупреждающих индикаторов."
        )
        timings = state.get("stage_timings") or {}
        if timings:
            sent_meta = state.get("sentiment_meta") or {}
            sent_extra = f", источник тональности - {sent_meta.get('source', sent_meta.get('backend', '-'))}" if sent_meta else ""
            st.caption(f"Последняя инициализация: тональность - {format_seconds(timings.get('sentiment_seconds'))}{sent_extra}, тикеры - {format_seconds(timings.get('ticker_seconds'))}, режим тикеров - {state.get('ticker_extraction_mode', '-') }.")

        topic_mapping_path = Path(str(state.get("topic_mapping") or paths.get("topic_mapping", c.data_dir / "processed" / "topic_mapping.csv")))
        topic_info_path = Path(str(state.get("topic_info") or paths.get("topic_info", c.data_dir / "processed" / "topic_info.csv")))
        topic_info_detailed_path = Path(str(state.get("topic_info_detailed") or paths.get("topic_info_detailed", c.data_dir / "processed" / "topic_info_detailed.csv")))
        if topic_mapping_path.exists() or topic_info_path.exists() or topic_info_detailed_path.exists():
            st.markdown("#### Темы последней подготовки")
            t_tabs = st.tabs(["Крупные группы", "Связь крупных групп и детальных тем", "Детальные темы"])
            with t_tabs[0]:
                if topic_info_path.exists():
                    try:
                        active_info = pd.read_csv(topic_info_path)
                        st.dataframe(active_info.head(100), width="stretch", hide_index=True)
                        download_dataframe_button(active_info, "topic_info.csv", "Скачать крупные группы", "download_active_topic_info")
                    except Exception as exc:
                        st.warning(f"Не удалось прочитать крупные группы: {exc}")
                else:
                    st.info("Файл с крупными группами ещё не создан. Запустите подготовку данных.")
            with t_tabs[1]:
                if topic_mapping_path.exists():
                    try:
                        mapping = pd.read_csv(topic_mapping_path)
                        st.caption("В этой таблице видно, какие детальные темы были отнесены к каждой крупной группе агентной симуляции.")
                        st.dataframe(mapping.head(300), width="stretch", hide_index=True)
                        download_dataframe_button(mapping, "topic_mapping.csv", "Скачать связь тем", "download_topic_mapping")
                    except Exception as exc:
                        st.warning(f"Не удалось прочитать связь тем: {exc}")
                else:
                    st.info("Связь крупных групп и детальных тем появится после подготовки данных.")
            with t_tabs[2]:
                if topic_info_detailed_path.exists():
                    try:
                        detailed_info = pd.read_csv(topic_info_detailed_path)
                        st.dataframe(detailed_info.head(200), width="stretch", hide_index=True)
                        download_dataframe_button(detailed_info, "topic_info_detailed.csv", "Скачать детальные темы", "download_detailed_topic_info")
                    except Exception as exc:
                        st.warning(f"Не удалось прочитать детальные темы: {exc}")
                else:
                    st.info("Детальные темы появятся после подготовки данных.")
    else:
        st.warning("Проект ещё не инициализирован.")

    if paths["messages"].exists():
        st.subheader("Пример обработанных сообщений")
        cols = ["timestamp", "author", "text_clean", "sentiment", "sentiment_intensity", "influence", "topic", "topic_detailed", "topic_indicator", "topic_modeling", "tickers", "ticker_mentions"]
        preview = pd.read_csv(paths["messages"], nrows=100)
        st.dataframe(preview[[x for x in cols if x in preview.columns]], width="stretch", hide_index=True)
        with st.expander("Выгрузка данных этапа", expanded=False):
            st.caption("Можно скачать основные артефакты последней инициализации для внешнего анализа.")
            export_cols = st.columns(3)
            with export_cols[0]:
                download_dataframe_button(pd.read_csv(paths["messages"]), "messages_enriched.csv", "Скачать сообщения", "download_enriched_messages")
            if paths["payoff"].exists():
                with export_cols[1]:
                    download_dataframe_button(pd.read_csv(paths["payoff"]), "payoff.csv", "Скачать payoff", "download_payoff")
            if paths["windows"].exists():
                try:
                    windows_export = pd.DataFrame(load_windows(paths["windows"]))
                    with export_cols[2]:
                        download_dataframe_button(windows_export, "windows.csv", "Скачать шаги дискретизации", "download_windows")
                except Exception:
                    pass

elif page == "Моделирование дискуссии":
    st.header("Моделирование дискуссии")
    info_panel(
        "Как задаётся сравнение",
        f"Начало выбранного временного интервала задаёт стартовый шаг дискретизации Wₜ. Конец интервала задаёт, на сколько последующих шагов дискретизации продолжать сравнение модельной симуляции с фактом. При шаге {window_hours_state} часов каждая единица горизонта соответствует следующим {window_hours_state} часам.",
        "green",
    )
    if not paths["windows"].exists() or not paths["payoff"].exists():
        st.error("Сначала выполните инициализацию: нужны windows.json и payoff.csv.")
    else:
        windows = load_windows(paths["windows"])
        model_interval_indices, model_interval_start, model_interval_end, model_interval_posts = _select_discretization_interval(
            windows, key_prefix="modeling", step_hours=window_hours_state, require_future=True
        )
        if not model_interval_indices:
            st.stop()
        # В этом режиме временной интервал управляет самой логикой запуска:
        # начало интервала задаёт стартовый шаг дискретизации W_t, а конец
        # интервала задаёт, насколько далеко продолжать сравнение с фактом
        start_index = int(min(model_interval_indices))
        max_future = max(1, len(windows) - int(start_index) - 1)
        compare_steps = max(1, min(int(len(model_interval_indices)), int(max_future)))
        st.session_state["model_start_index"] = int(start_index)
        st.session_state["model_compare_steps"] = int(compare_steps)
        calibrated_defaults = load_fixed_model_params(c, paths["best_params"])
        info_panel(
            "Как работает выбранный интервал",
            "Начало временного интервала определяет стартовый шаг дискретизации для агентной симуляции. Конец интервала определяет, до какого последующего шага дискретизации продолжать сравнение модельной симуляции с фактическим распределением тем. Отдельный ручной выбор стартового шага здесь намеренно убран, чтобы не было расхождения между интервалом и запуском.",
            "green",
        )
        interval_cols = st.columns(3)
        with interval_cols[0]:
            metric_card("Стартовый шаг Wₜ", str(start_index), str(windows[start_index].get("time", "-")), "blue")
        with interval_cols[1]:
            metric_card("Продолжить сравнение", f"+1 ... +{int(compare_steps)}", f"до {int(compare_steps) * window_hours_state} ч вперёд", "green")
        with interval_cols[2]:
            metric_card("Публикаций в интервале", str(model_interval_posts), "по выбранным шагам дискретизации", "gray")
        info_panel(
            "Начальные параметры",
            "Поля ниже автоматически заполняются по последней калибровке, если файл best_params.json найден. Пользователь может вручную изменить prob_revision, alpha, noise и веса payoff для текущего запуска.",
            "green" if calibrated_defaults.get("is_calibrated") else "orange",
        )
        st.caption(f"Источник параметров: {calibrated_defaults.get('source')}")
        st.subheader("Параметры моделирования")
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Старт задан интервалом", f"Wₜ = {int(start_index)}", "из поля «Начало интервала»", "blue")
            metric_card("Горизонт задан интервалом", f"{int(compare_steps)} шаг(а)", "из поля «Конец интервала»", "green")
        with col2:
            prob_revision = st.number_input(
                "prob_revision",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("model_pr", calibrated_defaults.get("prob_revision", 0.03))),
                step=0.001,
                format="%.4f",
                help="Вероятность пересмотра темы агентом за один шаг NetLogo. 0 = темы почти не меняются, 1 = пересмотр максимально активный.",
            )
            noise = st.number_input(
                "noise",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("model_noise", calibrated_defaults.get("noise", 0.4))),
                step=0.001,
                format="%.4f",
                help="Случайность поведения агентов. Чем выше значение, тем менее детерминированно агенты следуют payoff.",
            )
            st.session_state["model_pr"] = float(prob_revision)
            st.session_state["model_noise"] = float(noise)
        with col3:
            use_alpha = st.checkbox("Использовать alpha", value=bool(st.session_state.get("model_alpha_on", calibrated_defaults.get("alpha") is not None)), help="Если выключено, используется unweighted-режим NetLogo.")
            st.session_state["model_alpha_on"] = bool(use_alpha)
            alpha = st.number_input(
                "alpha",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("model_alpha", calibrated_defaults.get("alpha") if calibrated_defaults.get("alpha") is not None else 0.2)),
                step=0.001,
                format="%.4f",
                help="Смешивает собственный payoff темы и взаимодействие с текущим распределением тем. 0 = сильнее окружение, 1 = сильнее собственный payoff.",
            ) if use_alpha else None
            if alpha is not None:
                st.session_state["model_alpha"] = float(alpha)

        with st.expander("Веса payoff для этого запуска", expanded=False):
            sentiment_mode_run = st.selectbox(
                "Компонента тональности",
                ["magnitude", "signed_shift", "hybrid"],
                index=sentiment_mode_index(calibrated_defaults.get("sentiment_mode", state.get("sentiment_mode", c.get("payoff", "sentiment_mode", default="magnitude")))),
                key="model_sentiment_mode",
                help="magnitude учитывает силу эмоции, signed_shift - позитивность, hybrid - смешанный режим.",
            )
            w_sent_run, w_inf_run = weight_inputs(
                "Базово используется последний вес инициализации. При запуске payoff пересчитывается для выбранных весов.",
                float(calibrated_defaults.get("w_sentiment", default_w_sentiment(c, state))),
                key="model_weight",
            )

        payoff_for_run = None
        if st.button("Смоделировать выбранный шаг дискретизации", type="primary"):
            try:
                horizons = future_steps(int(compare_steps))
                payoff_for_run = runtime_payoff_path(c, paths, w_sent_run, w_inf_run, sentiment_mode_run, "single_window")
                with st.spinner("NetLogo моделирует динамику..."):
                    out = run_window_modeling(c, paths["windows"], payoff_for_run, int(start_index), horizons, float(prob_revision), alpha, float(noise))
                comp = pd.DataFrame(out["comparisons"])
                if comp.empty:
                    st.warning("Нет будущих реальных шагов дискретизации для сравнения.")
                else:
                    st.session_state["last_window_modeling"] = comp
                    st.session_state["last_window_modeling_key"] = f"W{int(start_index)} / +{int(compare_steps)}"
                    st.success("Моделирование завершено")
            except Exception as exc:
                st.exception(exc)

        comp = st.session_state.get("last_window_modeling")
        if comp is not None and not pd.DataFrame(comp).empty:
            comp = pd.DataFrame(comp)
            st.subheader("Результаты последнего моделирования")
            st.caption(st.session_state.get("last_window_modeling_key", ""))
            show_quality_cards(comp, window_hours_state)
            plot_forecast_errors(comp)
            plot_topic_trajectories(comp, topic_names=topic_name_map(c))
            st.dataframe(compact_forecasts(comp), width="stretch", hide_index=True)
            labels = comp["horizon"].astype(int).map(lambda h: f"+{h} шаг дискретизации / {h * window_hours_state} ч").tolist()
            selected_h = st.selectbox("Подробно показать распределение тем для будущего шага дискретизации", labels, key="topic_distribution_select")
            selected_num = int(str(selected_h).split()[0].replace("+", ""))
            row = comp[comp["horizon"].astype(int) == selected_num].iloc[0]
            plot_topic_distribution(row, f"Распределение тем: Wₜ → Wₜ₊{selected_num}", topic_names=topic_name_map(c))

        st.divider()
        info_panel(
            "Дальше: диагностика и рынок",
            "Массовое сравнение убрано из этой вкладки, чтобы не смешивать ручное моделирование одного шага дискретизации с исследовательской проверкой. Для проверки аналитических событий, публикаций и рыночной реакции используйте раздел «Расчёт предупреждающих индикаторов».",
            "gray",
        )

elif page == "Калибровка":
    st.header("Калибровка параметров")
    info_panel(
        "Что оптимизируем",
        "В калибровке можно выбрать целевую метрику: только nRMSE или nRMSE + L1 с заданным коэффициентом. nRMSE измеряет среднеквадратичную ошибку тематических долей в процентах, а L1 дополнительно штрафует суммарное расхождение распределений по темам.",
        "blue",
    )
    if not paths["windows"].exists() or not paths["payoff"].exists():
        st.error("Сначала выполните инициализацию.")
    else:
        windows = load_windows(paths["windows"])
        cal_interval_indices, cal_interval_start, cal_interval_end, cal_interval_posts = _select_discretization_interval(
            windows, key_prefix="calibration", step_hours=window_hours_state, require_future=True
        )
        if not cal_interval_indices:
            st.stop()
        calibration_windows_path = _write_temp_windows_subset(windows, cal_interval_indices, c, "calibration")
        calibration_windows_count = len(cal_interval_indices)
        max_possible_steps = max(1, calibration_windows_count - 1)
        method = st.radio("Метод подбора", ["Optuna", "Grid search"], horizontal=True, help="Optuna удобнее для гибкого поиска; grid search полезен для маленьких понятных сеток.")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            compare_steps_cal = st.number_input(
                "Сравнивать на N следующих шагов",
                min_value=1,
                max_value=max_possible_steps,
                value=min(2, max_possible_steps),
                step=1,
                help="Качество считается для ближайшего будущего шага дискретизации и продолжения до N шагов дискретизации. При шаге дискретизации 6 часов N=4 означает проверку до 24 часов вперёд.",
            )
        with col_b:
            max_windows = st.number_input(
                "Шагов дискретизации для оценки",
                min_value=1,
                max_value=max(1, calibration_windows_count - int(compare_steps_cal)),
                value=min(10, max(1, calibration_windows_count - int(compare_steps_cal))),
                step=1,
                help="Сколько стартовых шагов дискретизации использовать для оценки каждой комбинации параметров. Чем больше, тем надёжнее и дольше.",
            )
        with col_c:
            n_jobs = st.number_input(
                "n_jobs для Optuna",
                min_value=1,
                max_value=16,
                value=1,
                step=1,
                help="Сколько trial запускать параллельно. Увеличивайте осторожно: каждый trial запускает отдельный NetLogo-процесс, поэтому нагрузка на CPU и память растёт.",
            )
        if int(n_jobs) > 1:
            st.info("При n_jobs > 1 прогресс отдельных trial не выводится в реальном времени: Streamlit нельзя безопасно обновлять из параллельных потоков. Результаты появятся после завершения подбора.")
        st.caption("Все числовые параметры имеют ограниченные допустимые диапазоны. Если данных мало, верхние границы количества шагов дискретизации автоматически уменьшаются.")

        metric_col1, metric_col2 = st.columns([1.4, 1])
        with metric_col1:
            calibration_metric_choice = st.radio(
                "Целевая метрика калибровки",
                ["Только nRMSE", "nRMSE + L1"],
                horizontal=True,
                key="calibration_objective_metric_choice",
                help="nRMSE использует нормировку на размах распределения тем. Режим nRMSE + L1 дополнительно учитывает суммарное отличие тематических распределений.",
            )
        with metric_col2:
            calibration_l1_weight = st.number_input(
                "Коэффициент L1",
                min_value=0.0,
                max_value=1000.0,
                value=float(c.get("calibration", "objective_l1_weight", default=25.0)),
                step=1.0,
                disabled=(calibration_metric_choice == "Только nRMSE"),
                key="calibration_objective_l1_weight",
                help="Используется только в режиме nRMSE + L1. Чем больше коэффициент, тем сильнее калибровка штрафует L1-расхождение распределений.",
            )
        calibration_objective_metric = "nrmse_l1" if calibration_metric_choice == "nRMSE + L1" else "nrmse"
        calibration_cfg_raw = copy.deepcopy(c.raw)
        calibration_cfg_raw.setdefault("calibration", {})
        calibration_cfg_raw["calibration"]["objective_metric"] = calibration_objective_metric
        calibration_cfg_raw["calibration"]["objective_nrmse_weight"] = 1.0
        calibration_cfg_raw["calibration"]["objective_l1_weight"] = float(calibration_l1_weight)
        c_calibration = AppConfig(calibration_cfg_raw)
        if calibration_objective_metric == "nrmse_l1":
            st.info(f"Калибровка минимизирует objective = nRMSE + {float(calibration_l1_weight):g} · L1")
        else:
            st.info("Калибровка минимизирует только nRMSE. L1, RMSE и другие метрики сохраняются в таблице результатов как диагностические поля, но не влияют на выбор лучшего набора параметров.")

        if method == "Optuna":
            st.subheader("Границы поиска Optuna")
            c1, c2, c3 = st.columns(3)
            with c1:
                pr_min, pr_max = range_slider(
                    "Диапазон prob_revision",
                    0.0,
                    1.0,
                    (0.005, 0.15),
                    0.001,
                    "optuna_pr_range",
                    "Вероятность пересмотра темы агентом. Малые значения дают инерционную дискуссию, большие - быструю смену тем.",
                )
            with c2:
                alpha_min, alpha_max = range_slider(
                    "Диапазон alpha",
                    0.0,
                    1.0,
                    (0.0, 1.0),
                    0.001,
                    "optuna_alpha_range",
                    "Баланс между собственным payoff темы и влиянием текущего распределения. Optuna также может выбрать unweighted-режим.",
                )
            with c3:
                noise_min, noise_max = range_slider(
                    "Диапазон noise",
                    0.0,
                    1.0,
                    (0.0, 0.8),
                    0.001,
                    "optuna_noise_range",
                    "Случайность поведения агентов. Высокий noise снижает детерминированность переходов.",
                )
            c4, c5, c6 = st.columns(3)
            with c4:
                n_trials = st.number_input("Число trial", min_value=1, max_value=100000, value=20, step=10, help="Сколько наборов параметров попробует Optuna. Большие значения могут запускаться очень долго, потому что каждый trial вызывает NetLogo.")
            with c5:
                tune_weights = st.checkbox("Подбирать веса payoff", value=True, help="Если включено, Optuna будет подбирать w_sentiment, а w_influence = 1 - w_sentiment.")
            with c6:
                allow_unweighted_alpha = st.checkbox("Разрешить режим без alpha", value=False, help="Если выключено, Optuna всегда использует weighted alpha в указанном диапазоне. Это нужно для экспериментов, где alpha задана явно.")

            c7, c8 = st.columns(2)
            with c7:
                sentiment_mode_cal = st.selectbox("Компонента тональности", ["magnitude", "signed_shift", "hybrid"], index=0, key="cal_sentiment_mode_fixed", help="По умолчанию фиксируется выбранная компонента. Optuna не будет самовольно менять sentiment_mode, если не включён следующий переключатель.")
            with c8:
                tune_sentiment_mode = st.checkbox("Подбирать компоненту тональности", value=False, help="Если включено, Optuna будет выбирать между magnitude, signed_shift и hybrid. Если выключено, используется выбранная компонента.")

            if not tune_weights:
                with st.expander("Фиксированные веса payoff", expanded=True):
                    w_sent_cal, w_inf_cal = weight_inputs("Фиксированные веса для подбора", 0.5, key="cal_fixed_weight")
            else:
                w_sent_cal, w_inf_cal = 0.5, 0.5
            optuna_alpha_mode = "search" if allow_unweighted_alpha else "weighted"

            if st.button("Запустить Optuna-подбор", type="primary"):
                try:
                    payoff_for_cal = paths["payoff"]
                    if not tune_weights:
                        payoff_for_cal = runtime_payoff_path(c, paths, w_sent_cal, w_inf_cal, sentiment_mode_cal, "optuna_fixed")
                    with st.spinner("Optuna подбирает параметры через NetLogo..."):
                        df = calibrate_model_optuna(
                            c_calibration,
                            calibration_windows_path,
                            payoff_for_cal,
                            paths["messages"],
                            horizons=future_steps(int(compare_steps_cal)),
                            n_trials=int(n_trials),
                            max_windows=int(max_windows),
                            pr_min=float(pr_min), pr_max=float(pr_max),
                            alpha_min=float(alpha_min), alpha_max=float(alpha_max),
                            noise_min=float(noise_min), noise_max=float(noise_max),
                            tune_weights=bool(tune_weights),
                            n_jobs=int(n_jobs),
                            progress=progress_writer() if int(n_jobs) == 1 else None,
                            alpha_mode=optuna_alpha_mode,
                            sentiment_mode=sentiment_mode_cal,
                            tune_sentiment_mode=bool(tune_sentiment_mode),
                            fixed_w_sentiment=float(w_sent_cal),
                            fixed_w_influence=float(w_inf_cal),
                        )
                    st.success("Optuna-подбор завершён")
                    if not df.empty:
                        best = df.iloc[0]
                        a, b, d, e = st.columns(4)
                        with a: metric_card("Objective", fmt(best.get("objective")), "целевая метрика", "green")
                        with b: metric_card("nRMSE, %", fmt(best.get("mean_nrmse"), 2) + "%", "уровень распределений", "violet")
                        with d: metric_card("prob_revision", fmt(best.get("prob_revision")), "Вероятность пересмотра", "blue")
                        with e: metric_card("w_sent / w_inf", f"{fmt(best.get('w_sentiment'), 3)} / {fmt(best.get('w_influence'), 3)}", "Веса payoff", "gray")
                    st.dataframe(df, width="stretch", hide_index=True)
                except Exception as exc:
                    st.exception(exc)
        else:
            st.subheader("Grid search")
            info_panel("Как задавать сетку", "Введите значения через запятую. Каждое значение дополнительно проверяется на допустимый диапазон от 0 до 1.", "gray")
            with st.expander("Сетки параметров", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    pr_text = st.text_input("prob_revision grid", "0.01,0.03,0.06", help="Все значения должны быть в диапазоне от 0 до 1.")
                    alpha_text = st.text_input("alpha grid", "none,0.2,0.5", help="Можно указать none для unweighted-режима; остальные значения от 0 до 1.")
                with col2:
                    noise_text = st.text_input("noise grid", "0.2,0.4", help="Все значения должны быть в диапазоне от 0 до 1.")
                    sentiment_mode_grid = st.selectbox("Компонента тональности", ["magnitude", "signed_shift", "hybrid"], index=0, key="grid_sentiment_mode")
                w_sent_grid, w_inf_grid = weight_inputs("Веса payoff для grid search", 0.5, key="grid_weight")
            if st.button("Запустить grid search", type="primary"):
                try:
                    pr_values = parse_float_list(pr_text)
                    alpha_values = parse_float_list(alpha_text, allow_none=True)
                    noise_values = parse_float_list(noise_text)
                    for name, values in {"prob_revision": pr_values, "noise": noise_values}.items():
                        if any((v < 0 or v > 1) for v in values):
                            st.error(f"{name}: все значения должны быть в диапазоне от 0 до 1.")
                            st.stop()
                    if any((v is not None and (v < 0 or v > 1)) for v in alpha_values):
                        st.error("alpha: все числовые значения должны быть в диапазоне от 0 до 1, либо используйте none.")
                        st.stop()
                    payoff_grid = runtime_payoff_path(c, paths, w_sent_grid, w_inf_grid, sentiment_mode_grid, "grid")
                    with st.spinner("Калибровка через NetLogo..."):
                        df = calibrate_model(
                            c_calibration,
                            calibration_windows_path,
                            payoff_grid,
                            prob_revisions=pr_values,
                            alphas=alpha_values,
                            noises=noise_values,
                            horizons=future_steps(int(compare_steps_cal)),
                            max_windows=int(max_windows),
                            progress=progress_writer(),
                        )
                    st.success("Grid search завершён")
                    st.dataframe(df, width="stretch", hide_index=True)
                except Exception as exc:
                    st.exception(exc)

        st.subheader("Последние результаты")
        if paths["optuna_calibration"].exists():
            st.caption("Optuna")
            st.dataframe(pd.read_csv(paths["optuna_calibration"]), width="stretch", hide_index=True)
        if paths["grid_calibration"].exists():
            st.caption("Grid search")
            st.dataframe(pd.read_csv(paths["grid_calibration"]), width="stretch", hide_index=True)

elif page == "Расчёт предупреждающих индикаторов":
    section_header("Расчёт предупреждающих индикаторов", "Один запуск расчёта для выбранного шага дискретизации: отчёт, дискуссия, публикации, социальный граф, рыночная проверка и технические данные", term="предупреждающий дискуссионный индикатор")
    info_panel(
        "Зачем нужен этот раздел",
        "Раздел работает вокруг одного шага дискретизации дискуссии. Сервис моделирует развитие тематической структуры и рассчитывает несколько раздельных индикаторов: текущую статистическую необычность тикерного обсуждения, модельную ожидаемую тематическую поддержку и их комбинированную оценку. Это не частотная популярность и не пользовательская заинтересованность, а приоритет ручной проверки по сдвигу распределения тем, тикерной концентрации, эмоциональной интенсивности и графовой авторитетности участников. Для исторических данных дополнительно выполняется событийная проверка реакции рынка",
        "blue",
    )
    info_panel(
        "Извлечение тикеров",
        "Извлечение идёт не только из встроенного словаря на 33 тикера. Основной режим использует локальный кэш SEC/NASDAQ, cashtag-и вроде $TSLA, проверку символов, сопоставление названий компаний/алиасов и опционально GLiNER-small. Встроенный словарь применяется только как fallback, если кэш отсутствует и открытые справочники недоступны.",
        "green",
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        refresh_universe = st.button("Обновить справочник тикеров", help="Скачивает открытые справочники SEC/NASDAQ и сохраняет локальный кэш. Если интернет недоступен, используется встроенный fallback.")
    if refresh_universe:
        with st.spinner("Обновляю справочник тикеров..."):
            uni = load_ticker_universe(c, refresh=True)
        st.success(f"Справочник обновлён/проверен: {len(uni)} записей · {_ticker_universe_caption(uni)}")
    else:
        try:
            uni = load_ticker_universe(c, refresh=False)
            with col_b:
                metric_card("Тикерный справочник", str(len(uni)), _ticker_universe_caption(uni), "green")
        except Exception as exc:
            uni = pd.DataFrame()
            st.warning(f"Справочник тикеров пока недоступен: {exc}")
    with col_c:
        metric_card("Режим", "текущий эксперимент", "один интервал - один анализ", "blue")


    st.subheader("Выбор шага дискретизации и запуск анализа")
    info_panel(
        "Как читать этот раздел",
        "Выберите текущий шаг дискретизации и количество следующих шагов дискретизации N для анализа траектории дискуссии. Сервис не оценивает будущую цену и не строит торговая рекомендация. Он моделирует развитие обсуждения, подсвечивает аналитические события по темам и тикерам, а для исторических данных добавляет проверку: что реально произошло дальше в дискуссии и на рынке.",
        "green",
    )
    if not paths["windows"].exists() or not paths["payoff"].exists():
        st.warning("Для отчёта нужны windows.json и payoff.csv. Сначала выполните инициализацию.")
    else:
        windows = load_windows(paths["windows"])
        if len(windows) < 2:
            st.warning("Недостаточно шагов дискретизации для расчёта индекса события.")
        else:
            calibrated_params = load_fixed_model_params(c, paths["best_params"])
            st.markdown("#### Параметры текущего эксперимента")
            info_panel(
                "Как используются параметры",
                "Калибровка даёт рекомендуемые alpha, prob_revision и noise, но пользователь может вручную задать параметры для исследовательского эксперимента. Тональность и графовая авторитетность всегда учитываются в payoff; если веса не заданы, используется базовый режим 0.500 / 0.500.",
                "blue",
            )
            param_mode = st.radio(
                "Источник параметров",
                ["Из последней калибровки", "Задать вручную"],
                horizontal=True,
                help="Калиброванные параметры удобны для стандартной проверки. Ручной режим нужен для экспериментов и чувствительности модели.",
                key="market_param_mode",
            )
            if param_mode == "Из последней калибровки" and not calibrated_params.get("is_calibrated"):
                st.warning("Файл калибровки не найден. Ниже используются значения по умолчанию, их можно изменить в ручном режиме.")
            base_pr = float(calibrated_params.get("prob_revision", 0.03))
            base_alpha = calibrated_params.get("alpha")
            base_noise = float(calibrated_params.get("noise", 0.4))
            base_w_sent = float(calibrated_params.get("w_sentiment", 0.5) or 0.5)
            base_w_inf = float(calibrated_params.get("w_influence", 1.0 - base_w_sent) or (1.0 - base_w_sent))
            # если старая калибровка дала веса 0 и 1, показываем это пользователю
            # при этом оставляем быстрый переход к ручному соотношению 0 5 и 0 5
            base_sent_mode = str(calibrated_params.get("sentiment_mode") or c.get("payoff", "sentiment_mode", default="magnitude"))
            if param_mode == "Задать вручную":
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    pr_f = st.number_input(
                            "prob_revision",
                            min_value=0.001,
                            max_value=1.0,
                            value=clamp_float(base_pr, 0.001, 1.0, default=0.03),
                            step=0.001,
                            format="%.4f",
                            help="Вероятность пересмотра темы агентом за один шаг NetLogo. Значение из калибровки автоматически ограничивается допустимым диапазоном 0.001-1.000.",
                        )
                with m2:
                    alpha_text = st.text_input("alpha", value="none" if base_alpha is None else str(base_alpha), help="none - использовать обычную матрицу payoff; число в диапазоне от 0 до 1 - смешивание собственного payoff и окружения.")
                    alpha_f = parse_alpha_value(alpha_text)
                with m3:
                    noise_f = st.number_input(
                            "noise",
                            min_value=0.0,
                            max_value=2.0,
                            value=clamp_float(base_noise, 0.0, 2.0, default=0.4),
                            step=0.01,
                            format="%.3f",
                            help="Стохастичность переходов агентов между темами. Значение из калибровки автоматически ограничивается допустимым диапазоном.",
                        )
                with m4:
                    sentiment_mode_f = st.selectbox("Компонента тональности", ["magnitude", "signed_shift", "hybrid"], index=sentiment_mode_index(base_sent_mode), help="magnitude - сила эмоции |s|; signed_shift - позитивность; hybrid - смесь.")
                w_sent_f, w_inf_f = weight_inputs("Веса payoff текущего эксперимента", 0.5, key="market_manual_weight")
            else:
                pr_f = base_pr
                alpha_f = base_alpha
                noise_f = base_noise
                sentiment_mode_f = base_sent_mode
                w_sent_f, w_inf_f = _normalize_ui_weights(base_w_sent, base_w_inf)
            parameter_rail([
                ("prob_revision", fmt(pr_f, 4), "текущий эксперимент"),
                ("alpha", "none" if alpha_f is None else fmt(alpha_f, 4), "текущий эксперимент"),
                ("noise", fmt(noise_f, 4), "текущий эксперимент"),
                ("веса payoff", f"{fmt(w_sent_f, 3)} / {fmt(w_inf_f, 3)}", "тональность / графовая авторитетность"),
            ])
            st.markdown("#### Источник данных и шаг дискретизации анализа")
            raw_path_for_local: Path | None = None
            raw_path_valid = False
            local_start_time = None
            start_idx = 0
            max_steps_available = max(1, len(windows) - 1)

            source_col, preview_col = st.columns([1.05, 1.35], gap="large")
            with source_col:
                window_mode = st.radio(
                    "Как выбрать шаг дискретизации",
                    ["Из рассчитанных шагов дискретизации", "Локальный диапазон"],
                    horizontal=True,
                    help="Из рассчитанных шагов дискретизации - быстрее. Локальный диапазон читает подключённый raw CSV и пересобирает локальное тематическое пространство по контексту и наблюдаемой части шаги дискретизации.",
                    key="market_window_mode",
                )

                if window_mode == "Из рассчитанных шагов дискретизации":
                    labels = []
                    for i, w in enumerate(windows[:-1]):
                        t0 = pd.to_datetime(w.get("time"), errors="coerce")
                        t1 = t0 + pd.Timedelta(hours=window_hours_state) if not pd.isna(t0) else None
                        labels.append(f"W{i}: {t0} - {t1} · постов: {w.get('n_posts', 0)}")
                    selected_label = st.selectbox(
                        "Шаг дискретизации дискуссии",
                        labels,
                        index=min(int(st.session_state.get("market_forward_start", 0)), max(0, len(labels)-1)),
                        key="market_window_label",
                    )
                    start_idx = int(selected_label.split(":", 1)[0].replace("W", ""))
                    st.session_state["market_forward_start"] = start_idx
                    local_start_time = None
                    max_steps_available = max(1, len(windows) - int(start_idx) - 1)
                    st.caption("Используется уже подготовленное шаг дискретизации из результатов инициализации")
                else:
                    uploaded_raw = st.file_uploader(
                        "Подключить новый CSV к локальному анализу",
                        type=["csv"],
                        key="market_local_raw_upload",
                        help="Файл будет сохранён в data/raw/uploads и сразу подставлен как источник локального шага дискретизации. Поддерживаются колонки text, createdAt, author_name, post_id, parent_id, comment_id и другие алиасы.",
                    )
                    if uploaded_raw is not None:
                        try:
                            saved_raw = save_uploaded_raw_csv(uploaded_raw, c)
                            st.success(f"CSV подключён: {saved_raw}")
                        except Exception as exc:
                            st.error(f"Не удалось подключить CSV: {exc}")
                    resolved_raw_path = resolve_raw_dataset_path(c, state)
                    raw_default = str(resolved_raw_path or state.get("raw_path", "") or "data/raw/2024wallstreetbets.csv")
                    if "market_local_raw_path" not in st.session_state or not str(st.session_state.get("market_local_raw_path", "")).strip():
                        st.session_state["market_local_raw_path"] = raw_default
                    raw_path_value = st.text_input(
                        "Источник сообщений для локального шага дискретизации",
                        key="market_local_raw_path",
                        help="Можно указать сохранённый CSV, загруженный через поле выше, или путь к live-файлу, который обновляется краулером. Поддерживаются разные названия колонок, включая createdAt, author_name, author_id, post_id, parent_id, comment_id, childCount.",
                    )
                    raw_path_for_local = Path(str(raw_path_value).strip()).expanduser()
                    raw_path_valid = raw_path_for_local.exists() and raw_path_for_local.is_file()
                    if raw_path_valid:
                        try:
                            raw_stat = raw_path_for_local.stat()
                            st.caption(f"Источник подключён: {raw_path_for_local} · размер {raw_stat.st_size / 1024 / 1024:.2f} МБ · изменён {pd.to_datetime(raw_stat.st_mtime, unit='s')}")
                        except Exception:
                            st.caption(f"Источник подключён: {raw_path_for_local}")
                    else:
                        st.error("Нужен существующий CSV-файл с сообщениями. Сейчас путь пустой, указывает на папку или файл недоступен")

                    all_times = [pd.to_datetime(w.get("time"), errors="coerce") for w in windows if w.get("time") is not None]
                    default_dt = min([t for t in all_times if not pd.isna(t)]) if all_times else pd.Timestamp.now()
                    d_col, h_col = st.columns([1.2, 0.8])
                    with d_col:
                        d = st.date_input("Дата начала шага дискретизации", value=default_dt.date(), key="market_local_date")
                    with h_col:
                        hh = st.number_input("Час", min_value=0, max_value=23, value=int(default_dt.hour), step=1, key="market_local_hour")
                    local_start_time = pd.Timestamp(d) + pd.Timedelta(hours=int(hh))
                    start_idx = -1
                    max_steps_available = 30
                    st.caption("Темы пересобираются по историческому контексту и наблюдаемой части выбранного шага дискретизации. Будущие сообщения в состояние шага дискретизации не попадают")

                steps = st.number_input(
                    "Смоделировать N следующих шагов дискретизации дискуссии",
                    min_value=1,
                    max_value=max_steps_available,
                    value=min(4, max_steps_available),
                    step=1,
                    key="market_forward_steps",
                    help=f"Если размер шага дискретизации {window_hours_state} ч, то N=4 означает проверку примерно на {4 * window_hours_state} ч вперёд.",
                )

            with preview_col:
                if window_mode == "Локальный диапазон":
                    render_raw_dataset_preview(raw_path_for_local if raw_path_valid else None, key_prefix="market_local_raw_preview")
                else:
                    selected_window = windows[int(start_idx)] if windows and int(start_idx) < len(windows) else {}
                    info_panel(
                        "Данные рассчитанного шага дискретизации",
                        f"Выбрано шаг дискретизации W{int(start_idx)}. Постов в шаге дискретизации: {selected_window.get('n_posts', '-')}. Для просмотра структуры нового CSV переключитесь на локальный диапазон или загрузите файл",
                        "gray",
                    )

            run_col1, run_col2, run_col3 = st.columns([1.05, 1.05, 1.2], gap="large")
            with run_col1:
                context_windows = st.number_input(
                    "Контекст предыдущих шагов дискретизации",
                    min_value=0,
                    max_value=96,
                    value=int(c.get("topic_state", "context_windows", default=8)),
                    step=1,
                    key="market_context_windows",
                    help="Сколько предыдущих шагов дискретизации использовать для устойчивого пересчёта состояния тем и пользовательского графа. Будущие данные не используются.",
                )
                st.session_state["analysis_context_windows"] = int(context_windows)
            with run_col2:
                use_cache = st.checkbox("Использовать кэш локального анализа", value=True, key="market_use_state_cache", help="Если включено, сервис может взять ранее рассчитанное состояние шага дискретизации и кэш тональности. Если выключено, локальный анализ перечитывает CSV и пересчитывает тональность без использования сохранённых оценок.")
            with run_col3:
                if st.button("Очистить кэш анализа", type="secondary", key="clear_signal_analysis_cache", help="Удаляет временные результаты последнего анализа и локальный кэш состояний шагов дискретизации. Полезно, если вы меняли шаг дискретизации, параметры или видите старые публикации."):
                    clear_current_analysis_outputs(paths, c, clear_state_cache=True)
                    st.success("Кэш анализа очищен. Теперь запустите анализ шага дискретизации заново.")

            local_nlp_overrides: dict[str, Any] = {}
            if window_mode == "Локальный диапазон":
                with st.expander("Извлечение тикеров и NER для локального шага дискретизации", expanded=False):
                    st.caption("Быстрый слой всегда использует cashtag, биржевые символы, алиасы и названия компаний. GLiNER добавляет полноценный локальный NER/entity-linking слой без платных API")
                    ner_c1, ner_c2, ner_c3 = st.columns(3)
                    ticker_modes = ["hybrid_fast", "gliner_hybrid", "gliner_full", "none"]
                    current_tm = str(c.get("nlp", "ticker_extraction", default="hybrid_fast"))
                    with ner_c1:
                        local_ticker_mode = st.selectbox(
                            "Режим тикерной привязки",
                            ticker_modes,
                            index=ticker_modes.index(current_tm) if current_tm in ticker_modes else 0,
                            key="market_local_ticker_mode",
                            help="gliner_full запускает GLiNER по всем сообщениям в пределах лимита и подходит для контрольного анализа. hybrid_fast быстрее, но без полноценного NER-слоя",
                        )
                    with ner_c2:
                        local_gliner_policy = st.selectbox(
                            "Политика GLiNER",
                            ["missing_only", "all"],
                            index=0 if str(c.get("nlp", "gliner_policy", default="missing_only")) != "all" else 1,
                            key="market_local_gliner_policy",
                            help="missing_only запускает NER только там, где быстрый слой ничего не нашёл. all уточняет все сообщения. Для gliner_full всегда используется all",
                        )
                    with ner_c3:
                        local_gliner_max = st.number_input(
                            "Лимит GLiNER",
                            min_value=0,
                            max_value=1000000,
                            value=int(c.get("nlp", "gliner_max_messages", default=2000)),
                            step=500,
                            key="market_local_gliner_max",
                            help="0 означает без лимита. Для больших CSV это может быть очень долго",
                        )
                    local_nlp_overrides = {
                        "ticker_extraction": local_ticker_mode,
                        "gliner_policy": "all" if local_ticker_mode == "gliner_full" else local_gliner_policy,
                        "gliner_max_messages": int(local_gliner_max),
                    }

            raw_identity = "precomputed"
            raw_source_key = "precomputed"
            if local_start_time is not None:
                try:
                    resolved_raw = raw_path_for_local.resolve()
                    stat = raw_path_for_local.stat()
                    raw_source_key = str(resolved_raw)
                    raw_identity = f"{resolved_raw}|{stat.st_mtime}|{stat.st_size}"
                except Exception:
                    raw_source_key = str(raw_path_for_local)
                    raw_identity = str(raw_path_for_local)
            # ключ результата зависит от выбранных параметров, а не только от времени изменения файла
            # живой CSV может измениться, пока пользователь переключает разделы
            # уже рассчитанный результат должен оставаться видимым
            # интерфейс только сообщает, что источник изменился и расчёт можно повторить
            nlp_key = json.dumps(local_nlp_overrides, ensure_ascii=False, sort_keys=True)
            current_experiment_key = f"{raw_source_key}|{local_start_time or start_idx}|{window_hours_state}|{steps}|{context_windows}|{pr_f}|{alpha_f}|{noise_f}|{w_sent_f}|{w_inf_f}|{sentiment_mode_f}|{nlp_key}"
            if st.button("Запустить анализ шага дискретизации", type="primary"):
                if local_start_time is not None and not raw_path_valid:
                    st.error("Нельзя запустить локальный анализ: подключите существующий CSV-файл или укажите корректный путь к источнику сообщений")
                    st.stop()
                # все вспомогательные файлы относятся к предыдущему запуску
                # очищаем их перед новым запуском, чтобы публикации, граф и рыночная панель
                # не подтянулись незаметно из другого шага дискретизации
                clear_current_analysis_outputs(paths, c, clear_state_cache=not bool(use_cache))
                try:
                    with st.spinner("NetLogo моделирует развитие дискуссии и формирует индексы событий..."):
                        if local_start_time is not None:
                            run_cfg = c
                            if local_nlp_overrides or not bool(use_cache):
                                raw_cfg = json.loads(json.dumps(c.raw, ensure_ascii=False, default=str))
                                raw_cfg.setdefault("nlp", {}).update(local_nlp_overrides)
                                if not bool(use_cache):
                                    raw_cfg.setdefault("nlp", {})["sentiment_cache_enabled"] = False
                                run_cfg = AppConfig(raw_cfg)
                            fwd = run_predictive_market_indicator_for_raw_range(
                                run_cfg,
                                raw_path_for_local,
                                local_start_time,
                                window_hours_state,
                                int(steps),
                                float(pr_f),
                                alpha_f,
                                float(noise_f),
                                w_sentiment=float(w_sent_f),
                                w_influence=float(w_inf_f),
                                sentiment_mode=sentiment_mode_f,
                                output_path=paths["forward_indicators"],
                                context_windows=int(context_windows),
                                use_cache=bool(use_cache),
                                progress=progress_writer(),
                            )
                        else:
                            payoff_forward = runtime_payoff_path(c, paths, w_sent_f, w_inf_f, sentiment_mode_f, "forward_market")
                            fwd = run_predictive_market_indicator(
                                c,
                                paths["windows"],
                                payoff_forward,
                                int(start_idx),
                                int(steps),
                                float(pr_f),
                                alpha_f,
                                float(noise_f),
                                output_path=paths["forward_indicators"],
                            )
                    st.success("Анализ шага дискретизации рассчитан")
                    st.session_state["last_forward_indicators"] = fwd
                    st.session_state["last_forward_experiment_key"] = current_experiment_key
                    st.session_state["last_forward_raw_identity"] = raw_identity
                except Exception as exc:
                    st.exception(exc)
            fwd_current = st.session_state.get("last_forward_indicators")
            if st.session_state.get("last_forward_experiment_key") != current_experiment_key:
                fwd_current = None
                info_panel("Требуется новый расчёт", "Параметры или шаг дискретизации изменились. Нажмите 'Запустить анализ шага дискретизации', чтобы рассчитать результат именно для текущего выбора.", "blue")
            elif local_start_time is not None and st.session_state.get("last_forward_raw_identity") not in {None, raw_identity}:
                st.info("Источник сообщений изменился после последнего расчёта. Текущий результат оставлен на экране, но для учёта новых сообщений запустите анализ шага дискретизации заново.")
            if fwd_current is not None:
                fwd_current = pd.DataFrame(fwd_current)
                if fwd_current.empty:
                    info_panel("Тикерная привязка отсутствует", "В выбранном шаге дискретизации не найдено тикеров. Сервис может показать динамику тем и публикации, но тикерные аналитические события и рыночная проверка недоступны. Проверьте режим извлечения тикеров или выберите другое шаг дискретизации.", "gray")
                else:
                    render_analysis_context_header(fwd_current, window_hours_state)
                    analysis_sections = [
                        "Отчёт",
                        "Дискуссия",
                        "Публикации",
                        "Социальный граф",
                        "Рыночная проверка",
                        "Технические данные",
                    ]
                    if st.session_state.get("analysis_signal_section") not in analysis_sections:
                        st.session_state["analysis_signal_section"] = "Отчёт"
                    analysis_subtab = st.radio(
                        "Раздел анализа",
                        analysis_sections,
                        horizontal=True,
                        key="analysis_signal_section",
                        help="Выбор раздела сохраняется при изменении параметров внутри страницы. Это заменяет вкладки Streamlit, которые сбрасывались на первый раздел после rerun.",
                    )
                    if analysis_subtab == "Отчёт":
                        render_executive_report(fwd_current, window_hours_state)
                    if analysis_subtab == "Дискуссия":
                        discussion_start_time = fwd_current["start_time"].dropna().iloc[0] if not fwd_current.empty and "start_time" in fwd_current.columns and fwd_current["start_time"].notna().any() else None
                        render_current_window_topic_distribution(c, paths["messages"], paths["forward_indicators"], discussion_start_time, window_hours_state)
                        render_discussion_state_overview(fwd_current)
                        render_forecast_check_text(fwd_current)
                        plot_forward_validation(fwd_current)
                        plot_ticker_validation(fwd_current)
                    _ticker_summary_for_posts = _forward_ticker_summary(fwd_current) if not fwd_current.empty else _empty_ticker_summary()
                    top_tickers_for_posts = [] if _ticker_summary_for_posts.empty or "ticker" not in _ticker_summary_for_posts.columns else _ticker_summary_for_posts.head(3)["ticker"].dropna().astype(str).tolist()
                    if analysis_subtab == "Публикации":
                        try:
                            post_start_time = fwd_current["start_time"].dropna().iloc[0] if not fwd_current.empty and "start_time" in fwd_current.columns else None
                            posts, posts_source = load_publications_for_signal_window(
                                paths["messages"],
                                paths["forward_indicators"],
                                post_start_time,
                                window_hours_state,
                                tickers=top_tickers_for_posts,
                                top_n=18,
                            )
                            st.caption(f"Источник публикаций: {posts_source}. Шаг дискретизации: {post_start_time}.")
                            render_publication_cards(posts, top_tickers_for_posts)
                        except Exception as exc:
                            st.warning(f"Не удалось загрузить публикации шага дискретизации: {exc}")
                    if analysis_subtab == "Социальный граф":
                        st.markdown("#### Социальный граф текущего анализа")
                        st.caption("Граф строится по выбранному шагу дискретизации и контексту предыдущих шагов дискретизации. Удалённые/неизвестные авторы сохраняются как отдельные служебные узлы для каждого сообщения и получают графовую авторитетность 0; так структура старых обсуждений не схлопывается в один искусственный аккаунт.")
                        try:
                            graph_start_time = fwd_current["start_time"].dropna().iloc[0] if not fwd_current.empty and "start_time" in fwd_current.columns else None
                            graph_source = None
                            context_messages_path = Path(paths["forward_indicators"]).with_name(Path(paths["forward_indicators"]).stem + "_context_messages.csv")
                            local_posts_path = Path(paths["forward_indicators"]).with_name(Path(paths["forward_indicators"]).stem + "_publications.csv")
                            if context_messages_path.exists():
                                graph_source = context_messages_path
                            elif paths["messages"].exists():
                                graph_source = paths["messages"]
                            if graph_start_time is not None and graph_source is not None:
                                max_graph_nodes = st.slider("Максимум узлов на графе", min_value=20, max_value=200, value=80, step=10, help="Ограничение нужно, чтобы визуализация не превращалась в облако из тысяч точек.")
                                nodes_df, edges_df = build_user_graph_snapshot_from_messages(
                                    graph_source,
                                    graph_start_time,
                                    window_hours_state,
                                    context_windows=int(st.session_state.get("analysis_context_windows", 0)),
                                    cfg=c,
                                    max_nodes=int(max_graph_nodes),
                                )
                                render_user_graph(nodes_df, edges_df)
                                g1, g2 = st.columns(2)
                                with g1:
                                    download_dataframe_button(nodes_df, "user_graph_nodes.csv", "Скачать узлы графа", "download_graph_nodes")
                                with g2:
                                    download_dataframe_button(edges_df, "user_graph_edges.csv", "Скачать рёбра графа", "download_graph_edges")
                            else:
                                st.info("Сначала рассчитайте анализ шага дискретизации, чтобы появилась дата и источник данных для графа.")
                        except Exception as exc:
                            st.warning(f"Не удалось построить социальный граф: {exc}")

                    if analysis_subtab == "Рыночная проверка":
                        st.markdown("#### Рыночная проверка текущего эксперимента")
                        st.caption("Здесь показано фактическое движение цены вокруг анализируемого шага дискретизации. Дискуссионные индикаторы показываются отдельно от финансового ряда, чтобы не смешивать разные шкалы на одном графике.")

                        render_market_indicator_cards(fwd_current, max_tickers=4)

                        start_ts_for_price = pd.to_datetime(fwd_current["start_time"].dropna().iloc[0], errors="coerce") if "start_time" in fwd_current and fwd_current["start_time"].notna().any() else pd.Timestamp.now()
                        start_ts_for_price = pd.Timestamp(start_ts_for_price)
                        max_step_for_price = int(pd.to_numeric(fwd_current.get("future_step"), errors="coerce").max() or 1) if "future_step" in fwd_current else 1
                        context_steps_for_price = int(st.session_state.get("analysis_context_windows", 0) or 0)
                        if "context_windows" in fwd_current.columns:
                            _context_values = pd.to_numeric(fwd_current["context_windows"], errors="coerce").dropna()
                            if not _context_values.empty:
                                context_steps_for_price = max(context_steps_for_price, int(_context_values.max()))
                        step_end_for_price = start_ts_for_price + pd.Timedelta(hours=int(window_hours_state))
                        horizon_end_for_price = start_ts_for_price + pd.Timedelta(hours=int(window_hours_state) * max_step_for_price)
                        required_display_start = start_ts_for_price - pd.Timedelta(hours=int(window_hours_state) * max(0, context_steps_for_price))
                        required_display_end = max(step_end_for_price, horizon_end_for_price)
                        default_display_start = required_display_start
                        default_display_end = required_display_end

                        # Streamlit хранит значения date_input по ключу
                        # обычный default используется только при первом отображении
                        # сбрасываем интервал рыночной панели при изменении шага анализа, контекста или горизонта
                        price_context_key = f"{pd.Timestamp(start_ts_for_price).isoformat()}|{int(max_step_for_price)}|{int(context_steps_for_price)}|{int(window_hours_state)}"
                        if st.session_state.get("forward_price_context_key") != price_context_key:
                            st.session_state["forward_price_context_key"] = price_context_key
                            _reset_widget_state_value("forward_price_start_date", default_display_start.date())
                            _reset_widget_state_value("forward_price_start_hour", int(default_display_start.hour))
                            _reset_widget_state_value("forward_price_end_date", default_display_end.date())
                            _reset_widget_state_value("forward_price_end_hour", int(default_display_end.hour))
                            st.session_state.pop("last_forward_price_panel", None)

                        ts_col1, ts_col2, ts_col3 = st.columns(3)
                        with ts_col1:
                            price_top_n = st.number_input("Тикеров на графике", min_value=1, max_value=6, value=3, step=1, key="forward_price_top_n")
                        with ts_col2:
                            interval_label = st.selectbox(
                                "Гранулярность рыночного ряда",
                                options=["1 день", "1 час", "30 минут", "15 минут"],
                                index=0,
                                key="forward_price_interval_label",
                                help="Для 6-часовых шагов дискретизации полезны часовые данные. Если Yahoo/yfinance не отдаёт старую внутридневную историю, выберите дневной режим.",
                            )
                            interval_map = {"1 день": "1d", "1 час": "1h", "30 минут": "30m", "15 минут": "15m"}
                            price_interval = interval_map.get(interval_label, "1d")
                        with ts_col3:
                            price_benchmark = st.text_input("Бенчмарк", value=str(c.get("market", "market_index", default="SPY")), key="forward_price_benchmark", help="Бенчмарк не входит в дискуссионный индикатор. Он нужен как рыночный ориентир: тикер двигался вместе с широким рынком или заметно отличался от него.")

                        st.markdown("##### Интервал отображения рыночного ряда")
                        st.caption(
                            "Интервал должен покрывать исторический контекст, анализируемый шаг дискретизации "
                            "и горизонт сравнения с фактическими данными: "
                            f"с {required_display_start} до {required_display_end}."
                        )
                        int_col1, int_col2 = st.columns(2)
                        with int_col1:
                            _set_widget_state_default("forward_price_start_date", default_display_start.date())
                            _set_widget_state_default("forward_price_start_hour", int(default_display_start.hour))
                            start_date = st.date_input("Показывать с даты", key="forward_price_start_date")
                            start_hour = st.number_input("Час начала", min_value=0, max_value=23, step=1, key="forward_price_start_hour")
                        with int_col2:
                            _set_widget_state_default("forward_price_end_date", default_display_end.date())
                            _set_widget_state_default("forward_price_end_hour", int(default_display_end.hour))
                            end_date = st.date_input("Показывать до даты", key="forward_price_end_date")
                            end_hour = st.number_input("Час окончания", min_value=0, max_value=23, step=1, key="forward_price_end_hour")
                        display_start = pd.Timestamp(start_date) + pd.Timedelta(hours=int(start_hour))
                        display_end = pd.Timestamp(end_date) + pd.Timedelta(hours=int(end_hour))
                        interval_too_narrow = display_start > required_display_start or display_end < required_display_end
                        if display_end <= display_start:
                            st.warning("Конец интервала должен быть позже начала. Исправьте даты перед загрузкой ряда.")
                        elif interval_too_narrow:
                            st.error("Выбранный интервал слишком узкий: он должен покрывать исторический контекст, анализируемый шаг дискретизации и горизонт сравнения с фактическими данными.")
                        render_benchmark_explanation(price_benchmark)
                        if price_interval != "1d":
                            st.info("Внутридневные данные Yahoo/yfinance имеют ограниченную историческую глубину. Если ряд не загрузится для старой даты, переключитесь на режим '1 день'.")

                        if st.button("Построить рыночные ряды", key="build_forward_price_panel"):
                            if display_end <= display_start:
                                st.error("Невозможно загрузить ряд: некорректный интервал отображения.")
                            elif interval_too_narrow:
                                st.error("Невозможно загрузить ряд: интервал отображения не покрывает обязательный исторический и будущий диапазон.")
                            else:
                                try:
                                    with st.spinner("Загружаю рыночные временные ряды..."):
                                        panel = load_market_price_panel(
                                            c,
                                            fwd_current,
                                            window_hours_state,
                                            top_n=int(price_top_n),
                                            market_index=price_benchmark,
                                            progress=progress_writer(),
                                            interval=price_interval,
                                            display_start=display_start,
                                            display_end=display_end,
                                        )
                                    st.session_state["last_forward_price_panel"] = panel
                                    if panel is None or pd.DataFrame(panel).empty:
                                        st.warning("Рыночный ряд не загрузился. Проверьте тикеры, интернет и выбранную гранулярность данных.")
                                except Exception as exc:
                                    st.exception(exc)
                        panel = st.session_state.get("last_forward_price_panel")
                        if panel is not None and not pd.DataFrame(panel).empty:
                            render_price_interpretation(pd.DataFrame(panel), fwd_current)
                            plot_market_price_panel(pd.DataFrame(panel), fwd_current, window_hours_state)
                    if analysis_subtab == "Технические данные":
                        st.caption("Техническая детализация скрыта по умолчанию, чтобы основной экран оставался аналитическим отчётом, а не таблицей коэффициентов.")
                        with st.expander("Выгрузка данных текущего эксперимента", expanded=True):
                            package_bytes = build_experiment_package(fwd_current, paths, st.session_state.get("last_forward_price_panel"))
                            st.download_button(
                                "Скачать полный пакет эксперимента",
                                package_bytes,
                                file_name="discussion_market_experiment_package.zip",
                                mime="application/zip",
                                key="download_full_experiment_package",
                                help="Один архив со всеми CSV текущего эксперимента: индикаторы, обработанный срез, локальные темы, публикации, симуляция и рыночные ряды при наличии",
                            )
                            download_dataframe_button(fwd_current, "event_indicators.csv", "Скачать индексы событий", "download_interest_indicators")
                            sidecar_base = Path(paths["forward_indicators"])
                            export_files = [
                                (sidecar_base.with_name(sidecar_base.stem + "_model_forecast.csv"), "model_forecast_no_future.csv", "Скачать модельный симуляция без будущих фактов", "download_model_forecast"),
                                (sidecar_base.with_name(sidecar_base.stem + "_context_messages.csv"), "processed_context_messages.csv", "Скачать обработанный исторический срез", "download_context_messages"),
                                (sidecar_base.with_name(sidecar_base.stem + "_ticker_summary.csv"), "ticker_context_summary.csv", "Скачать сводку тикеров по контексту", "download_ticker_summary"),
                                (sidecar_base.with_name(sidecar_base.stem + "_topic_info.csv"), "local_topic_info.csv", "Скачать локальные темы", "download_topic_info"),
                                (sidecar_base.with_name(sidecar_base.stem + "_context_publications.csv"), "context_publications.csv", "Скачать публикации контекста", "download_context_publications"),
                                (sidecar_base.with_name(sidecar_base.stem + "_publications.csv"), "window_publications.csv", "Скачать публикации шага дискретизации", "download_window_publications"),
                            ]
                            for path, filename, label, key in export_files:
                                if path.exists():
                                    try:
                                        export_df = pd.read_csv(path)
                                        download_dataframe_button(export_df, filename, label, key)
                                    except Exception:
                                        pass
                            panel = st.session_state.get("last_forward_price_panel")
                            if panel is not None and not pd.DataFrame(panel).empty:
                                download_dataframe_button(pd.DataFrame(panel), "market_price_panel.csv", "Скачать рыночные ряды", "download_price_panel")
                        with st.expander("Показать строки индексов событий", expanded=False):
                            st.dataframe(compact_forward_indicators(fwd_current), width="stretch", hide_index=True)


if page != "Обзор":
    author_footer()
