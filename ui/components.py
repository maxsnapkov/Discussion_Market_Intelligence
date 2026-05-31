"""Переиспользуемые HTML-компоненты пользовательского интерфейса

Модуль содержит компактные карточки, заголовки, подсказки и панели
для отображения результатов анализа в Streamlit
"""

from __future__ import annotations

import re
from html import escape
from typing import Any

import streamlit as st

_DASHES = str.maketrans({"-": "-", "-": "-", "-": "-", "-": "-", "-": "-", "-": "-"})

GLOSSARY: dict[str, str] = {
    "шаг дискретизации дискуссии": "фиксированный временной интервал, внутри которого агрегируются сообщения, темы, тикеры, тональность и социальный граф",
    "кандидатное дискуссионное событие": "шаг дискретизации дискуссии, где тематическая, эмоциональная или сетевая динамика заметно отклоняется от модельной симуляции",
    "индекс отклонения дискуссии": "составной исследовательский индекс приоритета ручной проверки шага дискретизации, а не мера частотной популярности и не торговая рекомендация",
    "модельная траектория": "последовательность распределений тем, полученная агентной NetLogo-моделью для следующих шагов дискретизации",
    "распределение тем": "вектор долей тематических кластеров внутри шага дискретизации дискуссии",
    "сдвиг распределения тем": "расстояние между распределениями тем в двух шагах дискретизации",
    "графовая авторитетность": "PageRank-центральность автора в социальном графе внутри обсуждения",
    "payoff": "модельный выигрыш темы в эволюционной игре, рассчитанный по тональности и графовой авторитетности",
    "nRMSE": "нормированная среднеквадратичная ошибка между модельным и фактическим распределением тем",
    "тикер": "биржевой символ финансового инструмента, например TSLA или NVDA",
    "тикерная привязка": "сопоставление сообщения или шага дискретизации дискуссии с финансовым инструментом по тикеру, cashtag или названию компании",
    "рыночная проверка": "визуальное сопоставление дискуссионного шага дискретизации с последующим фактическим движением цены и бенчмарка",
    "бенчмарк": "рыночный ориентир для сравнения движения тикера, например SPY или QQQ",
}


def clean_text(value: Any, *, strip_period: bool = True) -> str:
    """Очищает текст сообщения без удаления финансово значимых обозначений
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
        strip_period: признак удаления завершающей точки из видимого текста
    
    Returns:
        очищенный текст без лишних пробелов и завершающей точки при необходимости
    """
    text = str(value if value is not None else "").translate(_DASHES)
    text = re.sub(r"\s+", " ", text).strip()
    if strip_period:
        text = re.sub(r"(?<!\d)[.]$", "", text)
    return text


def clean_html(value: Any) -> str:
    """Очищает HTML-фрагмент от завершающих точек в видимом тексте
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        HTML-фрагмент с нормализованным видимым текстом
    """
    text = str(value if value is not None else "").translate(_DASHES)
    # удаляем точки перед окончанием HTML-блоков и переносами, сохраняя десятичные числа
    text = re.sub(r"(?<!\d)\.\s*(</(?:div|p|section|li)>|<br\s*/?>)", r"\1", text)
    text = re.sub(r"(?<!\d)\.\s*$", "", text.strip())
    return text


def _safe(value: Any) -> str:
    """Экранирует значение для безопасной HTML-вставки
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        HTML-экранированная строка
    """
    return escape(clean_text(value), quote=True)


def _min(html: str) -> str:
    """Сжимает HTML-разметку для компактного вывода
    
    Args:
        html: HTML-фрагмент для вывода в интерфейс
    
    Returns:
        HTML-строка без лишних пробелов между тегами
    """
    return re.sub(r">\s+<", "><", html.strip())


def _html(html: str) -> None:
    """Выводит контролируемый HTML-фрагмент через Streamlit markdown
    
    Args:
        html: HTML-фрагмент для вывода в интерфейс
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    rendered = _min(clean_html(html))
    # st html в отдельных версиях Streamlit может иначе обрабатывать
    # санитизацию сложных inline-блоков
    # для контролируемых шаблонов безопаснее единый путь через markdown и unsafe_allow_html
    st.markdown(rendered, unsafe_allow_html=True)


def logo_mark(size: int = 40) -> str:
    """Возвращает SVG-логотип исследовательского интерфейса
    
    Args:
        size: размер элемента интерфейса
    
    Returns:
        SVG-разметка логотипа
    """
    return f"""
    <svg class="dmi-logo-svg" width="{size}" height="{size}" viewBox="0 0 96 96" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <defs>
        <linearGradient id="dmiLogoGradient" x1="10" y1="12" x2="86" y2="84" gradientUnits="userSpaceOnUse">
          <stop stop-color="#4F46E5"/>
          <stop offset="0.52" stop-color="#06B6D4"/>
          <stop offset="1" stop-color="#8B5CF6"/>
        </linearGradient>
      </defs>
      <rect x="8" y="8" width="80" height="80" rx="24" fill="url(#dmiLogoGradient)" fill-opacity="0.13" stroke="url(#dmiLogoGradient)" stroke-width="4"/>
      <path d="M24 58C35 58 36.8 34 50 34C59 34 63.5 46 72 46" stroke="url(#dmiLogoGradient)" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="24" cy="58" r="5" fill="#4F46E5"/>
      <circle cx="50" cy="34" r="5" fill="#06B6D4"/>
      <circle cx="72" cy="46" r="5" fill="#8B5CF6"/>
      <path d="M30 70H68" stroke="url(#dmiLogoGradient)" stroke-width="4" stroke-linecap="round" opacity="0.56"/>
    </svg>
    """


def term_help(term: str) -> str:
    """Возвращает всплывающую подсказку для термина интерфейса
    
    Args:
        term: термин, для которого требуется подсказка
    
    Returns:
        HTML-разметка подсказки или пустая строка
    """
    tip = GLOSSARY.get(term.lower(), "")
    if not tip:
        return ""
    return f'<span class="dmi-help" title="{escape(tip, quote=True)}">?</span>'


def term_label(label: str, term: str | None = None) -> str:
    """Добавляет подсказку к подписи элемента интерфейса
    
    Args:
        label: подпись элемента интерфейса
        term: термин, для которого требуется подсказка
    
    Returns:
        HTML-разметка подписи с подсказкой
    """
    key = (term or label).lower()
    return f"{escape(clean_text(label), quote=True)}{term_help(key)}"



def metric_card(title: str, value: str, caption: str = "", tone: str = "blue") -> None:
    """Показывает компактную карточку метрики в интерфейсе
    
    Args:
        title: заголовок блока интерфейса
        value: исходное значение, которое требуется нормализовать или преобразовать
        caption: поясняющая подпись блока интерфейса
        tone: визуальный тон карточки или панели
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    _html(f"""
    <div class="metric-card" data-tone="{_safe(tone)}">
      <div class="metric-title">{_safe(title)}</div>
      <div class="metric-value">{_safe(value)}</div>
      <div class="metric-caption">{_safe(caption)}</div>
    </div>
    """)


def info_panel(title: str, body: str, tone: str = "blue") -> None:
    """Показывает информационный блок с пояснением результата
    
    Args:
        title: заголовок блока интерфейса
        body: основной текст блока интерфейса
        tone: визуальный тон карточки или панели
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    _html(f"""
    <div class="info-panel" data-tone="{_safe(tone)}">
      <div class="info-title">{_safe(title)}</div>
      <div class="info-body">{clean_html(body)}</div>
    </div>
    """)


def app_header(
    title: str,
    subtitle: str,
    eyebrow: str = "Discussion Market Intelligence",
    flow: list[str | tuple[str, str]] | None = None,
    author: dict[str, str] | None = None,
    *,
    show_logo: bool = False,
    show_eyebrow: bool = False,
) -> None:
    """Показывает главный заголовок приложения
    
    Args:
        title: заголовок блока интерфейса
        subtitle: подзаголовок интерфейсного блока
        eyebrow: короткая верхняя подпись заголовка
        flow: список этапов, отображаемых в заголовке
        author: канонический идентификатор автора
        show_logo: признак отображения логотипа
        show_eyebrow: признак отображения верхней подписи
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    flow = flow or []
    flow_html = ""
    if flow:
        parts: list[str] = []
        for i, item in enumerate(flow, start=1):
            if isinstance(item, tuple):
                label, caption = item
            else:
                label, caption = item, ""
            parts.append(
                f'<div class="dmi-flow-node">'
                f'<div class="dmi-flow-node-top"><span class="dmi-flow-index">{i}</span><span class="dmi-flow-node-title">{_safe(label)}</span></div>'
                f'<div class="dmi-flow-node-caption">{_safe(caption)}</div>'
                f'</div>'
            )
        flow_html = f'<div class="dmi-flow-map">{"".join(parts)}</div>'
    author_html = ""
    if author:
        rows: list[str] = []
        for key, value in author.items():
            rows.append(f'<span><strong>{_safe(key)}</strong> {_safe(value)}</span>')
        author_html = f'<div class="dmi-author-line">{"".join(rows)}</div>'
    safe_title = _safe(title)
    if safe_title == "Discussion Market Intelligence":
        title_html = '<div class="dmi-brand-title">Discussion Market <span class="accent">Intelligence</span></div>'
    else:
        title_html = f'<div class="dmi-app-title">{safe_title}</div>'
    brand_parts: list[str] = []
    if show_logo or show_eyebrow:
        brand_parts.append('<div class="dmi-hero-brand">')
        if show_logo:
            brand_parts.append(logo_mark(48))
        if show_eyebrow:
            brand_parts.append(f'<div class="dmi-eyebrow">{_safe(eyebrow)}</div>')
        brand_parts.append('</div>')
    brand_html = ''.join(brand_parts)
    _html(f"""
    <section class="dmi-app-hero">
      {brand_html}
      {title_html}
      <div class="dmi-app-subtitle">{clean_html(subtitle)}</div>
      {flow_html}
      {author_html}
    </section>
    """)


def sidebar_brand(title: str, subtitle: str = "") -> None:
    """Показывает брендовый блок в боковой панели
    
    Args:
        title: заголовок блока интерфейса
        subtitle: подзаголовок интерфейсного блока
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    _html(f"""
    <div class="dmi-sidebar-brand">
      <div class="dmi-sidebar-brand-row">
        {logo_mark(34)}
        <div class="dmi-sidebar-brand-title">{_safe(title)}</div>
      </div>
      <div class="dmi-sidebar-brand-subtitle">{clean_html(subtitle)}</div>
    </div>
    """)


def page_header(title: str, description: str = "", step: str = "") -> None:
    """Показывает заголовок страницы или этапа анализа
    
    Args:
        title: заголовок блока интерфейса
        description: пояснение раздела или элемента интерфейса
        step: шаг изменения значения или номер шага анализа
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    _html(f"""
    <section class="dmi-page-hero">
      <div class="dmi-page-hero-inner">
        <div class="dmi-page-brandline">
          {logo_mark(28)}
          <div class="dmi-eyebrow dmi-page-eyebrow">Discussion Market Intelligence</div>
        </div>
        <div class="dmi-page-title">{_safe(title)}</div>
        <div class="dmi-page-desc">{clean_html(description)}</div>
      </div>
    </section>
    """)


def author_footer() -> None:
    """Показывает подпись автора в нижней части интерфейса
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    _html("""
    <footer class="dmi-author-footer">
      <span><strong>Магистр</strong> Снапков Максим Юрьевич</span>
      <span><strong>Тема исследования</strong> Методы анализа влияния мнений пользователей социальных сетей на ценообразование финансовых инструментов компаний</span>
      <span><strong>СПбГУ</strong> 2026</span>
    </footer>
    """)

def section_header(title: str, description: str = "", term: str | None = None) -> None:
    """Показывает заголовок раздела с дополнительным описанием
    
    Args:
        title: заголовок блока интерфейса
        description: пояснение раздела или элемента интерфейса
        term: термин, для которого требуется подсказка
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    _html(f"""
    <div class="dmi-section-head">
      <div class="dmi-section-title">{term_label(title, term)}</div>
      <div class="dmi-section-desc">{clean_html(description)}</div>
    </div>
    """)


def chip(label: str, tone: str = "blue") -> str:
    """Формирует HTML-чип для короткой метки состояния
    
    Args:
        label: подпись элемента интерфейса
        tone: визуальный тон карточки или панели
    
    Returns:
        HTML-разметка короткой метки
    """
    return f'<span class="dmi-chip" data-tone="{_safe(tone)}">{_safe(label)}</span>'


def chip_row(items: list[tuple[str, str]]) -> None:
    """Показывает строку HTML-чипов
    
    Args:
        items: список элементов для обработки
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    _html(f'<div class="dmi-chip-row">{"".join(chip(label, tone) for label, tone in items)}</div>')


def parameter_rail(items: list[tuple[str, str, str | None]]) -> None:
    """Показывает компактную панель параметров эксперимента
    
    Args:
        items: список элементов для обработки
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    html = "".join(
        f'<div class="dmi-param-item"><div class="dmi-param-label">{_safe(label)}</div><div class="dmi-param-value">{_safe(value)}</div><div class="dmi-param-caption">{_safe(caption or "")}</div></div>'
        for label, value, caption in items
    )
    _html(f'<div class="dmi-param-rail">{html}</div>')


def analysis_context_banner(title: str, caption: str, metrics: list[tuple[str, str, str | None]]) -> None:
    """Показывает баннер контекста выбранного анализа
    
    Args:
        title: заголовок блока интерфейса
        caption: поясняющая подпись блока интерфейса
        metrics: набор пар название значение для баннера
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    metric_html = "".join(
        f'<div class="dmi-context-metric"><div class="dmi-context-label">{term_label(label)}</div><div class="dmi-context-value">{_safe(value)}</div><div class="dmi-context-caption">{_safe(sub or "")}</div></div>'
        for label, value, sub in metrics
    )
    _html(f"""
    <section class="dmi-context-banner">
      <div class="dmi-context-grid">
        <div>
          <div class="dmi-eyebrow">Результат анализа шага дискретизации</div>
          <div class="dmi-context-title">{term_label(title, "кандидатное дискуссионное событие")}</div>
          <div class="dmi-context-caption">{clean_html(caption)}</div>
        </div>
        {metric_html}
      </div>
    </section>
    """)


def executive_board(title: str, body: str, facts: list[tuple[str, str]], chips: list[tuple[str, str]] | None = None, kicker: str = "Аналитическое резюме") -> None:
    """Показывает крупный сводный блок с фактами и метками
    
    Args:
        title: заголовок блока интерфейса
        body: основной текст блока интерфейса
        facts: набор ключевых фактов для сводного блока
        chips: набор коротких меток состояния
        kicker: короткая подпись над сводным блоком
    
    Returns:
        None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
    """
    facts_html = "".join(
        f'<div class="dmi-fact"><div class="dmi-fact-label">{term_label(label)}</div><div class="dmi-fact-value">{_safe(value)}</div></div>'
        for label, value in facts
    )
    chips_html = "".join(chip(label, tone) for label, tone in (chips or []))
    chip_block = f'<div class="dmi-chip-row">{chips_html}</div>' if chips_html else ""
    _html(f"""
    <section class="dmi-exec-board">
      <div class="dmi-exec-grid">
        <div>
          <div class="dmi-exec-kicker">{_safe(kicker)}</div>
          <div class="dmi-exec-title">{_safe(title)}</div>
          <div class="dmi-exec-body">{clean_html(body)}</div>
          {chip_block}
        </div>
        <div class="dmi-fact-list">{facts_html}</div>
      </div>
    </section>
    """)
