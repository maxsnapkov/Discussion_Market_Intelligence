"""Нормализация видимого текста и обёртки Streamlit-виджетов"""
from __future__ import annotations

from typing import Any, Callable
from .common import *

_DMI_DASH_TRANS = str.maketrans({"\u2014": "-", "\u2013": "-", "\u2212": "-", "\u2011": "-", "\u2012": "-", "\u2015": "-"})

def _clean_visible_text(value: Any) -> Any:
    """Нормализует видимый текст интерфейса перед выводом
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        Any: строка интерфейса после нормализации тире и завершающей точки либо исходное значение
    """
    if not isinstance(value, str):
        return value
    text = value.translate(_DMI_DASH_TRANS)
    replacements = {
        "шаг дискретизации": "шаг дискретизации",
        "Шаг дискретизации": "Шаг дискретизации",
        "шаги моделирования": "шаги моделирования",
        "Шаги моделирования": "Шаги моделирования",
        "шагам моделирования": "шагам моделирования",
        "шагов моделирования": "шагов моделирования",
        "шагах моделирования": "шагах моделирования",
        "шаг дискретизации дискуссии": "шаг дискретизации дискуссии",
        "Шаг дискретизации дискуссии": "Шаг дискретизации дискуссии",
        "шага моделирования дискуссии": "шага моделирования дискуссии",
        "Шаги моделирования дискуссии": "Шаги моделирования дискуссии",
        "шагов моделирования дискуссии": "шагов моделирования дискуссии",
        "шаге моделирования дискуссии": "шаге моделирования дискуссии",
        "шаг дискретизации": "шаг дискретизации",
        "Шаг дискретизации": "Шаг дискретизации",
        "шаги моделирования": "шаги моделирования",
        "Шаги моделирования": "Шаги моделирования",
        "шагу моделирования": "шагу моделирования",
        "шагом моделирования": "шагом моделирования",
        "шаге моделирования": "шаге моделирования",
        "шагов моделирования": "шагов моделирования",
        "Симуляция": "Симуляция",
        "симуляция": "симуляция",
        "симуляционный": "симуляционный",
        "Симуляционный": "Симуляционный",
        "симуляционная": "симуляционная",
        "Симуляционная": "Симуляционная",
        "симуляционное": "симуляционное",
        "Симуляционное": "Симуляционное",
        "симуляционные": "симуляционные",
        "Симуляционные": "Симуляционные",
        "симуляция": "симуляция",
        "Симуляция": "Симуляция",
        "модельная оценка": "модельная оценка",
        "Модельная оценка": "Модельная оценка",
        "модельные оценки": "модельные оценки",
        "Модельные оценки": "Модельные оценки",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # сохраняем десятичные числа, имена файлов и сокращения, удаляем только точки в конце видимого блока
    text = re.sub(r"(?<!\d)\.\s*$", "", text.strip())
    text = re.sub(r"(?<!\d)\.\s*(\n|<br\s*/?>)", r"\1", text)
    return text

def _original_streamlit_function(name: str) -> Callable[..., Any]:
    """Получает исходную функцию Streamlit до локальной обёртки
    
    Args:
        name: имя файла, функции или элемента интерфейса
    
    Returns:
        Callable[..., Any]: исходная функция Streamlit, сохранённая до установки локальной обёртки
    """
    attr = f"_dmi_original_{name}"
    if not hasattr(st, attr):
        setattr(st, attr, getattr(st, name))
    return getattr(st, attr)


def _patch_text_function(name: str, skip_html: bool = False) -> None:
    """Создаёт обёртку для очистки текстовых элементов Streamlit
    
    Args:
        name: имя файла, функции или элемента интерфейса
        skip_html: признак пропуска HTML-элементов при нормализации текста
    
    Returns:
        None
    """
    original = _original_streamlit_function(name)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Вызывает исходную функцию интерфейса после нормализации входного текста
        
        Args:
            *args: позиционные параметры, передаваемые во внутренний вызов
            **kwargs: дополнительные именованные параметры, передаваемые во внутренний вызов
        
        Returns:
            Any: результат вызова исходной функции Streamlit
        """
        if skip_html and kwargs.get("unsafe_allow_html"):
            return original(*args, **kwargs)
        args = tuple(_clean_visible_text(a) for a in args)
        return original(*args, **kwargs)
    setattr(st, name, wrapped)

for _name in ["caption", "info", "warning", "success", "error"]:
    if hasattr(st, _name):
        _patch_text_function(_name)
if hasattr(st, "markdown"):
    _patch_text_function("markdown", skip_html=False)

def _patch_widget_text(name: str) -> None:
    """Создаёт обёртку для очистки подписей виджетов Streamlit
    
    Args:
        name: имя файла, функции или элемента интерфейса
    
    Returns:
        None
    """
    if not hasattr(st, name):
        return
    original = _original_streamlit_function(name)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Вызывает исходную функцию интерфейса после нормализации входного текста
        
        Args:
            *args: позиционные параметры, передаваемые во внутренний вызов
            **kwargs: дополнительные именованные параметры, передаваемые во внутренний вызов
        
        Returns:
            Any: результат вызова исходной функции Streamlit
        """
        args = list(args)
        if args and isinstance(args[0], str):
            args[0] = _clean_visible_text(args[0])
        if len(args) > 1 and isinstance(args[1], (list, tuple)):
            args[1] = type(args[1])(_clean_visible_text(x) if isinstance(x, str) else x for x in args[1])
        args = tuple(args)
        for key in ["label", "help", "placeholder"]:
            if key in kwargs and isinstance(kwargs[key], str):
                kwargs[key] = _clean_visible_text(kwargs[key])
        if "options" in kwargs and isinstance(kwargs["options"], (list, tuple)):
            kwargs["options"] = type(kwargs["options"])(_clean_visible_text(x) if isinstance(x, str) else x for x in kwargs["options"])
        return original(*args, **kwargs)
    setattr(st, name, wrapped)
