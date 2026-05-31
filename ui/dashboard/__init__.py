"""
Пакет собирает модули панели управления, отчётов, публикаций, рынка
и тематических визуализаций
"""

from __future__ import annotations

from importlib import import_module

_MODULE_NAMES = [
    "session_state",
    "data_loading",
    "controls",
    "summaries",
    "reports",
    "publication_cards",
    "market_view",
    "exports_and_graphs",
    "topic_views",
    "text_normalization",
]

_MODULES = [import_module(f"{__name__}.{name}") for name in _MODULE_NAMES]

_EXPORTS: dict[str, object] = {}
for _module in _MODULES:
    for _name, _value in _module.__dict__.items():
        if _name.startswith("__"):
            continue
        if _name in {"annotations"}:
            continue
        _EXPORTS[_name] = _value

for _module in _MODULES:
    _module.__dict__.update(_EXPORTS)

globals().update(_EXPORTS)

__all__ = sorted(_EXPORTS) + ["bind_context"]


def bind_context(**values: object) -> None:
    """Передаёт значения текущего запуска в модули интерфейса
    
    Args:
        **values: имена и значения глобального контекста интерфейса
    
    Returns:
        None
    """
    for _module in _MODULES:
        _module.__dict__.update(values)
    globals().update(values)
