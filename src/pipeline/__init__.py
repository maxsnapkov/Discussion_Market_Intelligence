"""
Пакет собирает доменные модули анализа дискуссий
и сохраняет прежнюю точку доступа через src.workflow
"""

from __future__ import annotations

from importlib import import_module

_MODULE_PATHS = [
    "src.pipeline.configuration",
    "src.nlp.sentiment",
    "src.nlp.topic_modeling",
    "src.nlp.ticker_extraction",
    "src.discussion.influence_payoff",
    "src.data_ingestion.preparation",
    "src.simulation.modeling",
    "src.indicators.discussion_indicators",
    "src.discussion.publications",
    "src.simulation.dynamic_state",
    "src.simulation.calibration",
    "src.market.association",
    "src.data_ingestion.raw_interval",
    "src.discussion.user_graph",
]

_MODULES = [import_module(path) for path in _MODULE_PATHS]

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

__all__ = sorted(_EXPORTS)
