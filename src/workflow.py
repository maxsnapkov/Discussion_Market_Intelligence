"""Совместимая точка импорта вычислительного конвейера

Реальная реализация разнесена по доменным пакетам src.data_ingestion, src.nlp,
src.discussion, src.simulation, src.indicators и src.market
"""

from __future__ import annotations

from .pipeline import *  # noqa: F401,F403
from .market.association import load_market_price_panel  # noqa: F401
