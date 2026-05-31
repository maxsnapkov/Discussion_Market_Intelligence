"""Общие зависимости модулей Streamlit-интерфейса"""

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
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import streamlit as st

from ..theme import load_design_system, plotly_chart
from ..components import app_header, page_header, author_footer, section_header, analysis_context_banner, executive_board, parameter_rail, sidebar_brand
from ..components import metric_card as ui_metric_card, info_panel as ui_info_panel, chip_row

from src.workflow import (
    AppConfig,
    calibrate_model,
    calibrate_model_optuna,
    initialize_project,
    load_windows,
    load_ticker_universe,
    run_predictive_market_indicator,
    run_predictive_market_indicator_for_range,
    run_predictive_market_indicator_for_raw_range,
    top_publications_for_window,
    load_market_price_panel,
    run_backtest,
    run_window_modeling,
    recompute_payoff_file,
    load_fixed_model_params,
    build_dynamic_window_state,
    build_user_graph_snapshot_from_messages,
    service_author_label,
    is_service_author,
    normalize_columns,
    detect_raw_schema,
)
