"""Оформление Streamlit-интерфейса и графиков Plotly

Модуль загружает CSS, настраивает логотип страницы
и применяет единые параметры размеров к графикам
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"


def load_design_system() -> None:
    """Подключает CSS-файл дизайн-системы Streamlit-интерфейса
    
    Returns:
        None
    """
    css = ROOT / "theme.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def logo_path() -> Path:
    """Возвращает путь к SVG-логотипу проекта
    
    Returns:
        Path: путь к SVG-логотипу проекта
    """
    return ASSET_DIR / "logo.svg"


def setup_logo() -> None:
    """Настраивает логотип страницы Streamlit при наличии файла
    
    Returns:
        None
    """
    path = logo_path()
    if not path.exists():
        return
    try:
        st.logo(str(path), size="large")
    except Exception:
        try:
            st.sidebar.image(str(path), width="stretch")
        except Exception:
            pass


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Применяет единые размеры и поля к графику Plotly
    
    Args:
        fig: объект графика Plotly
    
    Returns:
        go.Figure: тот же объект графика Plotly с обновлённой геометрией
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=100, r=44, t=72, b=86),
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig.update_xaxes(
        automargin=True,
        title_standoff=30,
        zeroline=False,
    )
    fig.update_yaxes(
        automargin=True,
        title_standoff=42,
        zeroline=False,
    )
    return fig


def plotly_chart(fig: go.Figure, **kwargs: Any) -> None:
    """Выводит график Plotly с применением общей темы
    
    Args:
        fig: объект графика Plotly
        **kwargs: дополнительные именованные параметры, передаваемые во внутренний вызов
    
    Returns:
        None
    """
    kwargs.setdefault("width", "stretch")
    kwargs.setdefault("theme", "streamlit")
    st.plotly_chart(apply_chart_theme(fig), **kwargs)
