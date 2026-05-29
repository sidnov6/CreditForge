"""Chart-spec builders for the agent layer.

Tools attach a `_chart` spec; the runtime collects them and the cockpit renders
them with Recharts. Specs are small, declarative JSON — the LLM never fabricates
chart data, it just chooses which tool to call (and thus which chart appears).
"""
from __future__ import annotations

import uuid

# Semantic palette mirroring the cockpit theme
ACCENT = "#5b9cff"
GREEN = "#36d399"
AMBER = "#fbbf24"
RED = "#fb6f84"
BAND_COLORS = {"AAA": GREEN, "AA": GREEN, "A": "#5fcf9e",
               "BBB": ACCENT, "BB": AMBER, "B": RED}


def _id() -> str:
    return uuid.uuid4().hex[:8]


def bar(title, data, x, y, label=None, value_format="num",
        colors=None, x_label="", y_label="", reference=None):
    """Vertical bar chart. `colors` optional list aligned to data rows."""
    return {
        "id": _id(), "type": "bar", "title": title, "data": data,
        "x": x, "series": [{"key": y, "label": label or y, "color": ACCENT}],
        "colors": colors, "valueFormat": value_format,
        "xLabel": x_label, "yLabel": y_label, "reference": reference,
    }


def hbar(title, data, value_format="num", x_label="", diverging=False):
    """Horizontal bars (e.g. SHAP waterfall). data: [{name, value}]."""
    return {
        "id": _id(), "type": "hbar", "title": title, "data": data,
        "valueFormat": value_format, "xLabel": x_label, "diverging": diverging,
    }


def line(title, data, x, series, value_format="num", x_label="", y_label="",
         reference=None, diagonal=False):
    """Line chart. `series`: [{key,label,color}]."""
    return {
        "id": _id(), "type": "line", "title": title, "data": data, "x": x,
        "series": series, "valueFormat": value_format,
        "xLabel": x_label, "yLabel": y_label, "reference": reference,
        "diagonal": diagonal,
    }


def scatter(title, data, value_format="num", x_label="", y_label="",
            diagonal=False, domain=None):
    """Scatter (e.g. reliability curve). data: [{x, y}]."""
    return {
        "id": _id(), "type": "scatter", "title": title, "data": data,
        "valueFormat": value_format, "xLabel": x_label, "yLabel": y_label,
        "diagonal": diagonal, "domain": domain,
    }
