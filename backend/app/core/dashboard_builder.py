"""Turn a validated dataframe into chart payloads that the UI can render.

The LLM is intentionally not allowed to produce values.  It can choose a
visualization and fields, but this module performs every aggregation locally.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import math
import re

import numpy as np
import pandas as pd


class DashboardBuilder:
    chart_type_map = {
        "area": "Area Graph", "line": "Line Graph", "bar": "Bar Chart",
        "donut": "Donut Chart", "pie": "Donut Chart", "scatter": "Scatterplot",
        "histogram": "Histogram", "forecast": "Line Graph",
    }

    @staticmethod
    def _number(value: Any) -> float:
        value = float(value)
        return 0.0 if math.isnan(value) or math.isinf(value) else round(value, 2)

    def _field(self, requested: Optional[str], available: List[str], fallback: Optional[str] = None) -> Optional[str]:
        if requested:
            for column in available:
                if column.lower() == str(requested).lower():
                    return column
        return fallback or (available[0] if available else None)

    def _period_series(self, df: pd.DataFrame, date_col: str, metric: str) -> List[Dict[str, Any]]:
        frame = df[[date_col, metric]].copy()
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame.dropna(subset=[date_col])
        if frame.empty:
            return []
        frame["_period"] = frame[date_col].dt.to_period("M")
        grouped = frame.groupby("_period", sort=True)[metric].sum().tail(24)
        return [{"name": str(period), "value": self._number(value)} for period, value in grouped.items()]

    def _category_series(self, df: pd.DataFrame, dimension: str, metric: str) -> List[Dict[str, Any]]:
        frame = df[[dimension, metric]].dropna()
        grouped = frame.groupby(dimension, dropna=False)[metric].sum().sort_values(ascending=False).head(12)
        return [{"name": str(name), "value": self._number(value)} for name, value in grouped.items()]

    def _scatter_series(self, df: pd.DataFrame, x: str, y: str) -> List[Dict[str, Any]]:
        frame = df[[x, y]].dropna().head(300)
        return [{"x": self._number(row[x]), "y": self._number(row[y]), "z": 80} for _, row in frame.iterrows()]

    def materialize_chart(self, df: pd.DataFrame, summary: Dict[str, Any], spec: Dict[str, Any], chart_id: str) -> Optional[Dict[str, Any]]:
        numeric = summary.get("numeric_columns", [])
        dates = summary.get("date_columns", [])
        categories = summary.get("categorical_columns", [])
        metric = self._field(spec.get("y_axis") or spec.get("metric"), numeric, summary.get("primary_kpi"))
        chart_type = str(spec.get("type", "Bar Chart"))
        normalized = chart_type.lower()

        if "scatter" in normalized and len(numeric) >= 2:
            x = self._field(spec.get("x_axis"), numeric, numeric[0])
            y = self._field(spec.get("y_axis"), numeric, numeric[1] if len(numeric) > 1 else numeric[0])
            data = self._scatter_series(df, x, y)
            return {"id": chart_id, "title": spec.get("title", f"{y} vs {x}"), "type": "Scatterplot", "x_axis": x, "y_axis": y, "data": data, "insight_tooltip": spec.get("insight_tooltip", f"Explore the relationship between {x} and {y}.")}

        if not metric:
            return None
        use_time = ("line" in normalized or "area" in normalized or "trend" in str(spec.get("title", "")).lower()) and dates
        if use_time:
            dimension = self._field(spec.get("x_axis"), dates, dates[0])
            data = self._period_series(df, dimension, metric)
            final_type = "Area Graph" if "area" in normalized else "Line Graph"
        else:
            dimension = self._field(spec.get("x_axis") or spec.get("dimension"), categories, categories[0] if categories else None)
            if dimension:
                data = self._category_series(df, dimension, metric)
            elif dates:
                dimension = dates[0]
                data = self._period_series(df, dimension, metric)
            else:
                # A numeric-only dataset still gets a truthful distribution.
                values = df[metric].dropna()
                counts, bins = np.histogram(values, bins=min(10, max(3, int(np.sqrt(len(values))))))
                data = [{"name": f"{bins[i]:.1f}-{bins[i + 1]:.1f}", "value": int(counts[i])} for i in range(len(counts))]
                dimension = "Distribution"
                chart_type = "Histogram"
            final_type = "Donut Chart" if any(x in normalized for x in ("donut", "pie")) else ("Histogram" if "histogram" in normalized else "Bar Chart")

        if not data:
            return None
        top = max(data, key=lambda item: item["value"])
        return {
            "id": chart_id,
            "title": spec.get("title", f"{metric} by {dimension}"),
            "type": final_type,
            "x_axis": dimension,
            "y_axis": metric,
            "data": data,
            "insight_tooltip": spec.get("insight_tooltip", f"{top['name']} is the leading {dimension} at {top['value']:,.2f} {metric}."),
        }

    def default_plan(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        metric = summary.get("primary_kpi") or (summary.get("numeric_columns") or [None])[0]
        dates, categories, numeric = summary.get("date_columns", []), summary.get("categorical_columns", []), summary.get("numeric_columns", [])
        plan: List[Dict[str, Any]] = []
        if dates and metric:
            plan.append({"type": "Area Graph", "title": f"{metric} trend over time", "x_axis": dates[0], "y_axis": metric})
        if categories and metric:
            plan.extend([
                {"type": "Bar Chart", "title": f"{metric} by {categories[0]}", "x_axis": categories[0], "y_axis": metric},
                {"type": "Donut Chart", "title": f"{categories[0]} share of {metric}", "x_axis": categories[0], "y_axis": metric},
            ])
        if len(numeric) >= 2:
            plan.append({"type": "Scatterplot", "title": f"{numeric[1]} vs {numeric[0]}", "x_axis": numeric[0], "y_axis": numeric[1]})
        if not plan and metric:
            plan.append({"type": "Histogram", "title": f"Distribution of {metric}", "y_axis": metric})
        return plan[:4]

    def heuristic_plan(self, summary: Dict[str, Any], query: str, intent: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Robust offline planner; understands common analytical intents.

        When an ``intent`` dict (from IntentParser) is provided, uses its
        structured fields for smarter chart selection.  Without it, falls back
        to keyword matching — the app stays fully usable with zero LLM calls.
        """
        lower = query.lower()
        metric = summary.get("primary_kpi") or (summary.get("numeric_columns") or [None])[0]
        dates = summary.get("date_columns", [])
        cats = summary.get("categorical_columns", [])
        numeric = summary.get("numeric_columns", [])

        # --- Extract intent-aware fields ---
        intent_metrics = (intent or {}).get("metrics", [])
        intent_dimensions = (intent or {}).get("dimensions", [])
        time_filter = (intent or {}).get("time_filter")
        comparison = (intent or {}).get("comparison")
        top_bottom = (intent or {}).get("top_bottom")
        is_overview = (intent or {}).get("is_overview", False)

        # Pick best metric(s) and dimension(s)
        selected_metric = intent_metrics[0] if intent_metrics else next((c for c in numeric if c.lower() in lower), metric)
        selected_cat = (
            intent_dimensions[0] if intent_dimensions
            else next((c for c in cats if c.lower() in lower), cats[0] if cats else None)
        )

        plans: List[Dict[str, Any]] = []

        # --- Overview mode: return diverse set ---
        if is_overview:
            if dates and selected_metric:
                plans.append({"type": "Area Graph", "title": f"{selected_metric} trend over time", "x_axis": dates[0], "y_axis": selected_metric})
            if selected_cat and selected_metric:
                plans.append({"type": "Bar Chart", "title": f"{selected_metric} by {selected_cat}", "x_axis": selected_cat, "y_axis": selected_metric})
                plans.append({"type": "Donut Chart", "title": f"{selected_cat} share of {selected_metric}", "x_axis": selected_cat, "y_axis": selected_metric})
            if len(numeric) >= 2:
                plans.append({"type": "Scatterplot", "title": f"{numeric[1]} vs {numeric[0]}", "x_axis": numeric[0], "y_axis": numeric[1]})
            if selected_metric and not plans:
                plans.append({"type": "Histogram", "title": f"Distribution of {selected_metric}", "y_axis": selected_metric})
            return (plans or self.default_plan(summary))[:5]

        # --- Comparison queries: multiple metrics side-by-side ---
        if comparison:
            comp_cols = comparison.get("columns", [])
            for comp_metric in comp_cols[:3]:
                if comp_metric in numeric and selected_cat:
                    plans.append({"type": "Bar Chart", "title": f"{comp_metric} by {selected_cat}", "x_axis": selected_cat, "y_axis": comp_metric})
                elif comp_metric in numeric and dates:
                    plans.append({"type": "Line Graph", "title": f"{comp_metric} trend", "x_axis": dates[0], "y_axis": comp_metric})
            if len(comp_cols) >= 2 and all(c in numeric for c in comp_cols[:2]):
                plans.append({"type": "Scatterplot", "title": f"{comp_cols[1]} vs {comp_cols[0]}", "x_axis": comp_cols[0], "y_axis": comp_cols[1]})

        # --- Time-based queries ---
        has_time_intent = (
            time_filter
            or any(word in lower for word in ("trend", "over time", "month", "year", "current year", "forecast", "growth", "decline", "quarterly", "monthly", "yearly"))
        )
        if has_time_intent and dates and selected_metric:
            chart_type = "Area Graph" if any(w in lower for w in ("area", "growth", "cumulative")) else "Line Graph"
            plans.append({"type": chart_type, "title": f"{selected_metric} trend over time", "x_axis": dates[0], "y_axis": selected_metric})

        # --- Top / Bottom queries ---
        if top_bottom and selected_cat and selected_metric:
            direction = top_bottom.get("direction", "top")
            n = top_bottom.get("n", 5)
            plans.append({"type": "Bar Chart", "title": f"{direction.title()} {n} {selected_cat} by {selected_metric}", "x_axis": selected_cat, "y_axis": selected_metric})

        # --- Category breakdown ---
        if selected_cat and selected_metric:
            # Avoid duplicate if already added by comparison or top/bottom
            bar_exists = any(p.get("type") == "Bar Chart" and p.get("x_axis") == selected_cat and p.get("y_axis") == selected_metric for p in plans)
            if not bar_exists:
                plans.append({"type": "Bar Chart", "title": f"{selected_metric} by {selected_cat}", "x_axis": selected_cat, "y_axis": selected_metric})
            if any(word in lower for word in ("share", "composition", "distribution", "proportion", "percentage", "everything")):
                plans.append({"type": "Donut Chart", "title": f"{selected_cat} contribution", "x_axis": selected_cat, "y_axis": selected_metric})

        # --- Distribution / Histogram ---
        if any(word in lower for word in ("distribution", "spread", "histogram", "frequency")) and selected_metric:
            plans.append({"type": "Histogram", "title": f"Distribution of {selected_metric}", "y_axis": selected_metric})

        # --- Correlation / Scatter ---
        if any(word in lower for word in ("correlation", "relationship", "scatter")) and len(numeric) >= 2:
            plans.append({"type": "Scatterplot", "title": f"{numeric[1]} vs {numeric[0]}", "x_axis": numeric[0], "y_axis": numeric[1]})

        return (plans or self.default_plan(summary))[:5]

    def make_kpis(self, df: pd.DataFrame, summary: Dict[str, Any]) -> Dict[str, Any]:
        metric = summary.get("primary_kpi")
        total = self._number(df[metric].sum()) if metric and metric in df else 0
        return {"primary_kpi": metric or "Records", "value": total, "total_rows": len(df), "data_quality": summary.get("quality_score", 0)}


dashboard_builder = DashboardBuilder()
