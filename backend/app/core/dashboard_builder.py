"""Turn a validated dataframe into rich, interactive chart payloads for the UI.

The LLM is intentionally not allowed to produce fabricated numbers.
It specifies the visualization intent and fields, and DashboardBuilder performs
every aggregation, cross-tabulation, percentage, and statistical calculation locally.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DashboardBuilder:
    chart_type_map = {
        "area": "Area Graph",
        "area graph": "Area Graph",
        "line": "Line Graph",
        "line graph": "Line Graph",
        "bar": "Bar Chart",
        "bar chart": "Bar Chart",
        "stacked bar": "Stacked Bar Graph",
        "stacked bar graph": "Stacked Bar Graph",
        "multiset bar": "Multi-set Bar Chart",
        "multi-set bar": "Multi-set Bar Chart",
        "donut": "Donut Chart",
        "donut chart": "Donut Chart",
        "pie": "Donut Chart",
        "pie chart": "Donut Chart",
        "scatter": "Scatterplot",
        "scatterplot": "Scatterplot",
        "histogram": "Histogram",
        "treemap": "Treemap",
        "radar": "Radar Chart",
        "heatmap": "Heatmap",
        "funnel": "Funnel Chart",
        "box plot": "Box Plot",
    }

    @staticmethod
    def _number(value: Any) -> float:
        try:
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return round(val, 4)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _apply_aggregation(df_or_series: Any, metric: str, agg_func: str = "sum", grouped_by: Optional[str] = None) -> Any:
        """Dynamically compute aggregations across any mathematical method (sum, mean, median, count, min, max, std)."""
        agg = (agg_func or "sum").lower().strip()
        if grouped_by:
            gb = df_or_series.groupby(grouped_by, dropna=False)
            if agg in ["mean", "avg", "average"]:
                return gb[metric].mean()
            elif agg in ["median", "med"]:
                return gb[metric].median()
            elif agg in ["count", "cnt", "volume"]:
                return gb[metric].count()
            elif agg in ["count_distinct", "unique", "distinct"]:
                return gb[metric].nunique()
            elif agg in ["min", "minimum"]:
                return gb[metric].min()
            elif agg in ["max", "maximum"]:
                return gb[metric].max()
            elif agg in ["std", "stddev"]:
                return gb[metric].std().fillna(0)
            else:
                return gb[metric].sum()
        else:
            s = df_or_series[metric]
            if agg in ["mean", "avg", "average"]:
                return s.mean()
            elif agg in ["median", "med"]:
                return s.median()
            elif agg in ["count", "cnt", "volume"]:
                return s.count()
            elif agg in ["count_distinct", "unique", "distinct"]:
                return s.nunique()
            elif agg in ["min", "minimum"]:
                return s.min()
            elif agg in ["max", "maximum"]:
                return s.max()
            elif agg in ["std", "stddev"]:
                return s.std()
            else:
                return s.sum()

    @staticmethod
    def _format_value(value: float, metric_name: str = "", format_type: Optional[str] = None, unit: Optional[str] = None) -> str:
        """Format numbers dynamically across all domains (Healthcare, IoT, HR, Finance, etc.)."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "\u2014"
        num = float(value)
        fmt = (format_type or "").lower().strip()
        u = unit.strip() if unit else ""

        # Only mark as currency if explicitly told via format_type or unit — avoid false positives
        is_currency = fmt == "currency" or u in ("$", "\u20ac", "\u00a3", "\u00a5")
        is_pct = fmt == "percentage" or u == "%"

        if is_currency:
            prefix, suffix = u if u in ("$", "\u20ac", "\u00a3", "\u00a5") else "$", ""
        elif is_pct:
            prefix, suffix = "", "%"
        elif u:
            prefix, suffix = "", f" {u}" if len(u) > 1 else u
        else:
            prefix, suffix = "", ""

        abs_val = abs(num)
        if abs_val >= 1_000_000_000:
            val_str = f"{num / 1_000_000_000:.2f}B"
        elif abs_val >= 1_000_000:
            val_str = f"{num / 1_000_000:.2f}M"
        elif abs_val >= 10_000 and not is_pct:
            val_str = f"{num / 1_000:.1f}K"
        elif abs_val >= 100 or (num == int(num) and abs_val >= 1):
            val_str = f"{int(round(num)):,}"
        elif abs_val < 0.01 and abs_val > 0:
            val_str = f"{num:.4f}"
        else:
            val_str = f"{num:.2f}"

        return f"{prefix}{val_str}{suffix}"

    @classmethod
    def _format_currency_or_number(cls, value: float, metric_name: str = "") -> str:
        """Backward compatibility wrapper for legacy callers."""
        return cls._format_value(value, metric_name)

    def _field(self, requested: Optional[str], available: List[str], fallback: Optional[str] = None) -> Optional[str]:
        if requested:
            for column in available:
                if column.lower() == str(requested).lower() or column.lower().replace("_", " ") == str(requested).lower():
                    return column
        return fallback or (available[0] if available else None)

    # -----------------------------------------------------------------------
    # Aggregation Helpers
    # -----------------------------------------------------------------------
    def _period_series(
        self,
        df: pd.DataFrame,
        date_col: str,
        metric: str,
        agg_func: str = "sum",
        format_type: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if date_col not in df.columns or metric not in df.columns:
            return []
        frame = df[[date_col, metric]].copy()
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame.dropna(subset=[date_col])
        if frame.empty:
            return []

        # Determine best grouping frequency
        date_range_days = (frame[date_col].max() - frame[date_col].min()).days
        if date_range_days > 730:
            frame["_period"] = frame[date_col].dt.to_period("Q")
        elif date_range_days > 60:
            frame["_period"] = frame[date_col].dt.to_period("M")
        else:
            frame["_period"] = frame[date_col].dt.to_period("W")

        grouped = self._apply_aggregation(frame, metric, agg_func, grouped_by="_period").tail(36)
        # Compute percentage based on the displayed values (not the underlying raw total)
        displayed_values = [self._number(v) for v in grouped.values]
        total_displayed = sum(abs(v) for v in displayed_values) or 1.0

        series = []
        for period, value in grouped.items():
            num = self._number(value)
            pct = round((abs(num) / total_displayed) * 100, 1) if total_displayed else 0.0
            series.append({
                "name": str(period),
                "value": num,
                metric: num,
                "percentage": pct,
                "formatted_value": self._format_value(num, metric, format_type, unit),
            })
        return series

    def _category_series(
        self,
        df: pd.DataFrame,
        dimension: str,
        metric: str,
        agg_func: str = "sum",
        max_items: int = 12,
        sort_order: str = "descending",
        format_type: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if dimension not in df.columns or metric not in df.columns:
            return []
        frame = df[[dimension, metric]].dropna()
        if frame.empty:
            return []

        grouped = self._apply_aggregation(frame, metric, agg_func, grouped_by=dimension)
        if sort_order == "ascending":
            grouped = grouped.sort_values(ascending=True)
        else:
            grouped = grouped.sort_values(ascending=False)

        top_items = grouped.head(max_items)
        # Compute percentage based on the displayed subset values
        displayed_values = [self._number(v) for v in top_items.values]
        total_displayed = sum(abs(v) for v in displayed_values) or 1.0
        series = []
        for name, value in top_items.items():
            num = self._number(value)
            pct = round((abs(num) / total_displayed) * 100, 1) if total_displayed else 0.0
            series.append({
                "name": str(name),
                "value": num,
                metric: num,
                "percentage": pct,
                "formatted_value": self._format_value(num, metric, format_type, unit),
            })
        return series

    def _stacked_category_series(
        self,
        df: pd.DataFrame,
        dim1: str,
        dim2: str,
        metric: str,
        agg_func: str = "sum",
        format_type: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Cross-tabulate two dimensions for Stacked Bar and Grouped Bar charts."""
        if dim1 not in df.columns or dim2 not in df.columns or metric not in df.columns:
            return [], []
        frame = df[[dim1, dim2, metric]].dropna()
        if frame.empty:
            return [], []

        top_dim1 = frame.groupby(dim1)[metric].count().nlargest(8).index
        top_dim2 = frame.groupby(dim2)[metric].count().nlargest(5).index
        filtered = frame[frame[dim1].isin(top_dim1) & frame[dim2].isin(top_dim2)]

        agg = "mean" if agg_func in ["mean", "avg", "average"] else "sum"
        pivot = filtered.pivot_table(index=dim1, columns=dim2, values=metric, aggfunc=agg, fill_value=0)
        secondary_keys = [str(c) for c in pivot.columns]

        data = []
        for name, row in pivot.iterrows():
            row_dict = {
                "name": str(name),
                "value": self._number(row.sum()),
                "secondary_keys": secondary_keys,
            }
            for sec_col in secondary_keys:
                row_dict[sec_col] = self._number(row.get(sec_col, 0))
            data.append(row_dict)
        return data, secondary_keys

    def _scatter_series(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        format_type_x: Optional[str] = None,
        format_type_y: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if x not in df.columns or y not in df.columns:
            return []
        frame = df[[x, y]].dropna().head(300)
        return [
            {
                "x": self._number(row[x]),
                "y": self._number(row[y]),
                "z": 100,
                "x_label": x,
                "y_label": y,
                "name": f"Record {idx + 1}",
                "formatted_x": self._format_value(self._number(row[x]), x, format_type_x),
                "formatted_y": self._format_value(self._number(row[y]), y, format_type_y),
            }
            for idx, (_, row) in enumerate(frame.iterrows())
        ]

    def _histogram_series(self, df: pd.DataFrame, metric: str, format_type: Optional[str] = None, unit: Optional[str] = None) -> List[Dict[str, Any]]:
        if metric not in df.columns:
            return []
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if values.empty:
            return []

        # Use 5 to 7 clean bins to avoid squished/overlapping labels
        bins_count = 6 if len(values) >= 50 else min(5, max(3, len(values.unique())))
        counts, bin_edges = np.histogram(values, bins=bins_count)
        data = []

        is_all_ints = (values % 1 == 0).all()

        for i in range(len(counts)):
            b_min = float(bin_edges[i])
            b_max = float(bin_edges[i + 1])

            if is_all_ints:
                min_str = f"{int(round(b_min)):,}"
                max_str = f"{int(round(b_max)):,}"
            else:
                min_str = self._format_value(self._number(b_min), metric, format_type, unit)
                max_str = self._format_value(self._number(b_max), metric, format_type, unit)

            label = f"{min_str} - {max_str}"
            data.append({
                "name": label,
                "value": int(counts[i]),
                "count": int(counts[i]),
                "range_min": b_min,
                "range_max": b_max,
                "formatted_value": f"{int(counts[i]):,} records",
            })
        return data

    def _heatmap_series(self, df: pd.DataFrame, dim1: str, dim2: str, metric: str, agg_func: str = "sum") -> Dict[str, Any]:
        if dim1 not in df.columns or dim2 not in df.columns or metric not in df.columns:
            return {"x_labels": [], "y_labels": [], "data": [], "matrix": []}
        frame = df[[dim1, dim2, metric]].dropna()
        if frame.empty:
            return {"x_labels": [], "y_labels": [], "data": [], "matrix": []}

        top_dim1 = frame.groupby(dim1)[metric].count().nlargest(6).index.tolist()
        top_dim2 = frame.groupby(dim2)[metric].count().nlargest(5).index.tolist()
        filtered = frame[frame[dim1].isin(top_dim1) & frame[dim2].isin(top_dim2)]

        agg = "mean" if agg_func in ["mean", "avg"] else "sum"
        pivot = filtered.pivot_table(index=dim2, columns=dim1, values=metric, aggfunc=agg, fill_value=0)
        max_val = float(pivot.max().max()) or 1.0

        items = []
        matrix = []
        for y_idx, (y_name, row) in enumerate(pivot.iterrows()):
            row_vals = []
            for x_idx, (x_name, val) in enumerate(row.items()):
                num = self._number(val)
                intensity = round((num / max_val), 2) if max_val else 0
                row_vals.append(num)
                items.append({
                    "x": str(x_name),
                    "y": str(y_name),
                    "value": num,
                    "intensity": intensity,
                    "formatted_value": self._format_value(num, metric),
                })
            matrix.append(row_vals)

        return {
            "x_labels": [str(x) for x in pivot.columns],
            "y_labels": [str(y) for y in pivot.index],
            "matrix": matrix,
            "data": items,
        }

    def _box_plot_series(self, df: pd.DataFrame, dimension: Optional[str], metric: str) -> List[Dict[str, Any]]:
        if metric not in df.columns:
            return []
        if dimension and dimension in df.columns:
            top_cats = df.groupby(dimension)[metric].count().nlargest(5).index
            data = []
            for cat in top_cats:
                vals = df[df[dimension] == cat][metric].dropna()
                if len(vals) < 2:
                    continue
                q1, median, q3 = float(vals.quantile(0.25)), float(vals.quantile(0.50)), float(vals.quantile(0.75))
                min_val, max_val = float(vals.min()), float(vals.max())
                data.append({
                    "name": str(cat),
                    "min": self._number(min_val),
                    "q1": self._number(q1),
                    "median": self._number(median),
                    "q3": self._number(q3),
                    "max": self._number(max_val),
                    "value": self._number(median),
                    "formatted_value": f"Median: {self._format_value(median, metric)}",
                })
            if data:
                return data

        vals = df[metric].dropna()
        if len(vals) < 2:
            return []
        q1, median, q3 = float(vals.quantile(0.25)), float(vals.quantile(0.50)), float(vals.quantile(0.75))
        return [{
            "name": metric,
            "min": self._number(vals.min()),
            "q1": self._number(q1),
            "median": self._number(median),
            "q3": self._number(q3),
            "max": self._number(vals.max()),
            "value": self._number(median),
            "formatted_value": f"Median: {self._format_value(median, metric)}",
        }]

    # -----------------------------------------------------------------------
    # Master Materializer
    # -----------------------------------------------------------------------
    def materialize_chart(
        self,
        df: pd.DataFrame,
        summary: Dict[str, Any],
        spec: Dict[str, Any],
        chart_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Translate a chart specification into real computed dataframe aggregations using AI parameters."""
        numeric = summary.get("numeric_columns", [])
        dates = summary.get("date_columns", [])
        categories = summary.get("categorical_columns", [])

        metric = self._field(spec.get("y_axis") or spec.get("metric"), numeric, summary.get("primary_kpi"))
        chart_type = str(spec.get("type", "Bar Chart"))
        normalized = chart_type.lower()
        title = spec.get("title")

        agg_func = spec.get("aggregation", "sum")
        format_type = spec.get("format_type")
        unit = spec.get("unit")
        limit = spec.get("limit", 12)
        sort_order = spec.get("sort", "descending")

        # 1. SCATTERPLOT
        if "scatter" in normalized and len(numeric) >= 2:
            x = self._field(spec.get("x_axis"), numeric, numeric[0])
            y = self._field(spec.get("y_axis"), numeric, numeric[1] if len(numeric) > 1 else numeric[0])
            if x and y and x in df.columns and y in df.columns:
                data = self._scatter_series(df, x, y, format_type_x=format_type, format_type_y=format_type)
                return {
                    "id": chart_id,
                    "title": title or f"{y} vs {x} Correlation",
                    "type": "Scatterplot",
                    "x_axis": x,
                    "y_axis": y,
                    "data": data,
                    "insight_tooltip": spec.get("insight_tooltip", f"Examines bivariate distribution and correlation between {x} and {y}."),
                }

        if not metric or metric not in df.columns:
            return None

        # 2. TIME-SERIES: AREA GRAPH & LINE GRAPH
        if any(t in normalized for t in ["area", "line", "trend"]) and dates:
            dimension = self._field(spec.get("x_axis"), dates, dates[0])
            if dimension and dimension in df.columns:
                data = self._period_series(df, dimension, metric, agg_func=agg_func, format_type=format_type, unit=unit)
                final_type = "Area Graph" if "area" in normalized else "Line Graph"
                if data:
                    top_pt = max(data, key=lambda d: d["value"])
                    return {
                        "id": chart_id,
                        "title": title or f"{metric} Trend over Time",
                        "type": final_type,
                        "x_axis": dimension,
                        "y_axis": metric,
                        "data": data,
                        "insight_tooltip": spec.get("insight_tooltip", f"Peak period was {top_pt['name']} with {top_pt['formatted_value']} in {metric}."),
                    }

        # 3. STACKED BAR GRAPH & MULTI-SET BAR
        if any(t in normalized for t in ["stacked", "multi-set", "multiset", "grouped"]) and len(categories) >= 2:
            dim1 = self._field(spec.get("x_axis"), categories, categories[0])
            dim2_candidates = [c for c in categories if c != dim1]
            dim2 = self._field(spec.get("secondary_dimension"), dim2_candidates, dim2_candidates[0] if dim2_candidates else None)
            if dim1 and dim2:
                data, sec_keys = self._stacked_category_series(df, dim1, dim2, metric, agg_func=agg_func, format_type=format_type, unit=unit)
                final_type = "Stacked Bar Graph" if "stacked" in normalized else "Multi-set Bar Chart"
                if data:
                    return {
                        "id": chart_id,
                        "title": title or f"{metric} by {dim1} and {dim2}",
                        "type": final_type,
                        "x_axis": dim1,
                        "y_axis": metric,
                        "secondary_dimension": dim2,
                        "secondary_keys": sec_keys,
                        "data": data,
                        "insight_tooltip": spec.get("insight_tooltip", f"Compares {metric} contributions across {dim1} segmented by {dim2}."),
                    }

        # 4. HEATMAP
        if "heatmap" in normalized and len(categories) >= 2:
            dim1 = self._field(spec.get("x_axis"), categories, categories[0])
            dim2_candidates = [c for c in categories if c != dim1]
            dim2 = self._field(spec.get("secondary_dimension"), dim2_candidates, dim2_candidates[0] if dim2_candidates else None)
            if dim1 and dim2:
                heat_dict = self._heatmap_series(df, dim1, dim2, metric, agg_func=agg_func)
                if heat_dict["data"]:
                    return {
                        "id": chart_id,
                        "title": title or f"{metric} Density: {dim1} × {dim2}",
                        "type": "Heatmap",
                        "x_axis": dim1,
                        "y_axis": metric,
                        "secondary_dimension": dim2,
                        "data": heat_dict["data"],
                        "matrix_data": heat_dict,
                        "insight_tooltip": spec.get("insight_tooltip", f"Cross-dimensional density map of {metric} between {dim1} and {dim2}."),
                    }

        # 5. BOX PLOT
        if any(t in normalized for t in ["box", "whisker"]):
            dim = self._field(spec.get("x_axis"), categories, categories[0] if categories else None)
            data = self._box_plot_series(df, dim, metric)
            if data:
                return {
                    "id": chart_id,
                    "title": title or f"{metric} Distribution Variance",
                    "type": "Box Plot",
                    "x_axis": dim or "Overall",
                    "y_axis": metric,
                    "data": data,
                    "insight_tooltip": spec.get("insight_tooltip", f"Statistical distribution and quartile spread for {metric}."),
                }

        # 6. HISTOGRAM / DISTRIBUTION
        if any(t in normalized for t in ["histogram", "distribution", "spread", "freq"]):
            data = self._histogram_series(df, metric, format_type=format_type, unit=unit)
            if data:
                return {
                    "id": chart_id,
                    "title": title or f"Distribution of {metric}",
                    "type": "Histogram",
                    "x_axis": "Bins",
                    "y_axis": metric,
                    "data": data,
                    "insight_tooltip": spec.get("insight_tooltip", f"Frequency distribution across value ranges for {metric}."),
                }

        # 7. TREEMAP
        if "treemap" in normalized and categories:
            dimension = self._field(spec.get("x_axis"), categories, categories[0])
            cat_data = self._category_series(df, dimension, metric, agg_func=agg_func, max_items=min(limit, 10), sort_order=sort_order, format_type=format_type, unit=unit)
            if cat_data:
                treemap_items = [{"name": d["name"], "size": d["value"], "value": d["value"], "percentage": d["percentage"]} for d in cat_data]
                return {
                    "id": chart_id,
                    "title": title or f"{dimension} Volume Breakdown ({metric})",
                    "type": "Treemap",
                    "x_axis": dimension,
                    "y_axis": metric,
                    "data": treemap_items,
                    "insight_tooltip": spec.get("insight_tooltip", f"Hierarchical tree view of {metric} across {dimension} categories."),
                }

        # 8. RADAR CHART
        if "radar" in normalized and categories:
            dimension = self._field(spec.get("x_axis"), categories, categories[0])
            cat_data = self._category_series(df, dimension, metric, agg_func=agg_func, max_items=min(limit, 6), sort_order=sort_order, format_type=format_type, unit=unit)
            if cat_data:
                max_val = max(d["value"] for d in cat_data) or 1.0
                radar_items = [{"subject": d["name"], "A": round((d["value"] / max_val) * 100, 1), "value": d["value"], "fullMark": 100} for d in cat_data]
                return {
                    "id": chart_id,
                    "title": title or f"{metric} Multi-Axis Radar",
                    "type": "Radar Chart",
                    "x_axis": dimension,
                    "y_axis": metric,
                    "data": radar_items,
                    "insight_tooltip": spec.get("insight_tooltip", f"Multi-dimensional radial comparison of {dimension} based on {metric}."),
                }

        # 9. FUNNEL CHART
        if "funnel" in normalized and categories:
            dimension = self._field(spec.get("x_axis"), categories, categories[0])
            cat_data = self._category_series(df, dimension, metric, agg_func=agg_func, max_items=min(limit, 6), sort_order=sort_order, format_type=format_type, unit=unit)
            if cat_data:
                return {
                    "id": chart_id,
                    "title": title or f"{metric} Funnel by {dimension}",
                    "type": "Funnel Chart",
                    "x_axis": dimension,
                    "y_axis": metric,
                    "data": cat_data,
                    "insight_tooltip": spec.get("insight_tooltip", f"Ranked funnel stages of {metric} across {dimension}."),
                }

        # 10. CATEGORICAL BAR, DONUT, PIE CHART
        if categories:
            dimension = self._field(spec.get("x_axis"), categories, categories[0])
            data = self._category_series(df, dimension, metric, agg_func=agg_func, max_items=limit, sort_order=sort_order, format_type=format_type, unit=unit)
            if not data:
                return None
            final_type = "Donut Chart" if any(t in normalized for t in ["donut", "pie"]) else "Bar Chart"
            top_item = data[0]
            return {
                "id": chart_id,
                "title": title or f"{metric} by {dimension}",
                "type": final_type,
                "x_axis": dimension,
                "y_axis": metric,
                "data": data,
                "insight_tooltip": spec.get("insight_tooltip", f"{top_item['name']} leads with {top_item['formatted_value']} ({top_item['percentage']}% of total)."),
            }

        # Fallback to Histogram
        data = self._histogram_series(df, metric, format_type=format_type, unit=unit)
        if data:
            return {
                "id": chart_id,
                "title": title or f"Distribution of {metric}",
                "type": "Histogram",
                "x_axis": "Bins",
                "y_axis": metric,
                "data": data,
                "insight_tooltip": spec.get("insight_tooltip", f"Statistical distribution of {metric}."),
            }

        return None

    def default_plan(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate a rich, balanced 4-5 chart default layout on dataset ingestion."""
        metric = summary.get("primary_kpi") or (summary.get("numeric_columns") or [None])[0]
        dates = summary.get("date_columns", [])
        categories = summary.get("categorical_columns", [])
        numeric = summary.get("numeric_columns", [])

        # Determine metric's semantic aggregation
        metric_lower = (metric or "").lower()
        is_avg_metric = any(kw in metric_lower for kw in ['age', 'rating', 'score', 'temperature', 'heartrate', 'rate', 'pct', 'percentage', 'percent', 'ratio', 'margin', 'latency', 'tenure', 'gpa'])
        default_agg = "mean" if is_avg_metric else "sum"

        plan: List[Dict[str, Any]] = []

        # 1. Primary Time Series
        if dates and metric:
            plan.append({
                "type": "Area Graph",
                "title": f"{metric} Trend over Time",
                "x_axis": dates[0],
                "y_axis": metric,
                "aggregation": default_agg,
            })

        # 2. Primary Category Breakdown
        if categories and metric:
            plan.append({
                "type": "Bar Chart",
                "title": f"Average {metric} by {categories[0]}" if default_agg == "mean" else f"{metric} by {categories[0]}",
                "x_axis": categories[0],
                "y_axis": metric,
                "aggregation": default_agg,
            })

        # 3. Market Share / Donut
        if categories:
            cat_dim = categories[1] if len(categories) >= 2 else categories[0]
            if default_agg == "sum" and metric:
                plan.append({
                    "type": "Donut Chart",
                    "title": f"{cat_dim} Share of {metric}",
                    "x_axis": cat_dim,
                    "y_axis": metric,
                    "aggregation": "sum",
                })
            else:
                # For demographic/average metrics, Donut displays category record distribution count
                plan.append({
                    "type": "Donut Chart",
                    "title": f"{cat_dim} Distribution",
                    "x_axis": cat_dim,
                    "y_axis": metric or "Records",
                    "aggregation": "count",
                })

        # 4. Distribution / Histogram
        if metric:
            plan.append({
                "type": "Histogram",
                "title": f"Distribution of {metric}",
                "y_axis": metric,
            })

        # 5. Correlation Scatter (pick 2 numeric metrics, never ID)
        if len(numeric) >= 2:
            x_m = numeric[0]
            y_m = numeric[1]
            plan.append({
                "type": "Scatterplot",
                "title": f"{y_m} vs {x_m} Correlation",
                "x_axis": x_m,
                "y_axis": y_m,
            })
        elif len(categories) >= 2 and metric and len(plan) < 5:
            plan.append({
                "type": "Bar Chart",
                "title": f"{metric} by {categories[1]}",
                "x_axis": categories[1],
                "y_axis": metric,
                "aggregation": default_agg,
            })

        return plan[:5]

    def heuristic_plan(
        self,
        summary: Dict[str, Any],
        query: str,
        intent: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Robust offline rule-based planner covering all query patterns."""
        lower = query.lower()
        metric = summary.get("primary_kpi") or (summary.get("numeric_columns") or [None])[0]
        dates = summary.get("date_columns", [])
        cats = summary.get("categorical_columns", [])
        numeric = summary.get("numeric_columns", [])

        metric_lower = (metric or "").lower()
        is_avg_metric = any(kw in metric_lower for kw in ['age', 'rating', 'score', 'temp', 'rate', 'pct', 'ratio', 'margin', 'latency', 'tenure'])
        default_agg = "mean" if is_avg_metric else "sum"

        intent_metrics = (intent or {}).get("metrics", [])
        intent_dimensions = (intent or {}).get("dimensions", [])
        time_filter = (intent or {}).get("time_filter")
        comparison = (intent or {}).get("comparison")
        top_bottom = (intent or {}).get("top_bottom")
        is_overview = (intent or {}).get("is_overview", False)

        selected_metric = intent_metrics[0] if intent_metrics else next((c for c in numeric if c.lower() in lower), metric)
        selected_cat = (
            intent_dimensions[0]
            if intent_dimensions
            else next((c for c in cats if c.lower() in lower), cats[0] if cats else None)
        )
        second_cat = intent_dimensions[1] if len(intent_dimensions) > 1 else (cats[1] if len(cats) > 1 else None)

        plans: List[Dict[str, Any]] = []

        # Overview query
        if is_overview or any(w in lower for w in ["everything", "overview", "all charts", "dashboard", "insights"]):
            return self.default_plan(summary)

        # Distribution / Spread / Histogram
        if any(w in lower for w in ["distribution", "spread", "histogram", "freq", "frequency", "variance"]):
            target_metric = selected_metric or (numeric[0] if numeric else None)
            if target_metric:
                plans.append({"type": "Histogram", "title": f"Distribution of {target_metric}", "y_axis": target_metric})

        # Comparison query
        if comparison:
            comp_cols = comparison.get("columns", [])
            for c_metric in comp_cols:
                if c_metric in numeric and selected_cat:
                    plans.append({"type": "Bar Chart", "title": f"{c_metric} by {selected_cat}", "x_axis": selected_cat, "y_axis": c_metric, "aggregation": default_agg})
            if len(comp_cols) >= 2 and all(c in numeric for c in comp_cols[:2]):
                plans.append({"type": "Scatterplot", "title": f"{comp_cols[1]} vs {comp_cols[0]}", "x_axis": comp_cols[0], "y_axis": comp_cols[1]})

        # Time-based query
        has_time = time_filter or any(w in lower for w in ["trend", "forecast", "over time", "month", "year", "growth", "quarter", "timeline", "sales forecast"])
        if has_time and dates and selected_metric:
            ctype = "Area Graph" if "area" in lower else "Line Graph"
            plans.append({"type": ctype, "title": f"{selected_metric} Trend", "x_axis": dates[0], "y_axis": selected_metric, "aggregation": default_agg})

        # Top / Bottom
        if top_bottom and selected_cat and selected_metric:
            direction = top_bottom.get("direction", "top").title()
            n = top_bottom.get("n", 5)
            plans.append({"type": "Bar Chart", "title": f"{direction} {n} {selected_cat} by {selected_metric}", "x_axis": selected_cat, "y_axis": selected_metric, "aggregation": default_agg})

        # Stacked / Heatmap breakdown
        if selected_cat and second_cat and selected_metric and any(w in lower for w in ["breakdown", "stacked", "cross", "matrix", "segment"]):
            plans.append({"type": "Stacked Bar Graph", "title": f"{selected_metric} by {selected_cat} & {second_cat}", "x_axis": selected_cat, "secondary_dimension": second_cat, "y_axis": selected_metric, "aggregation": default_agg})

        # Category breakdown & Donut
        if selected_cat and selected_metric:
            if not any(p.get("type") == "Bar Chart" and p.get("x_axis") == selected_cat for p in plans):
                plans.append({"type": "Bar Chart", "title": f"{selected_metric} by {selected_cat}", "x_axis": selected_cat, "y_axis": selected_metric, "aggregation": default_agg})
            if len(plans) < 5:
                plans.append({"type": "Donut Chart", "title": f"{selected_cat} Share of {selected_metric}", "x_axis": selected_cat, "y_axis": selected_metric, "aggregation": default_agg})

        # Scatter
        if len(numeric) >= 2 and not any(p.get("type") == "Scatterplot" for p in plans) and len(plans) < 5:
            plans.append({"type": "Scatterplot", "title": f"{numeric[1]} vs {numeric[0]}", "x_axis": numeric[0], "y_axis": numeric[1]})

        return (plans or self.default_plan(summary))[:5]

    def make_kpis(self, df: pd.DataFrame, summary: Dict[str, Any], ai_kpi_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metric = (ai_kpi_spec.get("column") if ai_kpi_spec else None) or summary.get("primary_kpi") or (summary.get("numeric_columns") or ["Records"])[0]
        
        # Smart aggregation inference:
        metric_lower = (metric or "").lower()
        is_avg = any(kw in metric_lower for kw in ['age', 'rating', 'score', 'temp', 'temperature', 'rate', 'pct', 'percentage', 'ratio', 'margin', 'latency', 'tenure', 'gpa'])
        
        if ai_kpi_spec and ai_kpi_spec.get("aggregation"):
            agg_func = ai_kpi_spec["aggregation"]
        elif is_avg:
            agg_func = "mean"
        else:
            agg_func = "sum"

        format_type = (ai_kpi_spec.get("format_type") if ai_kpi_spec else None) or summary.get("format_type")
        unit = (ai_kpi_spec.get("unit") if ai_kpi_spec else None) or summary.get("unit")

        if metric in df.columns:
            val = self._number(self._apply_aggregation(df, metric, agg_func))
        else:
            val = float(len(df))

        formatted = self._format_value(val, metric, format_type=format_type, unit=unit)

        secondary_metric = (ai_kpi_spec.get("secondary_column") if ai_kpi_spec else None) or summary.get("secondary_kpi")
        secondary_val_str = "—"
        if secondary_metric and secondary_metric in df.columns:
            sec_lower = secondary_metric.lower()
            sec_default_agg = "mean" if any(kw in sec_lower for kw in ['age', 'rating', 'score', 'rate', 'pct', 'salary', 'income']) else "sum"
            sec_agg = (ai_kpi_spec.get("secondary_aggregation") if ai_kpi_spec else None) or sec_default_agg
            sec_val = self._number(self._apply_aggregation(df, secondary_metric, sec_agg))
            secondary_val_str = self._format_value(sec_val, secondary_metric)

        kpi_title = f"Avg {metric}" if (agg_func == "mean" and is_avg) else metric

        return {
            "primary_kpi": kpi_title,
            "metric_column": metric,
            "value": val,
            "formatted_value": formatted,
            "aggregation": agg_func,
            "secondary_kpi": secondary_metric,
            "secondary_formatted_value": secondary_val_str,
            "total_rows": len(df),
            "data_quality": summary.get("quality_score", 100.0),
        }


dashboard_builder = DashboardBuilder()
