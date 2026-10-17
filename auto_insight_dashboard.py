import os
import re
from pathlib import Path
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import streamlit as st
    import streamlit.components.v1 as components
except ImportError:
    st = None
    components = None

# Load .env automatically (with built-in parser fallback if python-dotenv is not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

from auto_data_pipeline import EnterpriseDataPipeline, DataPipelineConfig
from auto_report_pipeline import MatplotlibSeabornEngine, SeniorDataScientistReportPipeline

APP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = APP_DIR / "pipeline_output"


# =============================================================================
# 1. DECLARATIVE CHART SPECIFICATION & MATPLOTLIB/SEABORN ENGINE
# =============================================================================
@dataclass
class ChartSpec:
    chart_id: str
    title: str
    chart_type: str  # bar, horizontal_bar, line, area, scatter, histogram, box, violin, pie, heatmap
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    agg: str = "sum"  # sum, mean, median, count, min, max, none
    top_n: Optional[int] = 10
    sort_order: str = "desc"
    filter_col: Optional[str] = None
    filter_val: Optional[Any] = None
    width: int = 6  # 6 = half-width, 12 = full-width
    height_px: int = 360  # Default height in pixels (adjustable via drag-resize)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ChartEngine:
    """
    Enterprise Apache ECharts 5.5 Rendering & Telemetry Engine.
    Delivers 100% interactive, fluid 60FPS visual components matching
    https://echarts.apache.org/examples/en/index.html
    """

    SUPPORTED_TYPES = [
        "area", "line", "bar", "horizontal_bar", "donut", "pie",
        "rose", "scatter", "heatmap", "funnel"
    ]

    DARK_PALETTE = ["#6366f1", "#38bdf8", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#14b8a6", "#f43f5e", "#06b6d4", "#eab308"]
    LIGHT_PALETTE = ["#4f46e5", "#0284c7", "#059669", "#d97706", "#db2777", "#7c3aed", "#0d9488", "#e11d48", "#0891b2", "#ca8a04"]

    @classmethod
    def classify_columns(cls, df: pd.DataFrame, mutate_dates: bool = False) -> Dict[str, List[str]]:
        numeric, categorical, datetime_cols = [], [], []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                if EnterpriseDataPipeline._is_identifier_column_name(col):
                    categorical.append(col)
                else:
                    numeric.append(col)
            else:
                if df[col].dtype == "object" and ("date" in col.lower() or df[col].dropna().astype(str).str.len().mean() >= 8):
                    sample = df[col].dropna().head(100)
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() > 0.85:
                        if mutate_dates:
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                        datetime_cols.append(col)
                        continue
                if not EnterpriseDataPipeline._is_identifier_column_name(col):
                    categorical.append(col)
        return {"numeric": numeric, "categorical": categorical, "datetime": datetime_cols}

    @classmethod
    def pick_primary_measure(cls, num_cols: List[str]) -> Optional[str]:
        return EnterpriseDataPipeline.pick_primary_measure(num_cols)

    @classmethod
    def generate_default_dashboard(
        cls, df: pd.DataFrame, eda_summary: Optional[Dict[str, Any]] = None
    ) -> List[ChartSpec]:
        col_types = cls.classify_columns(df, mutate_dates=True)
        num_cols = col_types["numeric"]
        cat_cols = col_types["categorical"]
        dt_cols = col_types["datetime"]

        specs: List[ChartSpec] = []
        primary_measure = cls.pick_primary_measure(num_cols)
        low_card_cats = [c for c in cat_cols if 2 <= df[c].nunique() <= 35]

        # C1: Smooth Gradient Area Horizon with Timeline Zoom
        if dt_cols and primary_measure:
            specs.append(
                ChartSpec(
                    chart_id="C1",
                    title=f"Temporal Horizon: {primary_measure.replace('_', ' ').title()} Over Time",
                    chart_type="area",
                    x=dt_cols[0],
                    y=primary_measure,
                    agg="sum",
                    width=12,
                    height_px=380,
                    notes="Apache ECharts smooth gradient horizon with timeline slider & peak markers.",
                )
            )

        # C2: Ranked Horizontal Bars
        if low_card_cats and primary_measure:
            chosen_cat = next((c for c in low_card_cats if df[c].nunique() >= 5), low_card_cats[0])
            specs.append(
                ChartSpec(
                    chart_id=f"C{len(specs)+1}",
                    title=f"Top {chosen_cat.replace('_', ' ').title()} by {primary_measure.replace('_', ' ').title()}",
                    chart_type="horizontal_bar",
                    x=chosen_cat,
                    y=primary_measure,
                    agg="sum",
                    top_n=10,
                    sort_order="desc",
                    width=6,
                    height_px=360,
                    notes="Ranked horizontal bars with in-place value labels and zero label overlap.",
                )
            )

        # C3: Rounded Doughnut Composition
        if len(low_card_cats) >= 2 and primary_measure:
            second_cat = next((c for c in low_card_cats if c != specs[-1].x and 2 <= df[c].nunique() <= 7), low_card_cats[1])
            specs.append(
                ChartSpec(
                    chart_id=f"C{len(specs)+1}",
                    title=f"{primary_measure.replace('_', ' ').title()} Share by {second_cat.replace('_', ' ').title()}",
                    chart_type="donut",
                    x=second_cat,
                    y=primary_measure,
                    agg="sum",
                    top_n=7,
                    width=6,
                    height_px=360,
                    notes="Interactive Doughnut chart with center KPI and scrollable category legend.",
                )
            )

        # C4: Nightingale Rose Chart
        if low_card_cats and primary_measure:
            rose_cat = next((c for c in low_card_cats if 4 <= df[c].nunique() <= 12), low_card_cats[0])
            specs.append(
                ChartSpec(
                    chart_id=f"C{len(specs)+1}",
                    title=f"Segment Dynamics: Nightingale Rose ({rose_cat.replace('_', ' ').title()})",
                    chart_type="rose",
                    x=rose_cat,
                    y=primary_measure,
                    agg="sum",
                    top_n=8,
                    width=6,
                    height_px=360,
                    notes="Apache ECharts Nightingale Rose chart encoding magnitude in radius and angle.",
                )
            )

        # C5: Vertical Bar with Track Background
        if low_card_cats and primary_measure:
            third_cat = low_card_cats[-1]
            specs.append(
                ChartSpec(
                    chart_id=f"C{len(specs)+1}",
                    title=f"{primary_measure.replace('_', ' ').title()} by {third_cat.replace('_', ' ').title()}",
                    chart_type="bar",
                    x=third_cat,
                    y=primary_measure,
                    agg="sum",
                    top_n=8,
                    width=6,
                    height_px=360,
                    notes="Rounded pill bars with background track shadow and mean benchmark.",
                )
            )

        # C6: Correlation Scatter & Heatmap
        if len(num_cols) >= 2:
            specs.append(
                ChartSpec(
                    chart_id=f"C{len(specs)+1}",
                    title=f"Bivariate Scatter: {num_cols[0].replace('_', ' ').title()} vs {num_cols[1].replace('_', ' ').title()}",
                    chart_type="scatter",
                    x=num_cols[0],
                    y=num_cols[1],
                    agg="none",
                    width=6,
                    height_px=360,
                    notes="Interactive scatter plot with crosshair coordinate inspection.",
                )
            )
            specs.append(
                ChartSpec(
                    chart_id=f"C{len(specs)+1}",
                    title="Pearson Correlation Matrix",
                    chart_type="heatmap",
                    width=6,
                    height_px=360,
                    notes="Interactive 2D Correlation Heatmap with visualMap gradient scale.",
                )
            )

        return specs

    @classmethod
    def render_chart(
        cls, df: pd.DataFrame, spec: ChartSpec, theme: str = "light"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        is_dark = (theme.lower() == "dark")
        palette = cls.DARK_PALETTE if is_dark else cls.LIGHT_PALETTE
        primary_color = palette[0]
        secondary_color = palette[1]
        text_color = "#f8fafc" if is_dark else "#0f172a"
        subtext_color = "#94a3b8" if is_dark else "#475569"
        border_color = "#334155" if is_dark else "#e2e8f0"
        bg_card = "#1e293b" if is_dark else "#ffffff"

        work_df = df.copy()
        if spec.filter_col and spec.filter_val and spec.filter_col in work_df.columns:
            work_df = work_df[work_df[spec.filter_col] == spec.filter_val]

        telem: Dict[str, Any] = {
            "chart_id": spec.chart_id,
            "chart_type": spec.chart_type,
            "title": spec.title,
            "x": spec.x,
            "y": spec.y,
            "agg": spec.agg,
        }

        toolbox = {
            "show": True,
            "top": "2%",
            "right": "3%",
            "feature": {
                "magicType": {"type": ["line", "bar", "stack"], "title": {"line": "Line", "bar": "Bar", "stack": "Stack"}},
                "dataZoom": {"title": {"zoom": "Zoom", "back": "Reset"}},
                "dataView": {"readOnly": True, "title": "Data View"},
                "saveAsImage": {"title": "Save PNG", "pixelRatio": 2},
                "restore": {"title": "Restore"}
            },
            "iconStyle": {"borderColor": subtext_color}
        }

        grid = {"left": "3%", "right": "4%", "bottom": "14%", "top": "18%", "containLabel": True}

        # 1. AREA / LINE
        if spec.chart_type in ("area", "line"):
            x_col = spec.x or work_df.columns[0]
            y_col = spec.y or work_df.select_dtypes(include=[np.number]).columns[-1]

            if pd.api.types.is_datetime64_any_dtype(work_df[x_col]):
                grouped = work_df.groupby(work_df[x_col].dt.strftime("%Y-%m-%d"))[y_col].agg(spec.agg or "sum").reset_index()
            else:
                grouped = work_df.groupby(x_col)[y_col].agg(spec.agg or "sum").reset_index()

            grouped = grouped.sort_values(x_col)
            x_data = grouped[x_col].astype(str).tolist()
            y_data = grouped[y_col].round(2).tolist()

            telem.update({"categories": x_data[:10], "values": y_data[:10], "total": float(sum(y_data))})

            is_area = (spec.chart_type == "area")
            series_cfg = {
                "name": y_col.replace("_", " ").title(),
                "type": "line",
                "smooth": True,
                "data": y_data,
                "lineStyle": {"width": 3, "color": primary_color},
                "itemStyle": {"color": primary_color},
                "markPoint": {"data": [{"type": "max", "name": "Max"}, {"type": "min", "name": "Min"}]},
                "markLine": {"data": [{"type": "average", "name": "Avg"}]},
            }
            if is_area:
                series_cfg["areaStyle"] = {
                    "color": {
                        "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": primary_color},
                            {"offset": 1, "color": "rgba(99, 102, 241, 0.0)" if is_dark else "rgba(79, 70, 229, 0.0)"}
                        ]
                    }
                }

            opt = {
                "backgroundColor": "transparent",
                "color": palette,
                "title": {"text": spec.title, "textStyle": {"color": text_color, "fontSize": 14, "fontWeight": 600}},
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
                "toolbox": toolbox,
                "grid": grid,
                "xAxis": {"type": "category", "data": x_data, "boundaryGap": False, "axisLabel": {"color": subtext_color}},
                "yAxis": {"type": "value", "splitLine": {"lineStyle": {"color": border_color, "type": "dashed"}}, "axisLabel": {"color": subtext_color}},
                "dataZoom": [{"type": "inside"}, {"type": "slider", "height": 18, "bottom": "2%"}],
                "series": [series_cfg]
            }
            return opt, telem

        # 2. HORIZONTAL BAR
        elif spec.chart_type == "horizontal_bar":
            x_col = spec.x or work_df.select_dtypes(include=["object", "category"]).columns[0]
            y_col = spec.y or work_df.select_dtypes(include=[np.number]).columns[-1]
            top_n = spec.top_n or 10

            grouped = work_df.groupby(x_col)[y_col].agg(spec.agg or "sum").reset_index()
            grouped = grouped.sort_values(y_col, ascending=True).tail(top_n)

            y_cats = grouped[x_col].astype(str).tolist()
            x_vals = grouped[y_col].round(2).tolist()

            telem.update({"categories": y_cats, "values": x_vals})

            opt = {
                "backgroundColor": "transparent",
                "color": palette,
                "title": {"text": spec.title, "textStyle": {"color": text_color, "fontSize": 14, "fontWeight": 600}},
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "toolbox": toolbox,
                "grid": grid,
                "yAxis": {"type": "category", "data": y_cats, "axisLabel": {"color": text_color, "fontSize": 11}},
                "xAxis": {"type": "value", "splitLine": {"lineStyle": {"color": border_color, "type": "dashed"}}, "axisLabel": {"color": subtext_color}},
                "series": [{
                    "name": y_col.replace("_", " ").title(),
                    "type": "bar",
                    "data": x_vals,
                    "barMaxWidth": 26,
                    "itemStyle": {
                        "borderRadius": [0, 6, 6, 0],
                        "color": {
                            "type": "linear", "x": 0, "y": 0, "x2": 1, "y2": 0,
                            "colorStops": [
                                {"offset": 0, "color": primary_color},
                                {"offset": 1, "color": secondary_color}
                            ]
                        }
                    },
                    "label": {"show": True, "position": "right", "color": text_color, "fontSize": 11}
                }]
            }
            return opt, telem

        # 3. VERTICAL BAR
        elif spec.chart_type == "bar":
            x_col = spec.x or work_df.select_dtypes(include=["object", "category"]).columns[0]
            y_col = spec.y or work_df.select_dtypes(include=[np.number]).columns[-1]
            top_n = spec.top_n or 10

            grouped = work_df.groupby(x_col)[y_col].agg(spec.agg or "sum").reset_index()
            grouped = grouped.sort_values(y_col, ascending=False).head(top_n)

            x_cats = grouped[x_col].astype(str).tolist()
            y_vals = grouped[y_col].round(2).tolist()

            telem.update({"categories": x_cats, "values": y_vals})

            opt = {
                "backgroundColor": "transparent",
                "color": palette,
                "title": {"text": spec.title, "textStyle": {"color": text_color, "fontSize": 14, "fontWeight": 600}},
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "toolbox": toolbox,
                "grid": grid,
                "xAxis": {"type": "category", "data": x_cats, "axisLabel": {"color": text_color, "rotate": 25 if len(x_cats) > 6 else 0}},
                "yAxis": {"type": "value", "splitLine": {"lineStyle": {"color": border_color, "type": "dashed"}}, "axisLabel": {"color": subtext_color}},
                "series": [{
                    "name": y_col.replace("_", " ").title(),
                    "type": "bar",
                    "data": y_vals,
                    "showBackground": True,
                    "backgroundStyle": {"color": "rgba(255, 255, 255, 0.04)" if is_dark else "rgba(0, 0, 0, 0.03)"},
                    "barMaxWidth": 42,
                    "itemStyle": {
                        "borderRadius": [6, 6, 0, 0],
                        "color": {
                            "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                            "colorStops": [
                                {"offset": 0, "color": primary_color},
                                {"offset": 1, "color": secondary_color}
                            ]
                        }
                    },
                    "markLine": {"data": [{"type": "average", "name": "Avg"}]}
                }]
            }
            return opt, telem

        # 4. PIE / DONUT / ROSE
        elif spec.chart_type in ("donut", "pie", "rose"):
            x_col = spec.x or work_df.select_dtypes(include=["object", "category"]).columns[0]
            y_col = spec.y or work_df.select_dtypes(include=[np.number]).columns[-1]
            top_n = spec.top_n or 7

            grouped = work_df.groupby(x_col)[y_col].agg(spec.agg or "sum").reset_index()
            grouped = grouped.sort_values(y_col, ascending=False).head(top_n)

            pie_data = [{"name": str(r[x_col]), "value": round(float(r[y_col]), 2)} for _, r in grouped.iterrows()]
            telem.update({"segments": pie_data})

            is_rose = (spec.chart_type == "rose")
            radius = ["25%", "72%"] if is_rose else (["42%", "72%"] if spec.chart_type == "donut" else "70%")

            series_dict = {
                "name": y_col.replace("_", " ").title(),
                "type": "pie",
                "radius": radius,
                "center": ["50%", "56%"],
                "data": pie_data,
                "itemStyle": {"borderRadius": 8, "borderColor": bg_card, "borderWidth": 2},
                "label": {"color": text_color, "fontSize": 12},
                "emphasis": {"itemStyle": {"shadowBlur": 12, "shadowOffsetX": 0, "shadowColor": "rgba(0, 0, 0, 0.5)"}}
            }
            if is_rose:
                series_dict["roseType"] = "radius"

            opt = {
                "backgroundColor": "transparent",
                "color": palette,
                "title": {"text": spec.title, "textStyle": {"color": text_color, "fontSize": 14, "fontWeight": 600}},
                "tooltip": {"trigger": "item", "formatter": "{a} <br/>{b}: <b>{c}</b> ({d}%)"},
                "toolbox": toolbox,
                "legend": {"top": "8%", "textStyle": {"color": subtext_color}, "type": "scroll"},
                "series": [series_dict]
            }
            return opt, telem

        # 5. SCATTER
        elif spec.chart_type == "scatter":
            nums = work_df.select_dtypes(include=[np.number]).columns.tolist()
            x_col = spec.x if spec.x in nums else (nums[0] if nums else None)
            y_col = spec.y if spec.y in nums else (nums[1] if len(nums) > 1 else nums[0])

            s_df = work_df[[x_col, y_col]].dropna()
            if len(s_df) > 1500:
                s_df = s_df.sample(1500, random_state=42)

            pts = s_df[[x_col, y_col]].values.round(2).tolist()
            telem.update({"points_count": len(pts), "x_mean": float(s_df[x_col].mean()), "y_mean": float(s_df[y_col].mean())})

            opt = {
                "backgroundColor": "transparent",
                "color": palette,
                "title": {"text": spec.title, "textStyle": {"color": text_color, "fontSize": 14, "fontWeight": 600}},
                "tooltip": {"trigger": "item", "formatter": f"{x_col}: {{c[0]}}<br/>{y_col}: {{c[1]}}"},
                "toolbox": toolbox,
                "grid": grid,
                "xAxis": {"type": "value", "name": x_col, "splitLine": {"lineStyle": {"color": border_color, "type": "dashed"}}, "axisLabel": {"color": subtext_color}},
                "yAxis": {"type": "value", "name": y_col, "splitLine": {"lineStyle": {"color": border_color, "type": "dashed"}}, "axisLabel": {"color": subtext_color}},
                "series": [{
                    "type": "scatter",
                    "data": pts,
                    "symbolSize": 8,
                    "itemStyle": {"color": primary_color, "opacity": 0.75},
                    "markLine": {
                        "lineStyle": {"type": "dashed", "color": secondary_color},
                        "data": [{"type": "average", "name": "Y-Avg"}, {"xAxis": float(s_df[x_col].mean()), "name": "X-Avg"}]
                    }
                }]
            }
            return opt, telem

        # 6. HEATMAP (Correlation or 2D Cross-tab)
        elif spec.chart_type == "heatmap":
            num_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()
            corr = work_df[num_cols].corr().round(2)
            cols = list(corr.columns)

            heat_data = []
            for i, c1 in enumerate(cols):
                for j, c2 in enumerate(cols):
                    heat_data.append([i, j, float(corr.loc[c1, c2])])

            telem.update({"features": cols, "correlation_matrix": corr.to_dict()})

            opt = {
                "backgroundColor": "transparent",
                "title": {"text": spec.title, "textStyle": {"color": text_color, "fontSize": 14, "fontWeight": 600}},
                "tooltip": {"position": "top", "formatter": "{b} : {c}"},
                "toolbox": toolbox,
                "grid": grid,
                "xAxis": {"type": "category", "data": cols, "splitArea": {"show": True}, "axisLabel": {"color": text_color, "rotate": 30}},
                "yAxis": {"type": "category", "data": cols, "splitArea": {"show": True}, "axisLabel": {"color": text_color}},
                "visualMap": {
                    "min": -1, "max": 1, "calculable": True, "orient": "horizontal", "left": "center", "bottom": "0%",
                    "inRange": {
                        "color": ["#3b82f6", "#f8fafc", "#ef4444"] if not is_dark else ["#1e3a8a", "#1e293b", "#dc2626"]
                    },
                    "textStyle": {"color": subtext_color}
                },
                "series": [{
                    "name": "Correlation",
                    "type": "heatmap",
                    "data": heat_data,
                    "label": {"show": True, "color": text_color},
                    "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.5)"}}
                }]
            }
            return opt, telem

        # 7. FUNNEL
        elif spec.chart_type == "funnel":
            x_col = spec.x or work_df.select_dtypes(include=["object", "category"]).columns[0]
            y_col = spec.y or work_df.select_dtypes(include=[np.number]).columns[-1]
            top_n = spec.top_n or 6

            grouped = work_df.groupby(x_col)[y_col].agg(spec.agg or "sum").reset_index()
            grouped = grouped.sort_values(y_col, ascending=False).head(top_n)

            f_data = [{"name": str(r[x_col]), "value": round(float(r[y_col]), 2)} for _, r in grouped.iterrows()]
            telem.update({"stages": f_data})

            opt = {
                "backgroundColor": "transparent",
                "color": palette,
                "title": {"text": spec.title, "textStyle": {"color": text_color, "fontSize": 14, "fontWeight": 600}},
                "tooltip": {"trigger": "item", "formatter": "{a} <br/>{b} : {c}"},
                "toolbox": toolbox,
                "legend": {"top": "8%", "textStyle": {"color": subtext_color}},
                "series": [{
                    "name": y_col.replace("_", " ").title(),
                    "type": "funnel",
                    "left": "10%",
                    "top": 60,
                    "bottom": 20,
                    "width": "80%",
                    "min": 0,
                    "max": float(grouped[y_col].max() * 1.05) if not grouped.empty else 100,
                    "minSize": "0%",
                    "maxSize": "100%",
                    "sort": "descending",
                    "gap": 3,
                    "label": {"show": True, "position": "inside", "color": "#ffffff"},
                    "itemStyle": {"borderRadius": 6, "borderColor": bg_card, "borderWidth": 1},
                    "data": f_data
                }]
            }
            return opt, telem

        # Fallback to bar
        spec.chart_type = "bar"
        return cls.render_chart(df, spec, theme)

    @classmethod
    def display_chart(
        cls, option: Dict[str, Any], chart_id: str, height_px: int = 360, theme: str = "light"
    ):
        theme_str = "'dark'" if theme.lower() == "dark" else "null"
        opt_json = json.dumps(option)
        safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", str(chart_id))

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
          <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            html, body {{ width: 100%; height: 100%; overflow: hidden; background: transparent; }}
            #echarts_{safe_id} {{ width: 100%; height: 100%; border-radius: 10px; }}
          </style>
        </head>
        <body>
          <div id="echarts_{safe_id}"></div>
          <script>
            (function() {{
              const container = document.getElementById("echarts_{safe_id}");
              const chart = echarts.init(container, {theme_str}, {{ renderer: "canvas", useDirtyRect: true }});
              const option = {opt_json};
              chart.setOption(option);
              
              const ro = new ResizeObserver(() => chart.resize());
              ro.observe(container);
              window.addEventListener("resize", () => chart.resize());
            }})();
          </script>
        </body>
        </html>
        """
        components.html(html_content, height=height_px + 10)


# =============================================================================
# 2. AI CONTROLLER (LOADED STRICTLY FROM .ENV + LOCAL DETERMINISTIC ENGINE)
# =============================================================================
class AIController:
    """
    Loads LLM credentials strictly from `.env` / environment variables and
    includes a deterministic NLP & statistical fallback engine.
    """

    def __init__(self):
        self.api_key = os.environ.get("LLM7_API_KEY", os.environ.get("OPENAI_API_KEY", "unused"))
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.llm7.io/v1")
        self.model = os.environ.get("LLM_MODEL_NAME", "default")
        self.client = (
            OpenAI(api_key=self.api_key, base_url=self.base_url)
            if OPENAI_SDK_AVAILABLE
            else None
        )

    def build_system_prompt(
        self,
        df: pd.DataFrame,
        col_types: Dict[str, List[str]],
        chart_specs: List[ChartSpec],
        chart_telemetry: Dict[str, Dict[str, Any]],
        eda_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        schema_info = {
            "total_rows": len(df),
            "numeric_columns": col_types["numeric"],
            "categorical_columns": {
                c: df[c].dropna().astype(str).unique()[:10].tolist() for c in col_types["categorical"]
            },
            "datetime_columns": col_types["datetime"],
        }
        active_charts_summary = [spec.to_dict() for spec in chart_specs]

        return f"""You are the **AutoInsight  AI Data Scientist & Dashboard Orchestrator**.
1. **Chart Reading & Deep Data Q&A**: Cite exact numbers, percentages, and category names from `ACTIVE_CHART_TELEMETRY` and `EDA_SUMMARY`.
2. **Pipeline Command Orchestration**: Instruct the pipeline to create, update, remove, or reset Matplotlib/Seaborn visual components by returning a JSON action block.

### DATASET SCHEMA
{json.dumps(schema_info, indent=2)}

### ACTIVE DASHBOARD CHARTS (SPECS)
{json.dumps(active_charts_summary, indent=2)}

### ACTIVE CHART TELEMETRY
{json.dumps(chart_telemetry, indent=2, default=str)}

### EDA SUMMARY STATS
{json.dumps(eda_summary.get("numeric_summary", {}) if eda_summary else {}, indent=2)}

### RESPONSE FORMAT (STRICT JSON ONLY)
{{
  "action": "NONE" | "ADD_CHART" | "UPDATE_CHART" | "REMOVE_CHART" | "RESET_DASHBOARD",
  "target_chart_id": "C1 (only required for UPDATE_CHART or REMOVE_CHART, else null)",
  "chart_spec": {{
    "title": "Descriptive Chart Title",
    "chart_type": "bar | horizontal_bar | line | area | scatter | histogram | box | violin | pie | heatmap",
    "x": "exact_column_name_from_schema",
    "y": "exact_column_name_from_schema_or_null",
    "color": "exact_column_name_or_null",
    "agg": "sum | mean | median | count | min | max | none",
    "top_n": 10,
    "sort_order": "desc | asc | none",
    "width": 6,
    "notes": "Brief insight note"
  }},
  "reply": "Clear markdown response explaining the insights or confirming the visual change."
}}
"""

    def query(
        self,
        user_prompt: str,
        chat_history: List[Dict[str, str]],
        df: pd.DataFrame,
        col_types: Dict[str, List[str]],
        chart_specs: List[ChartSpec],
        chart_telemetry: Dict[str, Dict[str, Any]],
        eda_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.client is not None and self.api_key.strip().lower() != "offline":
            sys_prompt = self.build_system_prompt(df, col_types, chart_specs, chart_telemetry, eda_summary)
            messages = [{"role": "system", "content": sys_prompt}]
            for msg in chat_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_prompt})

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    timeout=15.0,
                )
                raw_content = response.choices[0].message.content.strip()
                return self._parse_ai_json(raw_content)
            except Exception:
                pass

        return self.deterministic_local_engine(
            user_prompt, df, col_types, chart_specs, chart_telemetry, eda_summary
        )

    @classmethod
    def deterministic_local_engine(
        cls,
        user_prompt: str,
        df: pd.DataFrame,
        col_types: Dict[str, List[str]],
        chart_specs: List[ChartSpec],
        chart_telemetry: Dict[str, Dict[str, Any]],
        eda_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        p_low = user_prompt.lower().strip()
        num_cols = col_types["numeric"]
        cat_cols = col_types["categorical"]
        primary_num = ChartEngine.pick_primary_measure(num_cols)

        if "reset" in p_low and "dashboard" in p_low:
            return {
                "action": "RESET_DASHBOARD",
                "target_chart_id": None,
                "chart_spec": None,
                "reply": "Reset the dashboard back to the default Matplotlib & Seaborn executive layout.",
            }

        cid_match = re.search(r"\b(c\d+)\b", p_low)
        target_cid = cid_match.group(1).upper() if cid_match else None
        if target_cid and any(w in p_low for w in ["remove", "delete", "hide", "drop"]):
            return {
                "action": "REMOVE_CHART",
                "target_chart_id": target_cid,
                "chart_spec": None,
                "reply": f"Removed visual component **`{target_cid}`** from the dashboard.",
            }

        detected_type = None
        for ct in ["violin", "box", "histogram", "scatter", "heatmap", "pie", "area", "line", "horizontal_bar", "bar"]:
            if ct.replace("_", " ") in p_low or (ct == "box" and "boxplot" in p_low) or (ct == "pie" and "donut" in p_low):
                detected_type = ct
                break

        detected_agg = None
        if any(w in p_low for w in ["average", "mean", "avg"]):
            detected_agg = "mean"
        elif "median" in p_low:
            detected_agg = "median"
        elif "count" in p_low:
            detected_agg = "count"
        elif any(w in p_low for w in ["sum", "total"]):
            detected_agg = "sum"

        mentioned_cols = [c for c in df.columns if c.lower() in p_low or c.lower().replace("_", " ") in p_low]

        if target_cid and any(w in p_low for w in ["change", "update", "modify", "switch", "convert", "make", "set"]):
            spec_patch: Dict[str, Any] = {}
            if detected_type:
                spec_patch["chart_type"] = detected_type
            if detected_agg:
                spec_patch["agg"] = detected_agg
            if "full width" in p_low or "wide" in p_low:
                spec_patch["width"] = 12
            elif "half width" in p_low:
                spec_patch["width"] = 6
            return {
                "action": "UPDATE_CHART",
                "target_chart_id": target_cid,
                "chart_spec": spec_patch,
                "reply": f"Updated visual component **`{target_cid}`** with settings: `{json.dumps(spec_patch)}`.",
            }

        if any(w in p_low for w in ["add", "create", "plot", "build", "new chart", "show a"]):
            cats_m = [c for c in mentioned_cols if c in cat_cols]
            nums_m = [c for c in mentioned_cols if c in num_cols]
            x_c = cats_m[0] if cats_m else (mentioned_cols[0] if mentioned_cols else (cat_cols[0] if cat_cols else df.columns[0]))
            y_c = nums_m[0] if nums_m else primary_num
            c_type = detected_type or "horizontal_bar"
            agg_v = detected_agg or ("none" if c_type in ("box", "violin", "histogram", "scatter") else "sum")
            title = f"{agg_v.title() if agg_v != 'none' else ''} {(y_c or '').replace('_', ' ').title()} by {x_c.replace('_', ' ').title()}".strip()
            return {
                "action": "ADD_CHART",
                "target_chart_id": None,
                "chart_spec": {
                    "title": title,
                    "chart_type": c_type,
                    "x": x_c,
                    "y": y_c,
                    "agg": agg_v,
                    "top_n": 10,
                    "sort_order": "desc",
                    "width": 6,
                    "notes": f"Created via AI Co-Pilot: '{user_prompt}'",
                },
                "reply": f"Added a new Matplotlib/Seaborn **{c_type.upper()}** visual (**{title}**) to the dashboard.",
            }

        if "anomal" in p_low or "outlier" in p_low:
            anom = (eda_summary or {}).get("anomaly_report", {})
            lines = ["### Statistical Outlier & Anomaly Audit"]
            for col, info in anom.items():
                if not EnterpriseDataPipeline._is_identifier_column_name(col):
                    lines.append(
                        f"- **`{col}`**: **{info.get('iqr_anomalies', 0)}** IQR outliers ({info.get('iqr_anomaly_pct', 0)}% of rows) "
                        f"outside fences `[${info.get('lower_bound'):,.2f}, ${info.get('upper_bound'):,.2f}]`."
                    )
            return {"action": "NONE", "target_chart_id": None, "chart_spec": None, "reply": "\n".join(lines)}

        summary_lines = [
            f"### Executive Summary (`{len(df):,}` Clean Records × `{len(df.columns)}` Features)",
        ]
        if primary_num:
            summary_lines.append(
                f"- **Primary Measure (`{primary_num}`):** Total = **${df[primary_num].sum():,.2f}** | "
                f"Mean = **${df[primary_num].mean():,.2f}** | Median = **${df[primary_num].median():,.2f}**"
            )
        for cid, t in list(chart_telemetry.items())[:5]:
            if "top_item" in t:
                summary_lines.append(
                    f"- **`{cid}` ({t.get('title')}):** Top segment is `{t['top_item']}` vs lowest `{t.get('bottom_item')}`."
                )
        return {
            "action": "NONE",
            "target_chart_id": None,
            "chart_spec": None,
            "reply": "\n".join(summary_lines),
        }

    @staticmethod
    def _parse_ai_json(raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
            if match:
                cleaned = match.group(1)
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start : end + 1]

        try:
            parsed = json.loads(cleaned)
            return {
                "action": parsed.get("action", "NONE"),
                "target_chart_id": parsed.get("target_chart_id"),
                "chart_spec": parsed.get("chart_spec"),
                "reply": parsed.get("reply", "Command processed."),
            }
        except json.JSONDecodeError:
            return {
                "action": "NONE",
                "target_chart_id": None,
                "chart_spec": None,
                "reply": raw_text,
            }


# =============================================================================
# 3. STREAMLIT ENTERPRISE APPLICATION UI (LIGHT/DARK MODE + POWERBI RESIZE)
# =============================================================================
def run_streamlit_app():
    st.set_page_config(
        page_title="AutoInsight | Enterprise AI BI & Data Science Suite",
        page_icon="logo.svg",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    defaults = {
        "df": None,
        "eda_summary": {},
        "integrity_audit": {},
        "senior_report": None,
        "chart_specs": [],
        "chart_telemetry": {},
        "chat_history": [],
        "dataset_name": None,
        "ui_theme": "dark",  # Default to clean Light Mode with toggle to Dark Mode
        "show_copilot": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    is_light = st.session_state.ui_theme == "light"

    # Dynamic Design System Tokens for Flawless Light & Dark Mode
    if is_light:
        theme_vars = {
            "bg": "#f8fafc",
            "sidebar_bg": "#ffffff",
            "card_bg": "#ffffff",
            "card_shadow": "0 2px 10px rgba(15, 23, 42, 0.05)",
            "text": "#0f172a",
            "subtext": "#475569",
            "muted": "#64748b",
            "border": "#e2e8f0",
            "input_bg": "#ffffff",
            "input_border": "#cbd5e1",
            "primary": "#4f46e5",
            "brand_gradient": "linear-gradient(135deg, #4f46e5 0%, #6366f1 100%)",
            "dropdown_bg": "#ffffff",
            "dropdown_text": "#0f172a",
            "dropdown_hover": "#f1f5f9",
            "kpi_bg": "#ffffff",
            "hero_bg": "linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%)",
            "hero_border": "#e2e8f0",
            "badge_bg": "#e0e7ff",
            "badge_text": "#3730a3",
            "tab_active": "#4f46e5",
        }
    else:
        theme_vars = {
            "bg": "#0b1120",
            "sidebar_bg": "#0f172a",
            "card_bg": "#1e293b",
            "card_shadow": "0 4px 18px rgba(0, 0, 0, 0.35)",
            "text": "#f8fafc",
            "subtext": "#cbd5e1",
            "muted": "#94a3b8",
            "border": "#334155",
            "input_bg": "#1e293b",
            "input_border": "#334155",
            "primary": "#6366f1",
            "brand_gradient": "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
            "dropdown_bg": "#1e293b",
            "dropdown_text": "#f8fafc",
            "dropdown_hover": "#334155",
            "kpi_bg": "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)",
            "hero_bg": "linear-gradient(135deg, #131c2e 0%, #0b1120 100%)",
            "hero_border": "#334155",
            "badge_bg": "#312e81",
            "badge_text": "#c7d2fe",
            "tab_active": "#818cf8",
        }

    st.markdown(
        f"""
        <style>
        :root {{
            --ai-bg: {theme_vars['bg']};
            --ai-sidebar-bg: {theme_vars['sidebar_bg']};
            --ai-card-bg: {theme_vars['card_bg']};
            --ai-card-shadow: {theme_vars['card_shadow']};
            --ai-text: {theme_vars['text']};
            --ai-subtext: {theme_vars['subtext']};
            --ai-muted: {theme_vars['muted']};
            --ai-border: {theme_vars['border']};
            --ai-input-bg: {theme_vars['input_bg']};
            --ai-input-border: {theme_vars['input_border']};
            --ai-primary: {theme_vars['primary']};
            --ai-brand-gradient: {theme_vars['brand_gradient']};
            --ai-dropdown-bg: {theme_vars['dropdown_bg']};
            --ai-dropdown-text: {theme_vars['dropdown_text']};
            --ai-dropdown-hover: {theme_vars['dropdown_hover']};
            --ai-kpi-bg: {theme_vars['kpi_bg']};
            --ai-hero-bg: {theme_vars['hero_bg']};
            --ai-hero-border: {theme_vars['hero_border']};
            --ai-badge-bg: {theme_vars['badge_bg']};
            --ai-badge-text: {theme_vars['badge_text']};
            --ai-tab-active: {theme_vars['tab_active']};
        }}

        /* Global App Canvas & Main Layout */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {{
            background-color: var(--ai-bg) !important;
            color: var(--ai-text) !important;
        }}
        header[data-testid="stHeader"] {{
            background-color: var(--ai-bg) !important;
            border-bottom: 1px solid var(--ai-border) !important;
        }}
        header[data-testid="stHeader"] button, header[data-testid="stHeader"] svg {{
            color: var(--ai-text) !important;
            fill: var(--ai-text) !important;
        }}
        .block-container {{
            padding-top: 1.2rem !important;
            padding-bottom: 2.2rem !important;
            max-width: 98% !important;
        }}

        /* Full Sidebar Theming (Fixes Dark on Dark & Light on Dark text) */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div,
        section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {{
            background-color: var(--ai-sidebar-bg) !important;
            # border-right: 1px solid var(--ai-border) !important;
            color: var(--ai-text) !important;
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {{
            color: var(--ai-text) !important;
            font-weight: 700 !important;
        }}
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] span[data-testid="stWidgetLabel"],
        section[data-testid="stSidebar"] span[data-baseweb="radio"] {{
            color: var(--ai-text) !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] small {{
            color: var(--ai-subtext) !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: var(--ai-border) !important;
        }}

        /* Main Content Typography */
        .main h1, .main h2, .main h3, .main h4, .main h5, .main h6,
        .main div[data-testid="stMarkdownContainer"] h1,
        .main div[data-testid="stMarkdownContainer"] h2,
        .main div[data-testid="stMarkdownContainer"] h3 {{
            color: var(--ai-text) !important;
            font-weight: 700 !important;
            letter-spacing: -0.015em;
        }}
        .main p, .main div[data-testid="stMarkdownContainer"] p {{
            color: var(--ai-text) !important;
            line-height: 1.6;
        }}
        .main div[data-testid="stCaptionContainer"], .main .stCaption {{
            color: var(--ai-subtext) !important;
        }}
        .main label, .main div[data-testid="stWidgetLabel"] label {{
            color: var(--ai-text) !important;
            font-weight: 500 !important;
        }}

        /* KPI Metric Cards */
        div[data-testid="stMetric"] {{
            background: var(--ai-kpi-bg) !important;
            border: 1px solid var(--ai-border) !important;
            padding: 14px 18px !important;
            border-radius: 12px !important;
            box-shadow: var(--ai-card-shadow) !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            border-color: var(--ai-primary) !important;
        }}
        div[data-testid="stMetric"] label[data-testid="stMetricLabel"],
        div[data-testid="stMetric"] label[data-testid="stMetricLabel"] p {{
            color: var(--ai-subtext) !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
        }}
        div[data-testid="stMetric"] div[data-testid="stMetricValue"],
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] * {{
            color: var(--ai-text) !important;
            font-weight: 700 !important;
            font-size: 1.45rem !important;
        }}

        /* PowerBI-Style Resizable Chart Cards */
        .powerbi-resizable-card {{
            resize: both;
            overflow: hidden;
            min-height: 260px;
            min-width: 280px;
            max-width: 100%;
            background: var(--ai-card-bg) !important;
            border: 1px solid var(--ai-border) !important;
            border-radius: 12px !important;
            padding: 10px !important;
            position: relative !important;
            box-shadow: var(--ai-card-shadow) !important;
            transition: border-color 0.18s ease;
        }}
        .powerbi-resizable-card:hover {{
            border-color: var(--ai-primary) !important;
        }}
        .powerbi-resizable-card img {{
            width: 100%;
            height: calc(100% - 22px);
            object-fit: contain;
            display: block;
        }}
        .powerbi-resize-hint {{
            position: absolute;
            bottom: 4px;
            right: 14px;
            font-size: 0.70rem;
            color: var(--ai-muted);
            pointer-events: none;
            font-family: 'SF Mono', Consolas, monospace;
        }}
        .chart-card-header {{
            font-size: 0.80rem;
            color: var(--ai-subtext);
            font-family: 'SF Mono', Consolas, monospace;
            margin-bottom: 4px;
            letter-spacing: 0.02em;
        }}

        /* Fix Streamlit 3-Dots Kebab & Popover Menus (No more dark-on-dark!) */
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="menu"] {{
            background-color: var(--ai-dropdown-bg) !important;
            border: 1px solid var(--ai-border) !important;
            border-radius: 10px !important;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25) !important;
        }}
        ul[role="menu"] li,
        ul[role="menu"] li *,
        ul[role="menu"] li span,
        ul[role="menu"] li p {{
            color: var(--ai-dropdown-text) !important;
        }}
        ul[role="menu"] li:hover,
        ul[role="menu"] li:hover * {{
            background-color: var(--ai-dropdown-hover) !important;
            color: var(--ai-primary) !important;
        }}
        div[data-baseweb="popover"] * {{
            color: var(--ai-dropdown-text) !important;
        }}

        /* Buttons Styling */
        button[kind="primary"],
        button[data-testid="baseButton-primary"] {{
            background: var(--ai-brand-gradient) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 8px 18px !important;
            box-shadow: 0 3px 10px rgba(79, 70, 229, 0.3) !important;
            transition: all 0.15s ease !important;
        }}
        button[kind="primary"] *,
        button[data-testid="baseButton-primary"] * {{
            color: #ffffff !important;
        }}
        button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {{
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.45) !important;
            transform: translateY(-1px) !important;
        }}
        button[kind="secondary"],
        button[data-testid="baseButton-secondary"] {{
            background-color: var(--ai-card-bg) !important;
            color: var(--ai-text) !important;
            border: 1px solid var(--ai-border) !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.15s ease !important;
        }}
        button[kind="secondary"]:hover,
        button[data-testid="baseButton-secondary"]:hover {{
            border-color: var(--ai-primary) !important;
            color: var(--ai-primary) !important;
        }}

        /* Clean Tab Bar */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px !important;
            border-bottom: 1px solid var(--ai-border) !important;
            background: transparent !important;
            padding-bottom: 2px !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: var(--ai-subtext) !important;
            font-weight: 600 !important;
            border-radius: 6px 6px 0 0 !important;
            padding: 8px 16px !important;
            background: transparent !important;
        }}
        .stTabs [data-baseweb="tab"] p {{
            color: var(--ai-subtext) !important;
            font-weight: 600 !important;
        }}
        .stTabs [aria-selected="true"] {{
            color: var(--ai-tab-active) !important;
            border-bottom-color: var(--ai-tab-active) !important;
        }}
        .stTabs [aria-selected="true"] p {{
            color: var(--ai-tab-active) !important;
            font-weight: 700 !important;
        }}

        /* File Uploader Container */
        [data-testid="stFileUploader"] section {{
            background-color: var(--ai-input-bg) !important;
            border: 1.5px dashed var(--ai-input-border) !important;
            border-radius: 10px !important;
        }}
        [data-testid="stFileUploader"] section * {{
            color: var(--ai-text) !important;
        }}

        /* Expanders & Containers */
        div[data-testid="stExpander"] {{
            background: var(--ai-card-bg) !important;
            border: 1px solid var(--ai-border) !important;
            border-radius: 10px !important;
        }}
        div[data-testid="stExpander"] summary span {{
            color: var(--ai-text) !important;
            font-weight: 600 !important;
        }}

        /* Form Inputs & Selects */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] {{
            background-color: var(--ai-input-bg) !important;
            border-color: var(--ai-input-border) !important;
            color: var(--ai-text) !important;
            border-radius: 8px !important;
        }}
        .stTextInput input {{
            color: var(--ai-text) !important;
        }}

        /* Hero & Feature Showcase Cards */
        .hero-banner-card {{
            background: var(--ai-hero-bg);
            border: 1px solid var(--ai-hero-border);
            border-radius: 16px;
            padding: 28px 32px;
            margin-bottom: 24px;
            box-shadow: var(--ai-card-shadow);
        }}
        .hero-badge {{
            display: inline-block;
            background: var(--ai-badge-bg);
            color: var(--ai-badge-text);
            font-size: 0.72rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 20px;
            letter-spacing: 0.06em;
            margin-bottom: 12px;
            text-transform: uppercase;
        }}
        .feature-card {{
            background: var(--ai-card-bg);
            border: 1px solid var(--ai-border);
            border-radius: 12px;
            padding: 20px 22px;
            height: 100%;
            box-shadow: var(--ai-card-shadow);
            transition: transform 0.15s ease, border-color 0.15s ease;
        }}
        .feature-card:hover {{
            transform: translateY(-2px);
            border-color: var(--ai-primary);
        }}
        .feature-card h4 {{
            margin-top: 0;
            margin-bottom: 8px;
            color: var(--ai-text) !important;
            font-size: 1.05rem;
            font-weight: 600;
        }}
        .feature-card p {{
            color: var(--ai-subtext) !important;
            font-size: 0.85rem;
            line-height: 1.5;
            margin: 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    def load_or_process_file(uploaded_file, run_cleaning: bool, output_dir: str = str(DEFAULT_OUTPUT_DIR)):
        suffix = Path(uploaded_file.name).suffix
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            if run_cleaning:
                config = DataPipelineConfig(handle_outliers=False, top_n_categories=10, purge_legacy_pngs=True)
                pipeline = EnterpriseDataPipeline(config=config, output_dir=output_dir)
                cleaned_df, _ = pipeline.run_pipeline(input_file=tmp_path, source_name=uploaded_file.name)
                st.session_state.df = cleaned_df
                st.session_state.eda_summary = pipeline.eda_summary
                st.session_state.integrity_audit = pipeline.integrity_audit
            else:
                if suffix.lower() == ".csv":
                    df_loaded = pd.read_csv(tmp_path)
                elif suffix.lower() in (".xlsx", ".xls"):
                    df_loaded = pd.read_excel(tmp_path)
                elif suffix.lower() == ".parquet":
                    df_loaded = pd.read_parquet(tmp_path)
                else:
                    df_loaded = pd.read_json(tmp_path)
                df_loaded.columns = [c.strip("_") for c in df_loaded.columns]
                st.session_state.df = df_loaded
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        st.session_state.dataset_name = uploaded_file.name
        st.session_state.chart_specs = ChartEngine.generate_default_dashboard(
            st.session_state.df, st.session_state.eda_summary
        )
        rep_pipe = SeniorDataScientistReportPipeline(output_dir=output_dir)
        st.session_state.senior_report, _ = rep_pipe.generate_report(
            st.session_state.df,
            eda_summary=st.session_state.eda_summary,
            integrity_audit=st.session_state.integrity_audit,
            dataset_name=uploaded_file.name,
        )
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    f"Ingested **`{uploaded_file.name}`** "
                    f"({len(st.session_state.df):,} rows × {len(st.session_state.df.columns)} columns) "
                    f"and generated **{len(st.session_state.chart_specs)} Matplotlib & Seaborn visual components** "
                    f"plus the ** Report**.\n\n"
                    "Ask me to summarize insights, spot anomalies, or add/modify any chart."
                ),
            }
        ]

    def execute_pipeline_action(action_payload: Dict[str, Any]):
        action = (action_payload.get("action") or "NONE").upper()
        target_id = action_payload.get("target_chart_id")
        spec_dict = action_payload.get("chart_spec") or {}
        df_curr = st.session_state.df

        if action == "NONE" or df_curr is None:
            return

        if action == "RESET_DASHBOARD":
            st.session_state.chart_specs = ChartEngine.generate_default_dashboard(
                df_curr, st.session_state.eda_summary
            )
            return

        if action == "REMOVE_CHART" and target_id:
            st.session_state.chart_specs = [
                s for s in st.session_state.chart_specs if s.chart_id.upper() != str(target_id).upper()
            ]
            return

        if action == "ADD_CHART" and spec_dict:
            existing_ids = [
                int(s.chart_id[1:]) for s in st.session_state.chart_specs
                if s.chart_id.startswith("C") and s.chart_id[1:].isdigit()
            ]
            next_num = (max(existing_ids) + 1) if existing_ids else (len(st.session_state.chart_specs) + 1)
            new_id = f"C{next_num}"

            new_spec = ChartSpec(
                chart_id=new_id,
                title=spec_dict.get("title") or f"Chart {new_id}",
                chart_type=spec_dict.get("chart_type") or "horizontal_bar",
                x=spec_dict.get("x"),
                y=spec_dict.get("y"),
                color=spec_dict.get("color"),
                agg=spec_dict.get("agg") or "sum",
                top_n=spec_dict.get("top_n", 10),
                sort_order=spec_dict.get("sort_order") or "desc",
                width=int(spec_dict.get("width") or 6),
                height_px=int(spec_dict.get("height_px") or 360),
                notes=spec_dict.get("notes") or "Matplotlib & Seaborn visual component.",
            )
            st.session_state.chart_specs.append(new_spec)
            return

        if action == "UPDATE_CHART" and target_id and spec_dict:
            for idx, existing in enumerate(st.session_state.chart_specs):
                if existing.chart_id.upper() == str(target_id).upper():
                    updated = ChartSpec(
                        chart_id=existing.chart_id,
                        title=spec_dict.get("title") or existing.title,
                        chart_type=spec_dict.get("chart_type") or existing.chart_type,
                        x=spec_dict.get("x") if "x" in spec_dict and spec_dict["x"] is not None else existing.x,
                        y=spec_dict.get("y") if "y" in spec_dict else existing.y,
                        color=spec_dict.get("color") if "color" in spec_dict else existing.color,
                        agg=spec_dict.get("agg") or existing.agg,
                        top_n=spec_dict.get("top_n", existing.top_n),
                        sort_order=spec_dict.get("sort_order") or existing.sort_order,
                        width=int(spec_dict.get("width") or existing.width),
                        height_px=int(spec_dict.get("height_px") or existing.height_px),
                        notes=spec_dict.get("notes") or existing.notes,
                    )
                    st.session_state.chart_specs[idx] = updated
                    break

    # Sidebar (No Hybrid AI Config — loaded strictly from .env)
    with st.sidebar:
        st.title(" AutoInsight")
        st.caption(" Interactive Apache ECharts 5.5 BI & Data Science Platform")

        # Light Mode / Dark Mode Switcher
        theme_choice = st.radio(
            "Appearance Mode",
            ["light", "dark"],
            index=1 if st.session_state.ui_theme == "dark" else 0,
            format_func=lambda m: "☀️ Light Mode" if m == "light" else "🌙 Dark Mode",
            horizontal=True,
        )
        if theme_choice != st.session_state.ui_theme:
            st.session_state.ui_theme = theme_choice
            st.rerun()

        st.session_state.show_copilot = st.toggle(" Show AI Co-Pilot Panel", value=st.session_state.show_copilot)

        st.divider()
        st.subheader("1. Data & Report Pipelines")
        uploaded_file = st.file_uploader(
            "Upload Raw or Cleaned Dataset",
            type=["csv", "xlsx", "xls", "parquet", "json"],
        )
        run_cleaning = st.checkbox(
            "Run Enterprise Cleaning & Report Pipelines",
            value=True,
            help="Normalizes missing tokens, solves formulas, builds Matplotlib/Seaborn visuals, and generates the Report.",
        )

        if uploaded_file is not None:
            if st.button(" Run Pipelines & Build Suite", type="primary", use_container_width=True):
                with st.spinner("Executing Data Cleaning &  Report Pipelines..."):
                    load_or_process_file(uploaded_file, run_cleaning)
                st.rerun()

        default_cleaned_path = DEFAULT_OUTPUT_DIR / "cleaned_visualization_data.csv"
        if st.session_state.df is None and default_cleaned_path.exists():
            if st.button(" Load Existing Cleaned Dataset", use_container_width=True, help="Load pipeline_output/cleaned_visualization_data.csv"):
                df_existing = pd.read_csv(default_cleaned_path)
                df_existing.columns = [c.strip("_") for c in df_existing.columns]
                st.session_state.df = df_existing
                st.session_state.dataset_name = default_cleaned_path.name
                eda_p = DEFAULT_OUTPUT_DIR / "eda_reports" / "eda_summary.json"
                audit_p = DEFAULT_OUTPUT_DIR / "eda_reports" / "data_integrity_audit.json"
                if eda_p.exists():
                    st.session_state.eda_summary = json.loads(eda_p.read_text(encoding="utf-8"))
                if audit_p.exists():
                    st.session_state.integrity_audit = json.loads(audit_p.read_text(encoding="utf-8"))
                st.session_state.chart_specs = ChartEngine.generate_default_dashboard(
                    st.session_state.df, st.session_state.eda_summary
                )
                rep_pipe = SeniorDataScientistReportPipeline(output_dir=str(DEFAULT_OUTPUT_DIR))
                st.session_state.senior_report, _ = rep_pipe.generate_report(
                    st.session_state.df,
                    eda_summary=st.session_state.eda_summary,
                    integrity_audit=st.session_state.integrity_audit,
                    dataset_name=default_cleaned_path.name,
                )
                st.rerun()

        if st.session_state.df is not None:
            st.divider()
            st.subheader("2. Visual Component Builder")
            with st.expander(" Add Custom Apache ECharts Visual"):
                df_cols = list(st.session_state.df.columns)
                m_title = st.text_input("Component Title", "Custom Segment Analysis")
                m_type = st.selectbox("Visual Type", ChartEngine.SUPPORTED_TYPES)
                m_x = st.selectbox("X-Axis / Dimension", df_cols)
                m_y = st.selectbox("Y-Axis / Measure (optional)", [None] + df_cols)
                m_agg = st.selectbox("Aggregation", ["sum", "mean", "median", "count", "min", "max", "none"])
                m_topn = st.slider("Top-N Categories", min_value=3, max_value=25, value=10)
                m_width = st.radio(
                    "Card Width",
                    [6, 12],
                    format_func=lambda w: "Half Width (6)" if w == 6 else "Full Width (12)",
                    horizontal=True,
                )
                if st.button("Add Component to Dashboard", use_container_width=True):
                    execute_pipeline_action(
                        {
                            "action": "ADD_CHART",
                            "chart_spec": {
                                "title": m_title,
                                "chart_type": m_type,
                                "x": m_x,
                                "y": m_y,
                                "agg": m_agg,
                                "top_n": m_topn,
                                "width": m_width,
                            },
                        }
                    )
                    st.rerun()

            if st.button(" Reset Dashboard Layout", use_container_width=True):
                execute_pipeline_action({"action": "RESET_DASHBOARD"})
                st.rerun()

    if st.session_state.df is None:
        default_cleaned_path = DEFAULT_OUTPUT_DIR / "cleaned_visualization_data.csv"
        has_existing_dataset = default_cleaned_path.exists()

        # Render Modern Executive Hero Banner
        st.markdown(
            """
            <div class="hero-banner-card">
              <span class="hero-badge">Autonomous Data Engine  • Enterprise Suite</span>
              <h1 style="margin: 0 0 10px 0; font-size: 2.2rem; font-weight: 800; color: var(--ai-text);">
                Welcome to AutoInsight Enterprise
              </h1>
              <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: var(--ai-subtext); max-width: 820px;">
                Autonomous enterprise tabular data cleaning, deterministic algebraic reconciliation, 
                PowerBI-style resizable Matplotlib & Seaborn visual components, andt report synthesis.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Action Section
        if has_existing_dataset:
            st.markdown("###  Quick Start")
            c_act1, c_act2 = st.columns([1.8, 2.2])
            with c_act1:
                if st.button(" Launch with Existing Cleaned Dataset", type="primary", use_container_width=True):
                    df_existing = pd.read_csv(default_cleaned_path)
                    df_existing.columns = [c.strip("_") for c in df_existing.columns]
                    st.session_state.df = df_existing
                    st.session_state.dataset_name = default_cleaned_path.name
                    eda_p = DEFAULT_OUTPUT_DIR / "eda_reports" / "eda_summary.json"
                    audit_p = DEFAULT_OUTPUT_DIR / "eda_reports" / "data_integrity_audit.json"
                    if eda_p.exists():
                        st.session_state.eda_summary = json.loads(eda_p.read_text(encoding="utf-8"))
                    if audit_p.exists():
                        st.session_state.integrity_audit = json.loads(audit_p.read_text(encoding="utf-8"))
                    st.session_state.chart_specs = ChartEngine.generate_default_dashboard(
                        st.session_state.df, st.session_state.eda_summary
                    )
                    rep_pipe = SeniorDataScientistReportPipeline(output_dir=str(DEFAULT_OUTPUT_DIR))
                    st.session_state.senior_report, _ = rep_pipe.generate_report(
                        st.session_state.df,
                        eda_summary=st.session_state.eda_summary,
                        integrity_audit=st.session_state.integrity_audit,
                        dataset_name=default_cleaned_path.name,
                    )
                    st.rerun()
            with c_act2:
                st.caption(
                    " **Pre-Cleaned Dataset Ready:** `pipeline_output/cleaned_visualization_data.csv` is loaded and verified. "
                    "Click above to immediately launch the interactive dashboard, or upload a custom CSV/Excel/Parquet dataset via the sidebar."
                )

        st.markdown("###  Platform Capabilities")
        feat1, feat2, feat3, feat4 = st.columns(4)
        with feat1:
            st.markdown(
                """
                <div class="feature-card">
                  <h4> Universal Cleaning</h4>
                  <p>Zero false positives on 'Not' prefixes, accounting negative normalization, and composite key lookup reconstruction.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with feat2:
            st.markdown(
                """
                <div class="feature-card">
                  <h4> Algebraic Solver</h4>
                  <p>Discovers multiplicative and additive deterministic relationships (A × B = C) and reconciles mathematical discrepancies.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with feat3:
            st.markdown(
                """
                <div class="feature-card">
                  <h4> Publication Visuals</h4>
                  <p>PowerBI-style resizable Matplotlib & Seaborn cards with zero label clipping and adaptive light/dark palette rendering.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with feat4:
            st.markdown(
                """
                <div class="feature-card">
                  <h4> AI Co-Pilot</h4>
                  <p>Hybrid cloud LLM and offline deterministic NLP engine for statistical Q&A and natural language chart manipulation.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.stop()

    df: pd.DataFrame = st.session_state.df
    col_types = ChartEngine.classify_columns(df, mutate_dates=True)
    primary_num = ChartEngine.pick_primary_measure(col_types["numeric"])

    # Header & KPI Banner
    st.markdown(f"##  AutoInsight Executive Suite — `{st.session_state.dataset_name}`")

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Clean Records", f"{len(df):,}")
    kpi2.metric("Active Features", f"{len(df.columns)}")
    kpi3.metric("Remaining Nulls", f"{int(df.isna().sum().sum())}")
    kpi4.metric("Active Visuals", f"{len(st.session_state.chart_specs)}")
    kpi5.metric("Data Health Score", f"{st.session_state.integrity_audit.get('data_health_score', 100.0)}/100")
    if primary_num:
        kpi6.metric(f"Total {primary_num.replace('_', ' ').title()}", f"${df[primary_num].sum():,.2f}")
    else:
        kpi6.metric("Categories", f"{len(col_types['categorical'])}")

    st.divider()

    if st.session_state.show_copilot:
        dashboard_col, ai_col = st.columns([6.8, 3.2], gap="large")
    else:
        dashboard_col = st.container()
        ai_col = None

    with dashboard_col:
        tab_dash, tab_report, tab_sim, tab_data = st.tabs(
            [
                " Interactive Apache ECharts Dashboard",
                " Report",
                " Pivot & What-If Simulator",
                " Cleaned Data & Exports",
            ]
        )

        with tab_dash:
            filtered_df = df.copy()
            with st.expander("Global Multi-Dimensional Cross-Filter Bar", expanded=False):
                fc1, fc2, fc3 = st.columns(3)
                if col_types["categorical"]:
                    sel_cat = fc1.selectbox("Filter Dimension", ["(All)"] + col_types["categorical"])
                    if sel_cat != "(All)":
                        u_vals = sorted(filtered_df[sel_cat].dropna().astype(str).unique().tolist())
                        fc2.markdown("<div style='font-size: 14px; font-weight: 400; "
    "line-height: 1.5; margin-bottom: 8px;'>"
    "Included Segments</div>",
    unsafe_allow_html=True,)
                        with fc2.popover("Choose segments", use_container_width=True):
                            select_all = st.checkbox("Select All", value=True, key=f"sel_all_{sel_cat}")
                            picked_vals = []
                            for val in u_vals:
                                if st.checkbox(str(val), value=select_all, key=f"incl_seg_{sel_cat}_{val}"):
                                    picked_vals.append(val)
                        filtered_df = filtered_df[filtered_df[sel_cat].astype(str).isin(picked_vals)]
                if primary_num and not filtered_df.empty:
                    min_v, max_v = float(filtered_df[primary_num].min()), float(filtered_df[primary_num].max())
                    if min_v < max_v:
                        r_low, r_high = fc3.slider(
                            f"{primary_num} Range",
                            min_value=min_v,
                            max_value=max_v,
                            value=(min_v, max_v),
                        )
                        filtered_df = filtered_df[(filtered_df[primary_num] >= r_low) & (filtered_df[primary_num] <= r_high)]
                st.caption(f"Showing **{len(filtered_df):,}** of **{len(df):,}** records • Hover any chart for interactive crosshair tooltips • Use the top-right toolbox (🔄 Switch Type, 🔍 Zoom, 💾 Save PNG) • Drag bottom-right corner `↘` to resize.")

            telemetry_store: Dict[str, Dict[str, Any]] = {}
            specs: List[ChartSpec] = st.session_state.chart_specs

            def render_chart_card(sp: ChartSpec):
                with st.container(border=True):
                    hdr_l, hdr_edit, hdr_del = st.columns([8.0, 1.0, 1.0])
                    hdr_l.markdown(
                        f"<div class='chart-card-header'>ID: <b>{sp.chart_id}</b> | "
                        f"Engine: APACHE ECHARTS 5.5 ({sp.chart_type.upper()}) | Agg: {sp.agg.upper()}</div>",
                        unsafe_allow_html=True,
                    )
                    if hdr_del.button("🗑️", key=f"del_{sp.chart_id}", help=f"Remove {sp.chart_id}"):
                        execute_pipeline_action({"action": "REMOVE_CHART", "target_chart_id": sp.chart_id})
                        st.rerun()

                    with hdr_edit.popover("⚙️", help=f"Customize {sp.chart_id}"):
                        new_ct = st.selectbox(
                            "Chart Type",
                            ChartEngine.SUPPORTED_TYPES,
                            index=ChartEngine.SUPPORTED_TYPES.index(sp.chart_type) if sp.chart_type in ChartEngine.SUPPORTED_TYPES else 0,
                            key=f"ct_{sp.chart_id}",
                        )
                        new_agg = st.selectbox(
                            "Aggregation",
                            ["sum", "mean", "median", "count", "min", "max", "none"],
                            index=["sum", "mean", "median", "count", "min", "max", "none"].index(sp.agg) if sp.agg in ["sum", "mean", "median", "count", "min", "max", "none"] else 0,
                            key=f"agg_{sp.chart_id}",
                        )
                        new_w = st.radio(
                            "Column Span",
                            [6, 12],
                            index=0 if sp.width == 6 else 1,
                            key=f"w_{sp.chart_id}",
                            horizontal=True,
                        )
                        new_h = st.slider("Card Height (px)", 260, 720, sp.height_px, step=20, key=f"h_{sp.chart_id}")
                        if st.button("Apply", key=f"apply_{sp.chart_id}", use_container_width=True):
                            execute_pipeline_action({
                                "action": "UPDATE_CHART",
                                "target_chart_id": sp.chart_id,
                                "chart_spec": {"chart_type": new_ct, "agg": new_agg, "width": new_w, "height_px": new_h},
                            })
                            st.rerun()

                    opt_chart, telem = ChartEngine.render_chart(filtered_df, sp, theme=st.session_state.ui_theme)
                    telemetry_store[sp.chart_id] = telem
                    ChartEngine.display_chart(opt_chart, chart_id=sp.chart_id, height_px=sp.height_px, theme=st.session_state.ui_theme)
                    if sp.notes:
                        st.caption(f"💡 *{sp.notes}*")

            if st.session_state.show_copilot:
                # AI Co-Pilot ON: first chart sits beside the AI Co-Pilot panel in
                # the same row (dashboard_col ~2/3 vs ai_col ~1/3 => approx 2:1),
                # all remaining charts render full-width below it.
                if specs:
                    render_chart_card(specs[0])
                for sp in specs[1:]:
                    render_chart_card(sp)
            else:
                # AI Co-Pilot OFF: keep the existing layout as it is
                idx = 0
                while idx < len(specs):
                    current_spec = specs[idx]
                    if current_spec.width == 12 or idx + 1 >= len(specs) or specs[idx + 1].width == 12:
                        render_chart_card(current_spec)
                        idx += 1
                    else:
                        c_left, c_right = st.columns(2)
                        with c_left:
                            render_chart_card(specs[idx])
                        with c_right:
                            render_chart_card(specs[idx + 1])
                        idx += 2

            st.session_state.chart_telemetry = telemetry_store

        # TAB 2: SENIOR DATA SCIENTIST REPORT (DATA SUMMARY + CHARTS WITH INSIGHTS + CONCLUSION)
        with tab_report:
            if st.session_state.senior_report is None:
                rep_pipe = SeniorDataScientistReportPipeline(output_dir=str(DEFAULT_OUTPUT_DIR))
                st.session_state.senior_report, _ = rep_pipe.generate_report(
                    filtered_df,
                    eda_summary=st.session_state.eda_summary,
                    integrity_audit=st.session_state.integrity_audit,
                    dataset_name=st.session_state.dataset_name or "Dataset",
                )
            rep = st.session_state.senior_report

            st.subheader("1. Executive Data Summary & Statistical Profile")
            st.markdown(
                f"Across **{rep['total_rows']:,}** validated records and **{rep['total_columns']}** features, "
                f"cumulative **{(rep['primary_measure'] or 'value').replace('_', ' ')}** reaches **${rep['total_primary']:,.2f}** "
                f"(Mean: **${rep['mean_primary']:,.2f}** | Median: **${rep['median_primary']:,.2f}** | Data Integrity Score: **{rep['health_score']}/100**)."
            )
            if rep["numeric_profile"]:
                st.markdown("**Continuous Feature Parametric & Non-Parametric Profile**")
                st.dataframe(pd.DataFrame(rep["numeric_profile"]), use_container_width=True, hide_index=True)
            if rep["categorical_profile"]:
                st.markdown("**Categorical Dimension Dominance Profile**")
                st.dataframe(pd.DataFrame(rep["categorical_profile"]), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("2. Matplotlib & Seaborn Visualizations with Insights")
            for idx_v, vis in enumerate(rep["visual_insights"]):
                with st.container(border=True):
                    st.markdown(f"#### Figure {idx_v + 1}: {vis['title']}")
                    vc_img, vc_txt = st.columns([6.2, 3.8])
                    with vc_img:
                        chosen_img = vis["img_light"] if is_light else vis["img_dark"]
                        st.markdown(
                            f"""<div class="powerbi-resizable-card" style="height: 360px;">
                              <img src="{chosen_img}" alt="{vis['title']}" />
                              <div class="powerbi-resize-hint">↘ Drag corner to resize</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    with vc_txt:
                        st.markdown(f"**1. Empirical Statistical Finding**\n\n{vis['finding']}")
                        st.markdown(f"**2. Interpretation**\n\n{vis['interpretation']}")
                        st.markdown(f"**3. Strategic & Modeling Recommendation**\n\n{vis['recommendation']}")

            st.divider()
            st.subheader("3. Short Conclusion & Strategic Action Plan")
            st.markdown(f"**{rep['conclusion']['headline']}**")
            for pillar in rep["conclusion"]["pillars"]:
                st.markdown(f"- {pillar}")

            html_rep_path = (DEFAULT_OUTPUT_DIR / "eda_reports" / "senior_data_scientist_report.html")
            if html_rep_path.exists():
                
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    browser = p.chromium.launch()
                    page = browser.new_page(
                        # format="A4"
                    )
                    page.goto(
                        html_rep_path.resolve().as_uri(),
                        wait_until="networkidle",
                    )
                    page.pdf_path = None
                    page.emulate_media(media="print")
                    pdf_bytes = page.pdf(
                        format="A4",
                        print_background=True,
                        prefer_css_page_size=True,
                        margin={
                            "top": "12mm",
                            "bottom": "12mm",
                            "left": "10mm",
                            "right": "10mm",
                        },
                    )
                    browser.close()
                
                st.download_button(
                    label="Download Report (PDF)",
                    data=pdf_bytes,
                    file_name="senior_data_scientist_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        # TAB 3: PIVOT & WHAT-IF SIMULATOR
        with tab_sim:
            st.subheader(" Cross-Tab Pivot Heatmap & What-If Simulator")
            p_col1, p_col2, p_col3 = st.columns(3)
            low_cats_sim = [c for c in col_types["categorical"] if 2 <= filtered_df[c].nunique() <= 20]
            if len(low_cats_sim) >= 2 and primary_num:
                row_dim = p_col1.selectbox("Pivot Row Dimension", low_cats_sim, index=0)
                col_dim = p_col2.selectbox("Pivot Column Dimension", low_cats_sim, index=min(1, len(low_cats_sim) - 1))
                val_agg = p_col3.selectbox("Pivot Aggregation", ["sum", "mean", "median", "count"], index=0)
                if row_dim != col_dim:
                    piv = pd.pivot_table(
                        filtered_df,
                        values=primary_num,
                        index=row_dim,
                        columns=col_dim,
                        aggfunc=val_agg,
                        fill_value=0,
                    ).round(1)
                    row_cats = [str(r) for r in piv.index.tolist()]
                    col_cats = [str(c) for c in piv.columns.tolist()]
                    h_data = []
                    for i, r in enumerate(row_cats):
                        for j, c in enumerate(col_cats):
                            h_data.append([j, i, float(piv.iloc[i, j])])
                    min_val = float(piv.values.min())
                    max_val = float(piv.values.max())
                    is_d = (st.session_state.ui_theme == "dark")
                    piv_opt = {
                        "backgroundColor": "transparent",
                        "title": {"text": f"Cross-Tab Heatmap: {primary_num.title()} ({val_agg.upper()})", "textStyle": {"color": "#f8fafc" if is_d else "#0f172a", "fontSize": 14}},
                        "tooltip": {"position": "top", "formatter": "{b} : {c}"},
                        "toolbox": {"feature": {"saveAsImage": {}, "dataView": {"readOnly": True}}},
                        "grid": {"left": "3%", "right": "4%", "bottom": "18%", "top": "15%", "containLabel": True},
                        "xAxis": {"type": "category", "data": col_cats, "name": col_dim, "splitArea": {"show": True}, "axisLabel": {"color": "#cbd5e1" if is_d else "#334155"}},
                        "yAxis": {"type": "category", "data": row_cats, "name": row_dim, "splitArea": {"show": True}, "axisLabel": {"color": "#cbd5e1" if is_d else "#334155"}},
                        "visualMap": {
                            "min": min_val, "max": max_val if max_val > min_val else min_val + 1,
                            "calculable": True, "orient": "horizontal", "left": "center", "bottom": "0%",
                            "inRange": {"color": ["#1e3a8a", "#6366f1", "#ec4899"] if is_d else ["#dbeafe", "#3b82f6", "#1d4ed8"]},
                            "textStyle": {"color": "#94a3b8" if is_d else "#475569"}
                        },
                        "series": [{
                            "name": "Pivot Value",
                            "type": "heatmap",
                            "data": h_data,
                            "label": {"show": True, "color": "#f8fafc" if is_d else "#0f172a"},
                            "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.5)"}}
                        }]
                    }
                    ChartEngine.display_chart(piv_opt, chart_id="PIVOT_HEATMAP", height_px=380, theme=st.session_state.ui_theme)

            if primary_num:
                st.divider()
                st.markdown("#### 📈 Revenue / Metric What-If Elasticity Simulator")
                sim_c1, sim_c2, sim_c3 = st.columns(3)
                price_delta_pct = sim_c1.slider("Simulated Unit Price / Rate Change (%)", -30.0, 50.0, 5.0, 1.0)
                vol_delta_pct = sim_c2.slider("Simulated Volume / Demand Change (%)", -30.0, 50.0, 3.0, 1.0)
                baseline_total = float(filtered_df[primary_num].sum())
                projected_total = baseline_total * (1.0 + price_delta_pct / 100.0) * (1.0 + vol_delta_pct / 100.0)
                net_lift = projected_total - baseline_total
                sim_c3.metric(
                    f"Projected {primary_num.replace('_', ' ').title()}",
                    f"${projected_total:,.2f}",
                    delta=f"${net_lift:+,.2f} ({((projected_total/max(1,baseline_total))-1)*100:+.1f}%)",
                )

        # TAB 4: CLEANED DATA EXPLORER & EXPORTS
        with tab_data:
            st.subheader("Cleaned & Reconciled Dataset")
            search_term = st.text_input("🔍 Quick Search Across All Columns", "")
            view_df = filtered_df
            if search_term:
                mask = view_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                view_df = view_df[mask]
            st.dataframe(view_df, use_container_width=True, height=450)

            dl1, dl2 = st.columns(2)
            csv_bytes = view_df.to_csv(index=False).encode("utf-8")
            dl1.download_button(
                "⬇️ Download Cleaned Dataset (CSV)",
                data=csv_bytes,
                file_name="auto_insight_cleaned_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
            html_report_path = DEFAULT_OUTPUT_DIR / "eda_reports" / "senior_data_scientist_report.html"
            if html_report_path.exists():
                dl2.download_button(
                    " Download  Report ",
                    data=html_report_path.read_bytes(),
                    file_name="senior_data_scientist_report.html",
                    mime="text/html",
                    use_container_width=True,
                )

    # AI Co-Pilot Panel + Visual Components Below Chatbot (Zero Vacant Space!)
    if ai_col is not None:
        with ai_col:
            st.subheader(" AutoInsight AI Co-Pilot")
            st.caption("Reads live chart telemetry & controls Matplotlib/Seaborn visuals via natural language.")

            q_col1, q_col2 = st.columns(2)
            quick_prompt = None
            if q_col1.button(" Executive Summary", use_container_width=True):
                quick_prompt = "Read all active charts on the dashboard and give me an executive summary with exact numbers."
            if q_col2.button(" Spot Anomalies", use_container_width=True):
                quick_prompt = "Based on the active charts and statistical distributions, what are the biggest outliers or anomalies?"

            chat_box = st.container(height=420, border=True)
            with chat_box:
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        if msg.get("badge"):
                            st.caption(f" **Action Executed:** `{msg['badge']}`")

            user_query = st.chat_input("Ask about any chart or add/edit a visual...")
            active_query = quick_prompt or user_query

            if active_query:
                st.session_state.chat_history.append({"role": "user", "content": active_query})
                ai_controller = AIController()
                with st.spinner("Inspecting telemetry & orchestrating pipeline..."):
                    ai_result = ai_controller.query(
                        user_prompt=active_query,
                        chat_history=st.session_state.chat_history[:-1],
                        df=filtered_df,
                        col_types=col_types,
                        chart_specs=st.session_state.chart_specs,
                        chart_telemetry=st.session_state.chart_telemetry,
                        eda_summary=st.session_state.eda_summary,
                    )

                action_taken = ai_result.get("action", "NONE")
                if action_taken and action_taken != "NONE":
                    execute_pipeline_action(ai_result)
                    badge_text = f"{action_taken} (Target: {ai_result.get('target_chart_id') or 'New Visual'})"
                else:
                    badge_text = None

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": ai_result.get("reply", "Done."),
                        "badge": badge_text,
                    }
                )
                st.rerun()

            # Utilize the space below the chatbot for live visual components!
            st.divider()
            st.markdown("####  Live Companion Visual Components")
            if col_types["categorical"] and primary_num:
                comp_cat = col_types["categorical"][-1]
                opt_s1, _ = ChartEngine.render_chart(
                    filtered_df,
                    ChartSpec(
                        chart_id="SIDE_1",
                        title=f"Mean {primary_num.replace('_', ' ').title()} by {comp_cat.replace('_', ' ').title()}",
                        chart_type="horizontal_bar",
                        x=comp_cat,
                        y=primary_num,
                        agg="mean",
                        top_n=7,
                        width=6,
                    ),
                    theme=st.session_state.ui_theme,
                )
                ChartEngine.display_chart(opt_s1, chart_id="SIDE_1", height_px=280, theme=st.session_state.ui_theme)

            if len(col_types["numeric"]) >= 2:
                sec_num = col_types["numeric"][0] if col_types["numeric"][0] != primary_num else col_types["numeric"][1]
                opt_s2, _ = ChartEngine.render_chart(
                    filtered_df,
                    ChartSpec(
                        chart_id="SIDE_2",
                        title=f"Distribution of {sec_num.replace('_', ' ').title()}",
                        chart_type="area",
                        x=sec_num,
                        y=primary_num,
                        width=6,
                    ),
                    theme=st.session_state.ui_theme,
                )
                ChartEngine.display_chart(opt_s2, chart_id="SIDE_2", height_px=280, theme=st.session_state.ui_theme)


if __name__ == "__main__":
    run_streamlit_app()
