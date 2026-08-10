"""Application service for the upload → clean → analyse → dashboard workflow."""
from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Generator
from uuid import uuid4

import pandas as pd

from app.core.dataset_profiler import dataset_profiler
from app.core.data_cleaner import data_cleaner
from app.core.dashboard_builder import dashboard_builder
from app.core.llm_client import nemotron_client
from app.agents.forecast_agent import forecasting_agent


class MasterOrchestrator:
    def __init__(self) -> None:
        # Deliberately in-memory for a single-user local app.  A production
        # deployment should replace this with object storage + a database.
        self.datasets: Dict[str, Dict[str, Any]] = {}

    def process_file_and_generate_initial_dashboard(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix == "csv":
            raw_df = pd.read_csv(BytesIO(file_bytes))
        elif suffix in {"xlsx", "xls"}:
            raw_df = pd.read_excel(BytesIO(file_bytes))
        else:
            raise ValueError("Please upload a CSV or Excel (.xlsx/.xls) file.")
        if raw_df.empty:
            raise ValueError("The uploaded dataset has no rows.")

        cleaned_df, cleaning_report = data_cleaner.clean_dataset(raw_df)
        summary = dataset_profiler.profile_csv(file_bytes, filename, cleaned_df, cleaning_report=cleaning_report)
        dataset_id = str(uuid4())
        self.datasets[dataset_id] = {"df": cleaned_df, "summary": summary, "cleaning_report": cleaning_report}

        charts = []
        for index, spec in enumerate(dashboard_builder.default_plan(summary)):
            chart = dashboard_builder.materialize_chart(cleaned_df, summary, spec, f"initial-{index + 1}")
            if chart:
                charts.append(chart)
        kpis = dashboard_builder.make_kpis(cleaned_df, summary)
        forecast = self._forecast_for_dataset(cleaned_df, summary)
        return {
            "status": "success", "dataset_id": dataset_id, "summary": summary,
            "cleaning_report": cleaning_report, "charts": charts, "kpi_summary": kpis,
            "forecast": forecast,
            "ai_insights": self._initial_insights(summary, cleaning_report, charts),
        }

    def _initial_insights(self, summary: Dict[str, Any], cleaning: Dict[str, Any], charts: list[Dict[str, Any]]) -> list[str]:
        primary = summary.get("primary_kpi") or "primary metric"
        return [
            f"Cleaned {cleaning['cleaned_rows']:,} rows; removed {cleaning['duplicates_removed']:,} duplicate rows.",
            f"Detected {primary} as the primary KPI and generated {len(charts)} data-backed visuals.",
            f"Data completeness score: {summary.get('quality_score', 0)}%.",
        ]

    def _forecast_for_dataset(self, df: pd.DataFrame, summary: Dict[str, Any]) -> Dict[str, Any] | None:
        metric, dates = summary.get("primary_kpi"), summary.get("date_columns", [])
        if not metric or not dates:
            return None
        series = dashboard_builder._period_series(df, dates[0], metric)
        values = [point["value"] for point in series]
        return forecasting_agent.forecast_metric(values, periods=4) if len(values) >= 2 else None

    def _frame_for_query(self, df: pd.DataFrame, summary: Dict[str, Any], query: str) -> pd.DataFrame:
        """Apply unambiguous temporal language before calculating chart values."""
        lower = query.lower()
        dates = summary.get("date_columns", [])
        if not dates or not any(phrase in lower for phrase in ("current year", "this year", "latest year")):
            return df
        date_col = dates[0]
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        if not parsed.notna().any():
            return df
        # "Current" is the newest year present in the uploaded data. This is
        # preferable to an empty calendar-year filter for historical datasets.
        latest_year = int(parsed.max().year)
        return df.loc[parsed.dt.year == latest_year].copy()

    def process_query_stream(self, dataset_id: str, query: str) -> Generator[Dict[str, Any], None, None]:
        dataset = self.datasets.get(dataset_id)
        if not dataset:
            yield {"type": "error", "message": "This dataset is no longer available. Upload it again to continue."}
            return
        df, summary = self._frame_for_query(dataset["df"], dataset["summary"], query), dataset["summary"]
        yield {"type": "thinking", "content": "Reading the cleaned local dataset and selecting valid fields for your request…"}

        # Nemotron plans the layout using the compact schema only.  If it is
        # unavailable/malformed, the deterministic planner still returns a real dashboard.
        llm_plan = nemotron_client.generate_chart_plan(summary, query)
        plan = llm_plan or dashboard_builder.heuristic_plan(summary, query)
        charts = []
        for index, spec in enumerate(plan):
            chart = dashboard_builder.materialize_chart(df, summary, spec, f"query-{index + 1}")
            if chart:
                charts.append(chart)
        if not charts:
            charts = [chart for i, spec in enumerate(dashboard_builder.default_plan(summary)) if (chart := dashboard_builder.materialize_chart(df, summary, spec, f"fallback-{i + 1}"))]

        kpis = dashboard_builder.make_kpis(df, summary)
        report = self._report(query, summary, dataset["cleaning_report"], charts, kpis)
        yield {"type": "payload", "data": {
            "dashboard_title": f"{kpis['primary_kpi']} analysis",
            "suggested_charts": charts,
            "kpi_summary": kpis,
            "ai_recommendations": [chart["insight_tooltip"] for chart in charts[:3]],
            "detailed_report": report,
            "forecast": self._forecast_for_dataset(df, summary),
        }}

    def _report(self, query: str, summary: Dict[str, Any], cleaning: Dict[str, Any], charts: list[Dict[str, Any]], kpis: Dict[str, Any]) -> str:
        findings = "\n".join(f"- {chart['insight_tooltip']}" for chart in charts)
        return (
            f"# Analysis report\n\n**Question:** {query}\n\n"
            f"The dataset contains {summary['row_count']:,} source rows and {summary['column_count']} columns. "
            f"After automated cleaning, {cleaning['cleaned_rows']:,} rows remain. "
            f"The selected primary KPI is **{kpis['primary_kpi']}** with an aggregate value of **{kpis['value']:,.2f}**.\n\n"
            f"## Key findings\n{findings}\n\n"
            "All displayed values are computed locally from the cleaned uploaded dataset."
        )


master_orchestrator = MasterOrchestrator()
