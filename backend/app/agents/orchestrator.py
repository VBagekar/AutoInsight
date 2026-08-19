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

LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB
PROFILE_SAMPLE_MAX_ROWS = 200_000
CHUNK_SIZE = 100_000


class MasterOrchestrator:
    def __init__(self) -> None:
        # Deliberately in-memory for a single-user local app.  A production
        # deployment should replace this with object storage + a database.
        self.datasets: Dict[str, Dict[str, Any]] = {}

    def _detect_schema_drift(self, filename: str, new_df: pd.DataFrame) -> Dict[str, Any] | None:
        """Compare new dataframe columns against existing dataset with same filename."""
        for ds in self.datasets.values():
            if ds.get("filename") == filename:
                old_df = ds["df"]
                old_cols = set(old_df.columns)
                new_cols = set(new_df.columns)
                
                columns_added = sorted(list(new_cols - old_cols))
                columns_removed = sorted(list(old_cols - new_cols))
                
                columns_type_changed = []
                for col in sorted(old_cols & new_cols):
                    old_dtype = str(old_df[col].dtype)
                    new_dtype = str(new_df[col].dtype)
                    if old_dtype != new_dtype:
                        columns_type_changed.append({
                            "column": col,
                            "old_type": old_dtype,
                            "new_type": new_dtype
                        })
                
                return {
                    "columns_added": columns_added,
                    "columns_removed": columns_removed,
                    "columns_type_changed": columns_type_changed
                }
        return None

    def process_file_and_generate_initial_dashboard(self, file_bytes: bytes, filename: str, sheet_name: str | None = None) -> Dict[str, Any]:
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        available_sheets: list[str] = []
        target_sheet = None
        is_large_file = len(file_bytes) > LARGE_FILE_THRESHOLD_BYTES
        
        if suffix == "csv":
            if is_large_file:
                chunks = pd.read_csv(BytesIO(file_bytes), chunksize=CHUNK_SIZE)
                raw_df = pd.concat(chunks, ignore_index=True)
            else:
                raw_df = pd.read_csv(BytesIO(file_bytes))
        elif suffix in {"xlsx", "xls"}:
            excel_file = pd.ExcelFile(BytesIO(file_bytes))
            available_sheets = excel_file.sheet_names
            target_sheet = sheet_name if sheet_name in available_sheets else available_sheets[0]
            raw_df = pd.read_excel(excel_file, sheet_name=target_sheet)
            # Excel: openpyxl loads fully; sampling for profiling only if large
            is_large_file = len(file_bytes) > LARGE_FILE_THRESHOLD_BYTES
        else:
            raise ValueError("Please upload a CSV or Excel (.xlsx/.xls) file.")
        if raw_df.empty:
            raise ValueError("The uploaded dataset has no rows.")

        # Schema drift detection (compare raw columns before cleaning)
        schema_drift = self._detect_schema_drift(filename, raw_df)

        cleaned_df, cleaning_report = data_cleaner.clean_dataset(raw_df)
        
        # For profiling large datasets, use a random sample
        profile_df = cleaned_df
        was_sampled = False
        sample_size = len(cleaned_df)
        if is_large_file and len(cleaned_df) > PROFILE_SAMPLE_MAX_ROWS:
            profile_df = cleaned_df.sample(n=PROFILE_SAMPLE_MAX_ROWS, random_state=42)
            was_sampled = True
            sample_size = len(profile_df)
        
        summary = dataset_profiler.profile_csv(
            file_bytes, filename, profile_df, 
            cleaning_report=cleaning_report,
            was_sampled=was_sampled,
            sample_size=sample_size,
            total_rows=len(cleaned_df)
        )
        dataset_id = str(uuid4())
        self.datasets[dataset_id] = {"df": cleaned_df, "summary": summary, "cleaning_report": cleaning_report, "sheet_name": target_sheet if suffix in {"xlsx", "xls"} else None, "filename": filename}

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
            "available_sheets": available_sheets,
            "schema_drift": schema_drift,
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

    def get_dataset_preview(self, dataset_id: str, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        dataset = self.datasets.get(dataset_id)
        if not dataset:
            raise KeyError("Dataset not found")
        
        df = dataset["df"]
        total_rows = len(df)
        page_size = min(page_size, 500)
        page = max(page, 1)
        total_pages = (total_rows + page_size - 1) // page_size
        page = min(page, total_pages) if total_pages > 0 else 1
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        
        page_df = df.iloc[start_idx:end_idx]
        
        rows = page_df.replace({pd.NA: None, pd.NaT: None, float('nan'): None}).where(pd.notnull(page_df), None).to_dict(orient="records")
        
        return {
            "columns": list(df.columns),
            "rows": rows,
            "total_rows": total_rows,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "cleaning_summary": dataset["summary"].get("cleaning_summary")
        }


master_orchestrator = MasterOrchestrator()
