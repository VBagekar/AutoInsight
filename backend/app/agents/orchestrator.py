"""Application service for the upload → clean → analyse → dashboard workflow.

Person 2 owns: query understanding, intent parsing, conversation memory, and
LLM orchestration.  All chart values are computed by DashboardBuilder from the
cleaned dataframe — the LLM never produces numbers.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from uuid import uuid4

import pandas as pd

from app.core.dataset_profiler import dataset_profiler
from app.core.data_cleaner import data_cleaner
from app.core.dashboard_builder import dashboard_builder
from app.core.llm_client import nemotron_client
from app.agents.forecast_agent import forecasting_agent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB
PROFILE_SAMPLE_MAX_ROWS = 200_000
CHUNK_SIZE = 100_000

# ---------------------------------------------------------------------------
# Logging / Telemetry setup
# ---------------------------------------------------------------------------
_log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Telemetry file handler (JSON lines)
_telemetry_path = _log_dir / "query_telemetry.jsonl"
_telemetry_handler = logging.FileHandler(str(_telemetry_path), encoding="utf-8")
_telemetry_handler.setLevel(logging.INFO)
_telemetry_handler.setFormatter(logging.Formatter("%(message)s"))

_telemetry_logger = logging.getLogger("query_telemetry")
_telemetry_logger.setLevel(logging.INFO)
_telemetry_logger.addHandler(_telemetry_handler)
_telemetry_logger.propagate = False


# ===================================================================
#  INTENT PARSER — replaces the single "current year" hack
# ===================================================================
class IntentParser:
    """Parse a natural-language query against a dataset schema to extract
    structured intent: time filters, comparisons, metrics, dimensions, and
    simple value filters.
    """

    # Relative time patterns
    _RELATIVE_TIME_PATTERNS = [
        # "last N months/years/quarters/weeks/days"
        (re.compile(r"\blast\s+(\d+)\s+(month|year|quarter|week|day)s?\b", re.I), "last_n"),
        # "past N months/years"
        (re.compile(r"\bpast\s+(\d+)\s+(month|year|quarter|week|day)s?\b", re.I), "last_n"),
        # "last quarter"
        (re.compile(r"\blast\s+quarter\b", re.I), "last_quarter"),
        # "this quarter"
        (re.compile(r"\bthis\s+quarter\b", re.I), "this_quarter"),
        # "last year"
        (re.compile(r"\blast\s+year\b", re.I), "last_year"),
        # "this year / current year / latest year"
        (re.compile(r"\b(?:this|current|latest)\s+year\b", re.I), "this_year"),
        # "year to date / YTD"
        (re.compile(r"\b(?:year\s+to\s+date|ytd)\b", re.I), "ytd"),
        # "month over month / MoM"
        (re.compile(r"\b(?:month\s+over\s+month|mom)\b", re.I), "mom"),
        # "year over year / YoY"
        (re.compile(r"\b(?:year\s+over\s+year|yoy)\b", re.I), "yoy"),
        # "in 2024" / "for 2023"
        (re.compile(r"\b(?:in|for|of)\s+(20\d{2})\b", re.I), "specific_year"),
    ]

    # Comparison patterns
    _COMPARE_PATTERN = re.compile(
        r"\b(?:compare|vs\.?|versus|against|difference\s+between)\b", re.I
    )

    # Filter patterns: "in the North region", "where category is X", "for product Y"
    _FILTER_PATTERNS = [
        re.compile(r"\bwhere\s+(\w+)\s+(?:is|=|equals?)\s+[\"']?(.+?)[\"']?\s*$", re.I),
        re.compile(r"\bin\s+(?:the\s+)?[\"']?(.+?)[\"']?\s+(\w+)\s*$", re.I),
        re.compile(r"\bfor\s+(?:the\s+)?[\"']?(.+?)[\"']?\s*$", re.I),
    ]

    # Top/bottom pattern
    _TOP_BOTTOM = re.compile(r"\b(top|bottom|best|worst|highest|lowest)\s*(\d+)?\b", re.I)

    @classmethod
    def parse(cls, query: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Return a structured QueryIntent dict."""
        lower = query.lower().strip()
        numeric = summary.get("numeric_columns", [])
        dates = summary.get("date_columns", [])
        categories = summary.get("categorical_columns", [])
        all_cols = numeric + dates + categories

        intent: Dict[str, Any] = {
            "raw_query": query,
            "time_filter": None,
            "comparison": None,
            "filters": [],
            "metrics": [],
            "dimensions": [],
            "top_bottom": None,
            "is_overview": False,
        }

        # --- Time filter ---
        intent["time_filter"] = cls._parse_time_filter(lower)

        # --- Metrics extraction (numeric columns mentioned in query) ---
        for col in numeric:
            if col.lower() in lower or col.lower().replace("_", " ") in lower:
                intent["metrics"].append(col)

        # --- Dimensions extraction (categorical columns mentioned in query) ---
        for col in categories:
            if col.lower() in lower or col.lower().replace("_", " ") in lower:
                intent["dimensions"].append(col)

        # --- Comparison detection ---
        if cls._COMPARE_PATTERN.search(lower):
            # Try to find "X vs Y" style
            vs_match = re.search(r"(\w+)\s+(?:vs\.?|versus|against)\s+(\w+)", lower, re.I)
            if vs_match:
                a, b = vs_match.group(1), vs_match.group(2)
                matched_a = cls._match_column(a, all_cols)
                matched_b = cls._match_column(b, all_cols)
                if matched_a and matched_b:
                    intent["comparison"] = {"columns": [matched_a, matched_b]}

            # If no explicit comparison columns found but "compare" is in query,
            # use extracted metrics
            if not intent["comparison"] and len(intent["metrics"]) >= 2:
                intent["comparison"] = {"columns": intent["metrics"][:2]}

        # --- Value filters ---
        intent["filters"] = cls._parse_filters(lower, categories, summary)

        # --- Top / Bottom ---
        top_match = cls._TOP_BOTTOM.search(lower)
        if top_match:
            direction = top_match.group(1).lower()
            n = int(top_match.group(2)) if top_match.group(2) else 5
            intent["top_bottom"] = {
                "direction": "top" if direction in ("top", "best", "highest") else "bottom",
                "n": n,
            }

        # --- Overview detection ---
        overview_keywords = ("everything", "overview", "analyze", "analyse", "dashboard", "all charts", "full analysis", "show me", "insights")
        if any(kw in lower for kw in overview_keywords) and not intent["metrics"]:
            intent["is_overview"] = True

        # --- Default metric if none detected ---
        if not intent["metrics"] and summary.get("primary_kpi"):
            intent["metrics"] = [summary["primary_kpi"]]

        return intent

    @classmethod
    def _parse_time_filter(cls, lower: str) -> Optional[Dict[str, Any]]:
        """Extract a time filter from the query string."""
        for pattern, filter_type in cls._RELATIVE_TIME_PATTERNS:
            match = pattern.search(lower)
            if match:
                if filter_type == "last_n":
                    n = int(match.group(1))
                    unit = match.group(2).lower()
                    return {"type": "last_n", "n": n, "unit": unit}
                elif filter_type == "specific_year":
                    return {"type": "year", "value": int(match.group(1))}
                elif filter_type == "last_quarter":
                    return {"type": "last_quarter"}
                elif filter_type == "this_quarter":
                    return {"type": "this_quarter"}
                elif filter_type == "last_year":
                    return {"type": "last_year"}
                elif filter_type in ("this_year", "ytd"):
                    return {"type": filter_type}
                elif filter_type in ("yoy", "mom"):
                    return {"type": filter_type}
        return None

    @classmethod
    def _parse_filters(cls, lower: str, categories: List[str], summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract simple column=value filters from the query."""
        filters: List[Dict[str, Any]] = []
        col_meta = summary.get("col_meta", {})

        # Strategy: for each categorical column, check if any of its known
        # values appear in the query
        for cat_col in categories:
            meta = col_meta.get(cat_col, {})
            sample_values = meta.get("sample_values", [])
            # Also check top_values if available
            top_values = meta.get("top_values", [])
            known_values = list(set(str(v) for v in (sample_values + top_values) if v))

            for val in known_values:
                val_str = str(val).strip()
                if len(val_str) >= 2 and val_str.lower() in lower:
                    filters.append({
                        "column": cat_col,
                        "operator": "==",
                        "value": val_str,
                    })
                    break  # one filter per column

        return filters

    @classmethod
    def _match_column(cls, text: str, columns: List[str]) -> Optional[str]:
        """Case-insensitive column name matching."""
        text_lower = text.lower().strip()
        for col in columns:
            if col.lower() == text_lower or col.lower().replace("_", " ") == text_lower:
                return col
        return None

    @classmethod
    def apply_time_filter(cls, df: pd.DataFrame, date_col: str, time_filter: Dict[str, Any]) -> pd.DataFrame:
        """Apply a parsed time filter to a dataframe."""
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        if not parsed.notna().any():
            return df

        latest_date = parsed.max()
        ftype = time_filter.get("type")

        if ftype == "this_year":
            latest_year = int(latest_date.year)
            return df.loc[parsed.dt.year == latest_year].copy()

        elif ftype == "last_year":
            latest_year = int(latest_date.year)
            return df.loc[parsed.dt.year == (latest_year - 1)].copy()

        elif ftype == "year" and "value" in time_filter:
            return df.loc[parsed.dt.year == time_filter["value"]].copy()

        elif ftype == "ytd":
            latest_year = int(latest_date.year)
            return df.loc[parsed.dt.year == latest_year].copy()

        elif ftype == "last_n":
            n = time_filter.get("n", 1)
            unit = time_filter.get("unit", "month")
            if unit == "month":
                cutoff = latest_date - pd.DateOffset(months=n)
            elif unit == "year":
                cutoff = latest_date - pd.DateOffset(years=n)
            elif unit == "quarter":
                cutoff = latest_date - pd.DateOffset(months=n * 3)
            elif unit == "week":
                cutoff = latest_date - timedelta(weeks=n)
            elif unit == "day":
                cutoff = latest_date - timedelta(days=n)
            else:
                cutoff = latest_date - pd.DateOffset(months=n)
            return df.loc[parsed >= cutoff].copy()

        elif ftype == "last_quarter":
            q = (latest_date.month - 1) // 3  # 0-based current quarter
            if q == 0:
                start = pd.Timestamp(year=latest_date.year - 1, month=10, day=1)
                end = pd.Timestamp(year=latest_date.year - 1, month=12, day=31)
            else:
                start_month = (q - 1) * 3 + 1
                start = pd.Timestamp(year=latest_date.year, month=start_month, day=1)
                end_month = q * 3
                end = pd.Timestamp(year=latest_date.year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
            return df.loc[(parsed >= start) & (parsed <= end)].copy()

        elif ftype == "this_quarter":
            q = (latest_date.month - 1) // 3
            start_month = q * 3 + 1
            start = pd.Timestamp(year=latest_date.year, month=start_month, day=1)
            return df.loc[parsed >= start].copy()

        return df

    @classmethod
    def apply_value_filters(cls, df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
        """Apply column=value filters to a dataframe."""
        result = df
        for f in filters:
            col = f.get("column")
            op = f.get("operator", "==")
            val = f.get("value")
            if col not in result.columns:
                continue
            try:
                if op == "==":
                    result = result.loc[result[col].astype(str).str.lower() == str(val).lower()].copy()
                elif op == "!=":
                    result = result.loc[result[col].astype(str).str.lower() != str(val).lower()].copy()
                elif op == ">":
                    result = result.loc[result[col] > float(val)].copy()
                elif op == "<":
                    result = result.loc[result[col] < float(val)].copy()
                elif op == ">=":
                    result = result.loc[result[col] >= float(val)].copy()
                elif op == "<=":
                    result = result.loc[result[col] <= float(val)].copy()
            except (ValueError, TypeError):
                continue
        return result


# ===================================================================
#  CONVERSATION MEMORY
# ===================================================================
class ConversationMemory:
    """Stores per-dataset conversation history for follow-up query support."""

    def __init__(self) -> None:
        # dataset_id -> list of {query, intent, timestamp}
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add(self, dataset_id: str, query: str, intent: Dict[str, Any]) -> None:
        self._history[dataset_id].append({
            "query": query,
            "intent": intent,
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Keep last 20 turns max per dataset
        if len(self._history[dataset_id]) > 20:
            self._history[dataset_id] = self._history[dataset_id][-20:]

    def get_context(self, dataset_id: str) -> List[Dict[str, str]]:
        """Return conversation context for the master prompt."""
        return [{"query": h["query"]} for h in self._history.get(dataset_id, [])]

    def get_last_intent(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent intent for follow-up resolution."""
        history = self._history.get(dataset_id, [])
        return history[-1]["intent"] if history else None

    def resolve_followup(self, dataset_id: str, current_intent: Dict[str, Any], query: str) -> Dict[str, Any]:
        """If the current query looks like a follow-up, merge with previous intent."""
        last = self.get_last_intent(dataset_id)
        if not last:
            return current_intent

        lower = query.lower().strip()
        is_followup = (
            len(lower.split()) <= 6  # short query
            or lower.startswith(("now ", "and ", "also ", "but ", "what about", "how about"))
            or any(word in lower for word in ("that", "those", "instead", "same", "break"))
        )

        if not is_followup:
            return current_intent

        # Merge: carry forward metrics/dimensions/time_filter from last intent
        merged = dict(current_intent)

        # If no metrics in current, reuse last
        if not merged["metrics"] and last.get("metrics"):
            merged["metrics"] = last["metrics"]

        # If no time filter in current, reuse last
        if not merged["time_filter"] and last.get("time_filter"):
            merged["time_filter"] = last["time_filter"]

        # If new dimensions are found, they ADD to the analysis (not replace)
        if merged["dimensions"] and last.get("dimensions"):
            # Keep new dimensions (user wants to "break down by" something new)
            pass
        elif not merged["dimensions"] and last.get("dimensions"):
            merged["dimensions"] = last["dimensions"]

        # If no filters in current, reuse last
        if not merged["filters"] and last.get("filters"):
            merged["filters"] = last["filters"]

        return merged

    def clear(self, dataset_id: str) -> None:
        self._history.pop(dataset_id, None)


# ===================================================================
#  QUERY TELEMETRY
# ===================================================================
def _log_query_telemetry(
    dataset_id: str,
    query: str,
    intent: Dict[str, Any],
    plan_source: str,
    chart_count: int,
) -> None:
    """Write a JSON-line entry to the telemetry log."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset_id": dataset_id,
        "query": query,
        "intent": {
            "time_filter": intent.get("time_filter"),
            "metrics": intent.get("metrics"),
            "dimensions": intent.get("dimensions"),
            "comparison": intent.get("comparison"),
            "filters_count": len(intent.get("filters", [])),
            "is_overview": intent.get("is_overview", False),
        },
        "plan_source": plan_source,
        "charts_materialized": chart_count,
    }
    try:
        _telemetry_logger.info(json.dumps(entry, ensure_ascii=True))
    except Exception:
        pass  # telemetry must never break the main flow


# ===================================================================
#  MASTER ORCHESTRATOR
# ===================================================================
class MasterOrchestrator:
    def __init__(self) -> None:
        # Deliberately in-memory for a single-user local app.  A production
        # deployment should replace this with object storage + a database.
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.conversation_memory = ConversationMemory()

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

    def _frame_for_query(self, df: pd.DataFrame, summary: Dict[str, Any], intent: Dict[str, Any]) -> pd.DataFrame:
        """Apply parsed intent filters (time, value) to narrow the dataframe before charting."""
        result = df

        # Apply time filter
        time_filter = intent.get("time_filter")
        dates = summary.get("date_columns", [])
        if time_filter and dates:
            result = IntentParser.apply_time_filter(result, dates[0], time_filter)

        # Apply value filters
        filters = intent.get("filters", [])
        if filters:
            result = IntentParser.apply_value_filters(result, filters)

        # If after filtering we have zero rows, fall back to original
        if result.empty:
            return df

        return result

    def process_query_stream(self, dataset_id: str, query: str) -> Generator[Dict[str, Any], None, None]:
        dataset = self.datasets.get(dataset_id)
        if not dataset:
            yield {"type": "error", "message": "This dataset is no longer available. Upload it again to continue."}
            return

        summary = dataset["summary"]

        # --- Step 1: Parse intent ---
        intent = IntentParser.parse(query, summary)

        # --- Step 2: Resolve follow-ups via conversation memory ---
        intent = self.conversation_memory.resolve_followup(dataset_id, intent, query)

        # --- Step 3: Store in conversation memory ---
        self.conversation_memory.add(dataset_id, query, intent)

        # --- Step 4: Apply filters to dataframe ---
        df = self._frame_for_query(dataset["df"], summary, intent)

        yield {"type": "thinking", "content": "Reading the cleaned local dataset and selecting valid fields for your request…"}

        # --- Step 5: Get chart plan (LLM with master prompt, or fallback) ---
        conversation_context = self.conversation_memory.get_context(dataset_id)
        llm_plan = nemotron_client.generate_chart_plan(summary, query, conversation_context)
        plan_source = "llm" if llm_plan else "fallback"
        plan = llm_plan or dashboard_builder.heuristic_plan(summary, query, intent)

        # --- Step 6: Materialize charts ---
        charts = []
        for index, spec in enumerate(plan):
            chart = dashboard_builder.materialize_chart(df, summary, spec, f"query-{index + 1}")
            if chart:
                charts.append(chart)
        if not charts:
            charts = [chart for i, spec in enumerate(dashboard_builder.default_plan(summary)) if (chart := dashboard_builder.materialize_chart(df, summary, spec, f"fallback-{i + 1}"))]

        kpis = dashboard_builder.make_kpis(df, summary)
        report = self._report(query, summary, dataset["cleaning_report"], charts, kpis)

        # --- Step 7: Log telemetry ---
        _log_query_telemetry(dataset_id, query, intent, plan_source, len(charts))

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
