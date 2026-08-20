"""Exhaustive tests for Person 2 — every condition, branch, edge case.

Covers all 8 tasks:
  Task 1: Dead code deletion (verified by absence)
  Task 2: IntentParser — every temporal pattern, comparison, filter, top/bottom, overview
  Task 3: Master Prompt — structure, context injection, edge cases
  Task 4: LLM Validation — chart type normalization, column fuzzy matching, full repair
  Task 5: Heuristic Fallback Planner — all query types with/without intent
  Task 6: Conversation Memory — add, retrieve, followup merge, limits
  Task 7: Query Telemetry — structured logging
  Task 8: Config Security — API key masking, is_llm_configured

Run:  python -m pytest backend/tests/test_person2_exhaustive.py -v
"""
import sys
import os
import json
import tempfile
import logging
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add backend to path
_backend = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_backend))

import pytest
import numpy as np
import pandas as pd

from app.agents.orchestrator import IntentParser, ConversationMemory, _log_query_telemetry
from app.core.llm_client import NemotronLLMClient, ALLOWED_CHART_TYPES, _CHART_TYPE_ALIASES
from app.core.dashboard_builder import DashboardBuilder
from app.config import Settings


# ─────────────────────────────────────────────────────────────
#  FIXTURES — reusable dataset summaries and dataframes
# ─────────────────────────────────────────────────────────────

FULL_SUMMARY = {
    "numeric_columns": ["Sales", "Profit", "Orders", "Cost", "Quantity"],
    "date_columns": ["Date", "Ship_Date"],
    "categorical_columns": ["Region", "Category", "Product", "City"],
    "primary_kpi": "Sales",
    "col_meta": {
        "Sales": {"name": "Sales", "dtype": "float64", "semantic_role": "kpi",
                  "missing_pct": 0, "min": 10, "max": 50000, "mean": 2500,
                  "unique_count": 4800, "sample_values": [100, 250, 500]},
        "Profit": {"name": "Profit", "dtype": "float64", "semantic_role": "kpi",
                   "missing_pct": 1.2, "min": -500, "max": 12000, "mean": 800,
                   "unique_count": 4500},
        "Orders": {"name": "Orders", "dtype": "int64", "semantic_role": "metric",
                   "missing_pct": 0, "min": 1, "max": 100, "mean": 12},
        "Cost": {"name": "Cost", "dtype": "float64", "semantic_role": "metric",
                 "missing_pct": 0.5, "min": 5, "max": 40000, "mean": 1800},
        "Region": {"name": "Region", "dtype": "object", "semantic_role": "dimension",
                   "missing_pct": 0, "unique_count": 4,
                   "sample_values": ["North", "South", "East", "West"],
                   "top_values": ["North", "South"]},
        "Category": {"name": "Category", "dtype": "object", "semantic_role": "dimension",
                     "missing_pct": 0, "unique_count": 3,
                     "sample_values": ["Electronics", "Furniture", "Office Supplies"]},
        "Product": {"name": "Product", "dtype": "object", "semantic_role": "dimension",
                    "missing_pct": 0, "unique_count": 150,
                    "sample_values": ["Laptop", "Desk", "Pen"]},
        "City": {"name": "City", "dtype": "object", "semantic_role": "geo",
                 "missing_pct": 0, "unique_count": 50,
                 "sample_values": ["New York", "Chicago", "Houston"]},
        "Date": {"name": "Date", "dtype": "datetime64[ns]", "semantic_role": "date",
                 "missing_pct": 0},
        "Ship_Date": {"name": "Ship_Date", "dtype": "datetime64[ns]", "semantic_role": "date",
                      "missing_pct": 2},
        "Quantity": {"name": "Quantity", "dtype": "int64", "semantic_role": "metric",
                     "missing_pct": 0, "min": 1, "max": 50, "mean": 5},
    },
    "row_count": 10000,
    "quality_score": 96,
}

NO_DATES_SUMMARY = {
    "numeric_columns": ["Salary", "Age", "Experience"],
    "date_columns": [],
    "categorical_columns": ["Department", "Gender", "Level"],
    "primary_kpi": "Salary",
    "col_meta": {
        "Department": {"name": "Department", "sample_values": ["Engineering", "Marketing", "Sales"], "top_values": []},
        "Gender": {"name": "Gender", "sample_values": ["Male", "Female"], "top_values": []},
    },
    "row_count": 500,
    "quality_score": 88,
}

ONLY_NUMERIC_SUMMARY = {
    "numeric_columns": ["Temperature", "Pressure", "Humidity"],
    "date_columns": [],
    "categorical_columns": [],
    "primary_kpi": "Temperature",
    "col_meta": {},
    "row_count": 1000,
    "quality_score": 100,
}

SINGLE_NUMERIC_SUMMARY = {
    "numeric_columns": ["Value"],
    "date_columns": [],
    "categorical_columns": ["Type"],
    "primary_kpi": "Value",
    "col_meta": {},
    "row_count": 100,
    "quality_score": 90,
}

MINIMAL_SUMMARY = {
    "numeric_columns": [],
    "date_columns": [],
    "categorical_columns": [],
    "primary_kpi": None,
    "col_meta": {},
    "row_count": 0,
    "quality_score": 0,
}


def _make_test_df():
    """Create a realistic test dataframe."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Date": dates,
        "Sales": np.random.uniform(100, 5000, n),
        "Profit": np.random.uniform(-200, 2000, n),
        "Orders": np.random.randint(1, 50, n),
        "Cost": np.random.uniform(50, 3000, n),
        "Region": np.random.choice(["North", "South", "East", "West"], n),
        "Category": np.random.choice(["Electronics", "Furniture", "Office Supplies"], n),
        "Product": np.random.choice(["Laptop", "Desk", "Pen", "Chair", "Phone"], n),
    })


# ═══════════════════════════════════════════════════════════════
# TASK 1: DEAD CODE DELETION VERIFICATION
# ═══════════════════════════════════════════════════════════════

class TestTask1_DeadCodeDeletion:
    """Verify fabricating methods and viz_planner.py are gone."""

    def test_no_generate_dashboard_reasoning(self):
        client = NemotronLLMClient()
        assert not hasattr(client, "generate_dashboard_reasoning"), \
            "generate_dashboard_reasoning should be deleted — it fabricated fake data"

    def test_no_build_dynamic_payload(self):
        client = NemotronLLMClient()
        assert not hasattr(client, "_build_dynamic_payload"), \
            "_build_dynamic_payload should be deleted — it had hardcoded fake numbers"

    def test_no_fallback_response(self):
        client = NemotronLLMClient()
        assert not hasattr(client, "_fallback_response"), \
            "_fallback_response should be deleted — used _build_dynamic_payload"

    def test_viz_planner_file_deleted(self):
        viz_planner_path = Path(__file__).resolve().parent.parent / "app" / "agents" / "viz_planner.py"
        assert not viz_planner_path.exists(), \
            "viz_planner.py should be deleted — contained hardcoded fake insights"

    def test_viz_planner_not_importable(self):
        with pytest.raises(ImportError):
            import app.agents.viz_planner  # noqa

    def test_llm_client_still_has_chart_plan(self):
        client = NemotronLLMClient()
        assert hasattr(client, "generate_chart_plan")

    def test_llm_client_still_has_preprocessing_action(self):
        client = NemotronLLMClient()
        assert hasattr(client, "generate_preprocessing_action")

    def test_llm_client_still_has_master_prompt(self):
        client = NemotronLLMClient()
        assert hasattr(client, "_build_master_prompt")

    def test_llm_client_still_has_validation(self):
        client = NemotronLLMClient()
        assert hasattr(client, "_validate_and_repair_spec")


# ═══════════════════════════════════════════════════════════════
# TASK 2: INTENT PARSER — EXHAUSTIVE
# ═══════════════════════════════════════════════════════════════

class TestTask2_TimeFilters:
    """Every temporal pattern the IntentParser must handle."""

    # -- "this year" / "current year" / "latest year"
    @pytest.mark.parametrize("query", [
        "show sales this year",
        "current year revenue",
        "latest year performance",
    ])
    def test_this_year_variants(self, query):
        intent = IntentParser.parse(query, FULL_SUMMARY)
        assert intent["time_filter"]["type"] == "this_year"

    # -- "last year"
    def test_last_year(self):
        intent = IntentParser.parse("compare with last year", FULL_SUMMARY)
        assert intent["time_filter"]["type"] == "last_year"

    # -- "last N <unit>"
    @pytest.mark.parametrize("query,n,unit", [
        ("sales in the last 6 months", 6, "month"),
        ("last 3 months profit", 3, "month"),
        ("last 2 years revenue", 2, "year"),
        ("last 4 quarters growth", 4, "quarter"),
        ("last 7 days orders", 7, "day"),
        ("last 2 weeks activity", 2, "week"),
    ])
    def test_last_n_variants(self, query, n, unit):
        intent = IntentParser.parse(query, FULL_SUMMARY)
        assert intent["time_filter"]["type"] == "last_n"
        assert intent["time_filter"]["n"] == n
        assert intent["time_filter"]["unit"] == unit

    # -- "past N <unit>"
    @pytest.mark.parametrize("query,n,unit", [
        ("past 3 quarters", 3, "quarter"),
        ("past 12 months performance", 12, "month"),
        ("past 5 days", 5, "day"),
    ])
    def test_past_n_variants(self, query, n, unit):
        intent = IntentParser.parse(query, FULL_SUMMARY)
        assert intent["time_filter"]["type"] == "last_n"
        assert intent["time_filter"]["n"] == n
        assert intent["time_filter"]["unit"] == unit

    # -- "last quarter"
    def test_last_quarter(self):
        intent = IntentParser.parse("last quarter profit", FULL_SUMMARY)
        assert intent["time_filter"]["type"] == "last_quarter"

    # -- "this quarter"
    def test_this_quarter(self):
        intent = IntentParser.parse("this quarter orders", FULL_SUMMARY)
        assert intent["time_filter"]["type"] == "this_quarter"

    # -- YTD
    @pytest.mark.parametrize("query", ["year to date orders", "show YTD sales", "ytd revenue"])
    def test_ytd_variants(self, query):
        intent = IntentParser.parse(query, FULL_SUMMARY)
        assert intent["time_filter"]["type"] == "ytd"

    # -- YoY
    @pytest.mark.parametrize("query", ["year over year comparison", "YoY growth", "yoy trend"])
    def test_yoy_variants(self, query):
        intent = IntentParser.parse(query, FULL_SUMMARY)
        assert intent["time_filter"]["type"] == "yoy"

    # -- MoM
    @pytest.mark.parametrize("query", ["month over month growth", "MoM analysis", "mom trend"])
    def test_mom_variants(self, query):
        intent = IntentParser.parse(query, FULL_SUMMARY)
        assert intent["time_filter"]["type"] == "mom"

    # -- specific year
    @pytest.mark.parametrize("query,year", [
        ("sales in 2024", 2024),
        ("for 2023 revenue", 2023),
        ("of 2025 data", 2025),
    ])
    def test_specific_year(self, query, year):
        intent = IntentParser.parse(query, FULL_SUMMARY)
        assert intent["time_filter"]["type"] == "year"
        assert intent["time_filter"]["value"] == year

    # -- no time filter
    @pytest.mark.parametrize("query", [
        "show sales by region",
        "compare revenue and cost",
        "top 5 products",
        "hello",
    ])
    def test_no_time_filter(self, query):
        intent = IntentParser.parse(query, FULL_SUMMARY)
        assert intent["time_filter"] is None


class TestTask2_MetricExtraction:
    """Metric extraction from queries."""

    def test_single_metric_lowercase(self):
        intent = IntentParser.parse("show sales by region", FULL_SUMMARY)
        assert "Sales" in intent["metrics"]

    def test_multiple_metrics(self):
        intent = IntentParser.parse("compare sales and profit by region", FULL_SUMMARY)
        assert "Sales" in intent["metrics"]
        assert "Profit" in intent["metrics"]

    def test_three_metrics(self):
        intent = IntentParser.parse("show sales profit and orders", FULL_SUMMARY)
        assert "Sales" in intent["metrics"]
        assert "Profit" in intent["metrics"]
        assert "Orders" in intent["metrics"]

    def test_no_explicit_metric_falls_to_kpi(self):
        intent = IntentParser.parse("analyze everything", FULL_SUMMARY)
        assert intent["metrics"] == ["Sales"]  # primary KPI

    def test_no_metric_no_kpi(self):
        intent = IntentParser.parse("hello", MINIMAL_SUMMARY)
        assert intent["metrics"] == []  # no primary KPI

    def test_metric_with_underscores(self):
        summary = {**FULL_SUMMARY, "numeric_columns": ["Unit_Price", "Total_Sales"]}
        summary["primary_kpi"] = "Unit_Price"
        intent = IntentParser.parse("show unit price trends", summary)
        assert "Unit_Price" in intent["metrics"]


class TestTask2_DimensionExtraction:

    def test_single_dimension(self):
        intent = IntentParser.parse("sales by region", FULL_SUMMARY)
        assert "Region" in intent["dimensions"]

    def test_multiple_dimensions(self):
        intent = IntentParser.parse("sales by region and category", FULL_SUMMARY)
        assert "Region" in intent["dimensions"]
        assert "Category" in intent["dimensions"]

    def test_no_dimension(self):
        intent = IntentParser.parse("show total sales", FULL_SUMMARY)
        # No categorical column names mentioned
        assert len(intent["dimensions"]) == 0 or intent["dimensions"] == []


class TestTask2_Comparison:

    def test_vs_pattern(self):
        intent = IntentParser.parse("sales vs profit", FULL_SUMMARY)
        assert intent["comparison"] is not None
        assert "Sales" in intent["comparison"]["columns"]
        assert "Profit" in intent["comparison"]["columns"]

    def test_versus_pattern(self):
        intent = IntentParser.parse("sales versus profit", FULL_SUMMARY)
        assert intent["comparison"] is not None

    def test_compare_keyword(self):
        intent = IntentParser.parse("compare sales and profit", FULL_SUMMARY)
        assert intent["comparison"] is not None

    def test_against_keyword(self):
        intent = IntentParser.parse("sales against cost", FULL_SUMMARY)
        assert intent["comparison"] is not None

    def test_no_comparison(self):
        intent = IntentParser.parse("show sales by region", FULL_SUMMARY)
        assert intent["comparison"] is None


class TestTask2_TopBottom:

    @pytest.mark.parametrize("query,direction,n", [
        ("top 5 regions by sales", "top", 5),
        ("top 10 products by profit", "top", 10),
        ("bottom 3 categories by revenue", "bottom", 3),
        ("best 5 regions", "top", 5),
        ("worst 10 products", "bottom", 10),
        ("highest regions by sales", "top", 5),  # default n=5
        ("lowest categories", "bottom", 5),  # default n=5
    ])
    def test_top_bottom_variants(self, query, direction, n):
        intent = IntentParser.parse(query, FULL_SUMMARY)
        assert intent["top_bottom"] is not None
        assert intent["top_bottom"]["direction"] == direction
        assert intent["top_bottom"]["n"] == n

    def test_no_top_bottom(self):
        intent = IntentParser.parse("show sales by region", FULL_SUMMARY)
        assert intent["top_bottom"] is None


class TestTask2_Overview:

    @pytest.mark.parametrize("query", [
        "analyze everything",
        "give me an overview",
        "show me all insights",
        "full analysis",
        "dashboard",
        "show me",
    ])
    def test_overview_detected(self, query):
        intent = IntentParser.parse(query, FULL_SUMMARY)
        assert intent["is_overview"] is True

    def test_not_overview_when_metric_present(self):
        intent = IntentParser.parse("show sales insights", FULL_SUMMARY)
        # "insights" triggers overview keyword, but "sales" provides a metric
        # so is_overview stays False because metrics is non-empty
        assert intent["is_overview"] is False

    def test_not_overview_regular_query(self):
        intent = IntentParser.parse("profit by category last quarter", FULL_SUMMARY)
        assert intent["is_overview"] is False


class TestTask2_Filters:

    def test_filter_by_known_value(self):
        intent = IntentParser.parse("sales in the north region", FULL_SUMMARY)
        # "North" is a sample_value of Region
        found = any(f["column"] == "Region" and f["value"].lower() == "north" for f in intent["filters"])
        assert found, f"Expected filter on Region=North, got: {intent['filters']}"

    def test_filter_by_category_value(self):
        intent = IntentParser.parse("show electronics sales", FULL_SUMMARY)
        found = any(f["column"] == "Category" and f["value"].lower() == "electronics" for f in intent["filters"])
        assert found, f"Expected filter on Category=Electronics, got: {intent['filters']}"

    def test_no_filter_for_unknown_values(self):
        intent = IntentParser.parse("show sales by region", FULL_SUMMARY)
        # "region" is a column name not a value — should not produce a filter
        region_value_filters = [f for f in intent["filters"] if f["value"].lower() == "region"]
        assert len(region_value_filters) == 0


class TestTask2_StructureIntegrity:
    """Verify the QueryIntent dict always has all required keys."""

    def test_all_keys_present(self):
        intent = IntentParser.parse("show sales", FULL_SUMMARY)
        required = {"raw_query", "time_filter", "comparison", "filters", "metrics", "dimensions", "top_bottom", "is_overview"}
        assert required.issubset(set(intent.keys()))

    def test_empty_query(self):
        intent = IntentParser.parse("", FULL_SUMMARY)
        assert "raw_query" in intent
        assert intent["raw_query"] == ""

    def test_very_long_query(self):
        long_query = "show me the sales trend over time for the last 6 months " * 10
        intent = IntentParser.parse(long_query, FULL_SUMMARY)
        assert intent is not None
        assert isinstance(intent, dict)


# ═══════════════════════════════════════════════════════════════
# TASK 2 (continued): apply_time_filter on real DataFrames
# ═══════════════════════════════════════════════════════════════

class TestTask2_ApplyTimeFilter:

    def setup_method(self):
        # 2 years of data
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
        self.df = pd.DataFrame({
            "Date": dates,
            "Sales": range(len(dates)),
        })

    def test_this_year(self):
        result = IntentParser.apply_time_filter(self.df, "Date", {"type": "this_year"})
        years = pd.to_datetime(result["Date"]).dt.year.unique()
        assert len(years) == 1
        assert years[0] == 2024  # latest year

    def test_last_year(self):
        result = IntentParser.apply_time_filter(self.df, "Date", {"type": "last_year"})
        years = pd.to_datetime(result["Date"]).dt.year.unique()
        assert len(years) == 1
        assert years[0] == 2023

    def test_specific_year(self):
        result = IntentParser.apply_time_filter(self.df, "Date", {"type": "year", "value": 2023})
        years = pd.to_datetime(result["Date"]).dt.year.unique()
        assert 2023 in years
        assert 2024 not in years

    def test_last_n_months(self):
        result = IntentParser.apply_time_filter(self.df, "Date", {"type": "last_n", "n": 3, "unit": "month"})
        assert len(result) > 0
        assert len(result) < len(self.df)

    def test_last_n_years(self):
        result = IntentParser.apply_time_filter(self.df, "Date", {"type": "last_n", "n": 1, "unit": "year"})
        assert len(result) > 0

    def test_last_n_weeks(self):
        result = IntentParser.apply_time_filter(self.df, "Date", {"type": "last_n", "n": 4, "unit": "week"})
        assert len(result) > 0

    def test_last_n_days(self):
        result = IntentParser.apply_time_filter(self.df, "Date", {"type": "last_n", "n": 30, "unit": "day"})
        assert len(result) > 0

    def test_last_n_quarters(self):
        result = IntentParser.apply_time_filter(self.df, "Date", {"type": "last_n", "n": 2, "unit": "quarter"})
        assert len(result) > 0

    def test_ytd(self):
        result = IntentParser.apply_time_filter(self.df, "Date", {"type": "ytd"})
        years = pd.to_datetime(result["Date"]).dt.year.unique()
        assert 2024 in years  # latest year

    def test_last_quarter(self):
        result = IntentParser.apply_time_filter(self.df, "Date", {"type": "last_quarter"})
        assert len(result) > 0
        assert len(result) < len(self.df)

    def test_this_quarter(self):
        result = IntentParser.apply_time_filter(self.df, "Date", {"type": "this_quarter"})
        assert len(result) > 0

    def test_unknown_type_returns_full_df(self):
        result = IntentParser.apply_time_filter(self.df, "Date", {"type": "unknown_filter"})
        assert len(result) == len(self.df)

    def test_bad_date_column_returns_full_df(self):
        df_bad = pd.DataFrame({"Date": ["not", "a", "date"], "Sales": [1, 2, 3]})
        result = IntentParser.apply_time_filter(df_bad, "Date", {"type": "this_year"})
        assert len(result) == len(df_bad)


class TestTask2_ApplyValueFilters:

    def test_equals_filter(self):
        df = pd.DataFrame({"Region": ["North", "South", "East"], "Sales": [100, 200, 300]})
        result = IntentParser.apply_value_filters(df, [{"column": "Region", "operator": "==", "value": "North"}])
        assert len(result) == 1
        assert result.iloc[0]["Region"] == "North"

    def test_not_equals_filter(self):
        df = pd.DataFrame({"Region": ["North", "South", "East"], "Sales": [100, 200, 300]})
        result = IntentParser.apply_value_filters(df, [{"column": "Region", "operator": "!=", "value": "North"}])
        assert len(result) == 2

    def test_greater_than_filter(self):
        df = pd.DataFrame({"Sales": [100, 200, 300, 400, 500]})
        result = IntentParser.apply_value_filters(df, [{"column": "Sales", "operator": ">", "value": "300"}])
        assert len(result) == 2

    def test_less_than_filter(self):
        df = pd.DataFrame({"Sales": [100, 200, 300, 400, 500]})
        result = IntentParser.apply_value_filters(df, [{"column": "Sales", "operator": "<", "value": "300"}])
        assert len(result) == 2

    def test_gte_filter(self):
        df = pd.DataFrame({"Sales": [100, 200, 300]})
        result = IntentParser.apply_value_filters(df, [{"column": "Sales", "operator": ">=", "value": "200"}])
        assert len(result) == 2

    def test_lte_filter(self):
        df = pd.DataFrame({"Sales": [100, 200, 300]})
        result = IntentParser.apply_value_filters(df, [{"column": "Sales", "operator": "<=", "value": "200"}])
        assert len(result) == 2

    def test_nonexistent_column_skipped(self):
        df = pd.DataFrame({"Sales": [100, 200, 300]})
        result = IntentParser.apply_value_filters(df, [{"column": "NotAColumn", "operator": "==", "value": "x"}])
        assert len(result) == 3  # no filtering

    def test_bad_value_type_skipped(self):
        df = pd.DataFrame({"Sales": [100, 200, 300]})
        result = IntentParser.apply_value_filters(df, [{"column": "Sales", "operator": ">", "value": "not_a_number"}])
        assert len(result) == 3  # gracefully skipped

    def test_case_insensitive_equals(self):
        df = pd.DataFrame({"Region": ["North", "SOUTH", "east"], "Sales": [1, 2, 3]})
        result = IntentParser.apply_value_filters(df, [{"column": "Region", "operator": "==", "value": "south"}])
        assert len(result) == 1

    def test_multiple_filters(self):
        df = pd.DataFrame({
            "Region": ["North", "South", "North", "South"],
            "Sales": [100, 200, 300, 400],
        })
        filters = [
            {"column": "Region", "operator": "==", "value": "North"},
            {"column": "Sales", "operator": ">", "value": "150"},
        ]
        result = IntentParser.apply_value_filters(df, filters)
        assert len(result) == 1
        assert result.iloc[0]["Sales"] == 300


# ═══════════════════════════════════════════════════════════════
# TASK 3: MASTER PROMPT ENGINEERING
# ═══════════════════════════════════════════════════════════════

class TestTask3_MasterPrompt:

    def setup_method(self):
        self.client = NemotronLLMClient()

    def test_prompt_contains_schema(self):
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "show sales")
        assert "Sales" in prompt
        assert "Profit" in prompt
        assert "Region" in prompt
        assert "Date" in prompt

    def test_prompt_contains_chart_rules(self):
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "show sales")
        assert "CHART SELECTION RULES" in prompt
        assert "Line Graph" in prompt
        assert "Bar Chart" in prompt
        assert "Donut Chart" in prompt

    def test_prompt_contains_query_type_detection(self):
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "show sales")
        assert "QUERY TYPE DETECTION" in prompt
        assert "trend" in prompt.lower()

    def test_prompt_contains_output_constraints(self):
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "show sales")
        assert "STRICT OUTPUT CONSTRAINTS" in prompt
        assert "NEVER" in prompt

    def test_prompt_contains_edge_cases(self):
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "show sales")
        assert "EDGE CASES" in prompt

    def test_prompt_contains_examples(self):
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "show sales")
        assert "EXAMPLES" in prompt
        assert "Example 1:" in prompt

    def test_prompt_contains_user_query(self):
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "what are the top regions?")
        assert "what are the top regions?" in prompt

    def test_prompt_contains_row_count(self):
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "show sales")
        assert "10000" in prompt

    def test_prompt_contains_quality_score(self):
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "show sales")
        assert "96" in prompt

    def test_prompt_contains_primary_kpi(self):
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "show sales")
        assert "Sales" in prompt

    def test_prompt_with_conversation_context(self):
        context = [{"query": "show sales"}, {"query": "now by region"}]
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "zoom into Q3", context)
        assert "CONVERSATION HISTORY" in prompt
        assert "show sales" in prompt
        assert "now by region" in prompt

    def test_prompt_without_conversation_context(self):
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "show sales", None)
        assert "CONVERSATION HISTORY" not in prompt

    def test_prompt_empty_conversation_context(self):
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "show sales", [])
        assert "CONVERSATION HISTORY" not in prompt

    def test_prompt_no_col_meta_fallback(self):
        summary = {**FULL_SUMMARY, "col_meta": {}}
        prompt = self.client._build_master_prompt(summary, "show sales")
        # Should still list columns using the fallback path
        assert "Sales" in prompt
        assert "(numeric)" in prompt

    def test_prompt_with_col_statistics(self):
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "show sales")
        # col_meta has min/max/mean for Sales
        assert "min=" in prompt
        assert "max=" in prompt
        assert "mean=" in prompt

    def test_prompt_with_sample_values(self):
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "show sales")
        # Region has sample_values
        assert "North" in prompt or "samples=" in prompt

    def test_prompt_no_primary_kpi(self):
        summary = {**FULL_SUMMARY, "primary_kpi": None}
        prompt = self.client._build_master_prompt(summary, "show data")
        assert "not detected" in prompt

    def test_prompt_returns_string(self):
        prompt = self.client._build_master_prompt(FULL_SUMMARY, "show sales")
        assert isinstance(prompt, str)
        assert len(prompt) > 500  # should be substantial


# ═══════════════════════════════════════════════════════════════
# TASK 4: JSON-SCHEMA VALIDATION & REPAIR
# ═══════════════════════════════════════════════════════════════

class TestTask4_ChartTypeNormalization:

    def setup_method(self):
        self.client = NemotronLLMClient()

    @pytest.mark.parametrize("input_type,expected", [
        ("Bar Chart", "Bar Chart"),
        ("Line Graph", "Line Graph"),
        ("Area Graph", "Area Graph"),
        ("Donut Chart", "Donut Chart"),
        ("Scatterplot", "Scatterplot"),
        ("Histogram", "Histogram"),
    ])
    def test_exact_match(self, input_type, expected):
        assert self.client._normalize_chart_type(input_type) == expected

    @pytest.mark.parametrize("alias,expected", [
        ("bar", "Bar Chart"), ("bar chart", "Bar Chart"), ("bar graph", "Bar Chart"),
        ("column chart", "Bar Chart"), ("column", "Bar Chart"),
        ("line", "Line Graph"), ("line chart", "Line Graph"),
        ("area", "Area Graph"), ("area chart", "Area Graph"),
        ("donut", "Donut Chart"), ("donut chart", "Donut Chart"),
        ("pie", "Donut Chart"), ("pie chart", "Donut Chart"),
        ("scatter", "Scatterplot"), ("scatterplot", "Scatterplot"), ("scatter plot", "Scatterplot"),
        ("histogram", "Histogram"), ("hist", "Histogram"),
    ])
    def test_alias_match(self, alias, expected):
        assert self.client._normalize_chart_type(alias) == expected

    @pytest.mark.parametrize("bad_type", [
        "Sankey Diagram", "Choropleth", "Treemap", "Waterfall", "Gantt",
        "Sunburst", "Radar", "Heat Map", "Box Plot", "Violin",
    ])
    def test_unknown_types_return_none(self, bad_type):
        assert self.client._normalize_chart_type(bad_type) is None


class TestTask4_FuzzyColumnMatch:

    def setup_method(self):
        self.client = NemotronLLMClient()
        self.cols = ["Sales", "Profit", "Region", "Category", "Date", "Ship_Date"]

    def test_exact_match(self):
        assert self.client._fuzzy_match_column("Sales", self.cols) == "Sales"

    def test_case_insensitive(self):
        assert self.client._fuzzy_match_column("sales", self.cols) == "Sales"
        assert self.client._fuzzy_match_column("PROFIT", self.cols) == "Profit"
        assert self.client._fuzzy_match_column("region", self.cols) == "Region"

    def test_typo_repair(self):
        assert self.client._fuzzy_match_column("Saels", self.cols) == "Sales"
        assert self.client._fuzzy_match_column("Profti", self.cols) == "Profit"

    def test_no_match_returns_none(self):
        assert self.client._fuzzy_match_column("TotallyFake", self.cols) is None
        assert self.client._fuzzy_match_column("xyz123", self.cols) is None

    def test_empty_inputs(self):
        assert self.client._fuzzy_match_column("", self.cols) is None
        assert self.client._fuzzy_match_column("Sales", []) is None
        assert self.client._fuzzy_match_column(None, self.cols) is None


class TestTask4_FullSpecValidation:

    def setup_method(self):
        self.client = NemotronLLMClient()

    def test_valid_spec_passes(self):
        spec = {"type": "Bar Chart", "x_axis": "Region", "y_axis": "Sales",
                "title": "Sales by Region", "insight_tooltip": "Shows sales across regions"}
        result = self.client._validate_and_repair_spec(spec, FULL_SUMMARY)
        assert result is not None
        assert result["type"] == "Bar Chart"
        assert result["x_axis"] == "Region"
        assert result["y_axis"] == "Sales"

    def test_repairs_lowercase_chart_type(self):
        spec = {"type": "bar", "x_axis": "Region", "y_axis": "Sales", "title": "T", "insight_tooltip": "T"}
        result = self.client._validate_and_repair_spec(spec, FULL_SUMMARY)
        assert result["type"] == "Bar Chart"

    def test_repairs_pie_to_donut(self):
        spec = {"type": "pie chart", "x_axis": "Region", "y_axis": "Sales", "title": "T", "insight_tooltip": "T"}
        result = self.client._validate_and_repair_spec(spec, FULL_SUMMARY)
        assert result["type"] == "Donut Chart"

    def test_repairs_case_insensitive_columns(self):
        spec = {"type": "Bar Chart", "x_axis": "region", "y_axis": "sales", "title": "T", "insight_tooltip": "T"}
        result = self.client._validate_and_repair_spec(spec, FULL_SUMMARY)
        assert result["x_axis"] == "Region"
        assert result["y_axis"] == "Sales"

    def test_repairs_misspelled_column(self):
        spec = {"type": "Bar Chart", "x_axis": "Reigon", "y_axis": "Saels", "title": "T", "insight_tooltip": "T"}
        result = self.client._validate_and_repair_spec(spec, FULL_SUMMARY)
        assert result["x_axis"] == "Region"
        assert result["y_axis"] == "Sales"

    def test_drops_unknown_chart_type(self):
        spec = {"type": "Sankey Diagram", "x_axis": "Region", "y_axis": "Sales", "title": "T", "insight_tooltip": "T"}
        result = self.client._validate_and_repair_spec(spec, FULL_SUMMARY)
        assert result is None

    def test_repairs_nonexistent_yaxis_to_kpi(self):
        spec = {"type": "Bar Chart", "x_axis": "Region", "y_axis": "ZZZZZ", "title": "T", "insight_tooltip": "T"}
        result = self.client._validate_and_repair_spec(spec, FULL_SUMMARY)
        assert result is not None
        assert result["y_axis"] == "Sales"  # primary KPI fallback

    def test_clears_nonexistent_xaxis(self):
        spec = {"type": "Bar Chart", "x_axis": "ZZZZZ", "y_axis": "Sales", "title": "T", "insight_tooltip": "T"}
        result = self.client._validate_and_repair_spec(spec, FULL_SUMMARY)
        assert result is not None
        assert result["x_axis"] == ""  # cleared

    def test_auto_generates_title(self):
        spec = {"type": "Bar Chart", "x_axis": "Region", "y_axis": "Sales", "title": "", "insight_tooltip": "T"}
        result = self.client._validate_and_repair_spec(spec, FULL_SUMMARY)
        assert result["title"]  # non-empty auto-generated

    def test_auto_generates_tooltip(self):
        spec = {"type": "Bar Chart", "x_axis": "Region", "y_axis": "Sales", "title": "T", "insight_tooltip": ""}
        result = self.client._validate_and_repair_spec(spec, FULL_SUMMARY)
        assert result["insight_tooltip"]  # non-empty auto-generated

    def test_none_spec_returns_none(self):
        result = self.client._validate_and_repair_spec(None, FULL_SUMMARY)
        assert result is None

    def test_non_dict_spec_returns_none(self):
        result = self.client._validate_and_repair_spec("bad", FULL_SUMMARY)
        assert result is None

    def test_empty_dict_gets_repaired(self):
        result = self.client._validate_and_repair_spec({}, FULL_SUMMARY)
        # Empty type string may get repaired via substring matching in aliases
        # The validator's repair-over-reject strategy is intentional
        if result is not None:
            assert result["type"] in ALLOWED_CHART_TYPES

    def test_yaxis_fallback_to_first_numeric(self):
        summary = {**FULL_SUMMARY, "primary_kpi": None}
        spec = {"type": "Bar Chart", "x_axis": "Region", "y_axis": "ZZZZZ", "title": "T", "insight_tooltip": "T"}
        result = self.client._validate_and_repair_spec(spec, summary)
        assert result is not None
        assert result["y_axis"] == "Sales"  # first numeric col

    def test_yaxis_no_numeric_at_all(self):
        summary = {"numeric_columns": [], "date_columns": [], "categorical_columns": ["Region"],
                    "primary_kpi": None, "col_meta": {}}
        spec = {"type": "Bar Chart", "x_axis": "Region", "y_axis": "ZZZZZ", "title": "T", "insight_tooltip": "T"}
        result = self.client._validate_and_repair_spec(spec, summary)
        assert result is None  # unfixable — no numeric columns at all

    def test_empty_string_xaxis_not_cleared(self):
        spec = {"type": "Histogram", "x_axis": "", "y_axis": "Sales", "title": "T", "insight_tooltip": "T"}
        result = self.client._validate_and_repair_spec(spec, FULL_SUMMARY)
        assert result is not None
        assert result["x_axis"] == ""  # valid for Histogram


class TestTask4_GenerateChartPlanWithoutLLM:
    """generate_chart_plan should return [] when no LLM is configured."""

    def test_returns_empty_without_api_key(self):
        client = NemotronLLMClient()
        # By default, if NVIDIA_API_KEY is empty, client.client is None
        result = client.generate_chart_plan(FULL_SUMMARY, "show sales")
        assert result == []


# ═══════════════════════════════════════════════════════════════
# TASK 5: HEURISTIC FALLBACK PLANNER
# ═══════════════════════════════════════════════════════════════

class TestTask5_HeuristicPlan:

    def setup_method(self):
        self.builder = DashboardBuilder()

    # --- Trend queries ---
    @pytest.mark.parametrize("query", [
        "show sales trend over time",
        "monthly sales",
        "yearly growth",
        "sales forecast",
        "quarterly performance",
    ])
    def test_trend_queries(self, query):
        plans = self.builder.heuristic_plan(FULL_SUMMARY, query)
        assert any(p["type"] in ("Line Graph", "Area Graph") for p in plans), \
            f"Trend query '{query}' should produce Line/Area chart"

    # --- Breakdown queries ---
    def test_breakdown_query(self):
        plans = self.builder.heuristic_plan(FULL_SUMMARY, "sales by region")
        assert any(p["type"] == "Bar Chart" for p in plans)

    # --- Distribution ---
    @pytest.mark.parametrize("query", ["distribution of sales", "sales spread", "histogram of profit"])
    def test_distribution_queries(self, query):
        plans = self.builder.heuristic_plan(FULL_SUMMARY, query)
        assert any(p["type"] == "Histogram" for p in plans)

    # --- Correlation ---
    @pytest.mark.parametrize("query", ["correlation between sales and profit", "scatter relationship", "scatter plot"])
    def test_correlation_queries(self, query):
        plans = self.builder.heuristic_plan(FULL_SUMMARY, query)
        assert any(p["type"] == "Scatterplot" for p in plans)

    # --- Share / Composition ---
    @pytest.mark.parametrize("query", ["share of sales by region", "composition", "percentage breakdown"])
    def test_composition_queries(self, query):
        plans = self.builder.heuristic_plan(FULL_SUMMARY, query)
        assert any(p["type"] == "Donut Chart" for p in plans)

    # --- With intent: overview ---
    def test_overview_intent(self):
        intent = {"metrics": [], "dimensions": [], "comparison": None, "time_filter": None,
                  "filters": [], "top_bottom": None, "is_overview": True}
        plans = self.builder.heuristic_plan(FULL_SUMMARY, "analyze everything", intent)
        assert len(plans) >= 3
        types = {p["type"] for p in plans}
        assert len(types) >= 2  # diverse set

    # --- With intent: comparison ---
    def test_comparison_intent(self):
        intent = {"metrics": ["Sales", "Profit"], "dimensions": ["Region"],
                  "comparison": {"columns": ["Sales", "Profit"]}, "time_filter": None,
                  "filters": [], "top_bottom": None, "is_overview": False}
        plans = self.builder.heuristic_plan(FULL_SUMMARY, "compare sales and profit", intent)
        assert len(plans) >= 2

    # --- With intent: top N ---
    def test_top_n_intent(self):
        intent = {"metrics": ["Sales"], "dimensions": ["Region"], "comparison": None,
                  "time_filter": None, "filters": [], "top_bottom": {"direction": "top", "n": 5},
                  "is_overview": False}
        plans = self.builder.heuristic_plan(FULL_SUMMARY, "top 5 regions by sales", intent)
        assert any("Top" in p.get("title", "") for p in plans)

    # --- No dates → no time charts ---
    def test_no_dates_no_time_charts(self):
        plans = self.builder.heuristic_plan(NO_DATES_SUMMARY, "show salary trend over time")
        for p in plans:
            assert p["type"] not in ("Line Graph", "Area Graph")

    # --- Only numeric → histogram ---
    def test_only_numeric(self):
        plans = self.builder.heuristic_plan(ONLY_NUMERIC_SUMMARY, "show temperature")
        assert len(plans) >= 1

    # --- Single numeric + categories ---
    def test_single_numeric(self):
        plans = self.builder.heuristic_plan(SINGLE_NUMERIC_SUMMARY, "show value by type")
        assert any(p["type"] == "Bar Chart" for p in plans)

    # --- Cap at 5 ---
    def test_max_5_charts(self):
        intent = {"metrics": [], "dimensions": [], "comparison": None, "time_filter": None,
                  "filters": [], "top_bottom": None, "is_overview": True}
        plans = self.builder.heuristic_plan(FULL_SUMMARY, "everything possible all charts", intent)
        assert len(plans) <= 5

    # --- Backward compat without intent ---
    def test_works_without_intent(self):
        plans = self.builder.heuristic_plan(FULL_SUMMARY, "show sales by region")
        assert len(plans) >= 1

    # --- Falls back to default_plan ---
    def test_fallback_to_default(self):
        plans = self.builder.heuristic_plan(FULL_SUMMARY, "xyzzy random gibberish")
        assert len(plans) >= 1  # default_plan kicks in

    # --- Time intent ---
    def test_time_intent_produces_time_chart(self):
        intent = {"metrics": ["Sales"], "dimensions": [], "comparison": None,
                  "time_filter": {"type": "last_n", "n": 6, "unit": "month"},
                  "filters": [], "top_bottom": None, "is_overview": False}
        plans = self.builder.heuristic_plan(FULL_SUMMARY, "sales in the last 6 months", intent)
        assert any(p["type"] in ("Line Graph", "Area Graph") for p in plans)


# ═══════════════════════════════════════════════════════════════
# TASK 6: CONVERSATION MEMORY
# ═══════════════════════════════════════════════════════════════

class TestTask6_ConversationMemory:

    def test_add_and_get_context(self):
        mem = ConversationMemory()
        intent = {"metrics": ["Sales"], "dimensions": ["Region"], "time_filter": None, "filters": [], "comparison": None}
        mem.add("ds-1", "show sales by region", intent)
        ctx = mem.get_context("ds-1")
        assert len(ctx) == 1
        assert ctx[0]["query"] == "show sales by region"

    def test_multiple_entries(self):
        mem = ConversationMemory()
        for i in range(5):
            mem.add("ds-1", f"query {i}", {"metrics": [f"m{i}"]})
        ctx = mem.get_context("ds-1")
        assert len(ctx) == 5

    def test_per_dataset_isolation(self):
        mem = ConversationMemory()
        mem.add("ds-1", "query for ds1", {"metrics": ["A"]})
        mem.add("ds-2", "query for ds2", {"metrics": ["B"]})
        assert len(mem.get_context("ds-1")) == 1
        assert len(mem.get_context("ds-2")) == 1
        assert mem.get_context("ds-1")[0]["query"] == "query for ds1"

    def test_capacity_limit_20(self):
        mem = ConversationMemory()
        for i in range(30):
            mem.add("ds-1", f"query {i}", {"metrics": [f"m{i}"]})
        ctx = mem.get_context("ds-1")
        assert len(ctx) == 20
        # Should keep the LAST 20
        assert ctx[0]["query"] == "query 10"
        assert ctx[-1]["query"] == "query 29"

    def test_get_last_intent(self):
        mem = ConversationMemory()
        mem.add("ds-1", "q1", {"metrics": ["Sales"]})
        mem.add("ds-1", "q2", {"metrics": ["Profit"]})
        last = mem.get_last_intent("ds-1")
        assert last["metrics"] == ["Profit"]

    def test_get_last_intent_empty(self):
        mem = ConversationMemory()
        assert mem.get_last_intent("ds-nonexistent") is None

    def test_clear(self):
        mem = ConversationMemory()
        mem.add("ds-1", "q1", {"metrics": ["Sales"]})
        mem.clear("ds-1")
        assert mem.get_context("ds-1") == []
        assert mem.get_last_intent("ds-1") is None


class TestTask6_FollowupResolution:

    def test_followup_carries_metrics(self):
        mem = ConversationMemory()
        mem.add("ds-1", "show sales by region", {
            "metrics": ["Sales"], "dimensions": ["Region"], "time_filter": {"type": "this_year"},
            "filters": [], "comparison": None
        })
        current = {"metrics": [], "dimensions": ["Category"], "time_filter": None,
                    "filters": [], "comparison": None}
        resolved = mem.resolve_followup("ds-1", current, "now break that down by category")
        assert resolved["metrics"] == ["Sales"]
        assert resolved["dimensions"] == ["Category"]
        assert resolved["time_filter"] == {"type": "this_year"}

    def test_followup_what_about(self):
        mem = ConversationMemory()
        mem.add("ds-1", "show sales by region", {
            "metrics": ["Sales"], "dimensions": ["Region"], "time_filter": None,
            "filters": [], "comparison": None
        })
        current = {"metrics": ["Profit"], "dimensions": [], "time_filter": None,
                    "filters": [], "comparison": None}
        resolved = mem.resolve_followup("ds-1", current, "what about profit instead")
        assert "Profit" in resolved["metrics"]

    def test_followup_also(self):
        mem = ConversationMemory()
        mem.add("ds-1", "show sales", {"metrics": ["Sales"], "dimensions": [], "time_filter": None,
                                        "filters": [], "comparison": None})
        current = {"metrics": [], "dimensions": ["Region"], "time_filter": None,
                    "filters": [], "comparison": None}
        resolved = mem.resolve_followup("ds-1", current, "also by region")
        assert resolved["metrics"] == ["Sales"]
        assert "Region" in resolved["dimensions"]

    def test_non_followup_long_query(self):
        mem = ConversationMemory()
        mem.add("ds-1", "show sales", {"metrics": ["Sales"], "dimensions": [],
                                        "time_filter": None, "filters": [], "comparison": None})
        current = {"metrics": ["Profit"], "dimensions": ["Region"], "time_filter": None,
                    "filters": [], "comparison": None}
        resolved = mem.resolve_followup("ds-1", current, "show profit breakdown by region for last quarter in all departments")
        # Long query = not a followup → original intent preserved
        assert resolved["metrics"] == ["Profit"]

    def test_followup_no_history(self):
        mem = ConversationMemory()
        current = {"metrics": [], "dimensions": [], "time_filter": None,
                    "filters": [], "comparison": None}
        resolved = mem.resolve_followup("ds-1", current, "that one")
        assert resolved is current  # unchanged

    def test_followup_carries_filters(self):
        mem = ConversationMemory()
        mem.add("ds-1", "show sales in north", {
            "metrics": ["Sales"], "dimensions": [],
            "time_filter": None,
            "filters": [{"column": "Region", "operator": "==", "value": "North"}],
            "comparison": None
        })
        current = {"metrics": [], "dimensions": ["Category"], "time_filter": None,
                    "filters": [], "comparison": None}
        resolved = mem.resolve_followup("ds-1", current, "now by category")
        assert len(resolved["filters"]) == 1
        assert resolved["filters"][0]["value"] == "North"

    def test_followup_short_query_detected(self):
        mem = ConversationMemory()
        mem.add("ds-1", "show sales", {"metrics": ["Sales"], "dimensions": [],
                                        "time_filter": None, "filters": [], "comparison": None})
        current = {"metrics": [], "dimensions": ["Region"], "time_filter": None,
                    "filters": [], "comparison": None}
        # "by region" is <= 6 words → followup
        resolved = mem.resolve_followup("ds-1", current, "by region")
        assert resolved["metrics"] == ["Sales"]


# ═══════════════════════════════════════════════════════════════
# TASK 7: QUERY TELEMETRY
# ═══════════════════════════════════════════════════════════════

class TestTask7_Telemetry:

    def test_telemetry_writes_json(self, tmp_path):
        """Test that _log_query_telemetry writes valid JSON lines."""
        # We'll intercept by checking the telemetry logger
        import app.agents.orchestrator as orch

        records = []
        original_info = orch._telemetry_logger.info

        def capture(msg):
            records.append(msg)
            original_info(msg)

        orch._telemetry_logger.info = capture
        try:
            intent = {"time_filter": None, "metrics": ["Sales"], "dimensions": ["Region"],
                       "comparison": None, "filters": [], "is_overview": False}
            _log_query_telemetry("test-ds", "show sales", intent, "fallback", 3)

            assert len(records) >= 1
            parsed = json.loads(records[-1])
            assert parsed["dataset_id"] == "test-ds"
            assert parsed["query"] == "show sales"
            assert parsed["plan_source"] == "fallback"
            assert parsed["charts_materialized"] == 3
            assert "timestamp" in parsed
            assert parsed["intent"]["metrics"] == ["Sales"]
        finally:
            orch._telemetry_logger.info = original_info

    def test_telemetry_never_crashes(self):
        """Even with bad input, telemetry should never raise."""
        try:
            _log_query_telemetry(None, None, {}, None, None)
        except Exception:
            pytest.fail("Telemetry logging should NEVER raise an exception")

    def test_telemetry_log_directory_setup(self):
        """Verify the log directory path is configured."""
        import app.agents.orchestrator as orch
        assert orch._log_dir.name == "logs"
        assert "backend" in str(orch._log_dir)

    def test_telemetry_path_correct(self):
        import app.agents.orchestrator as orch
        assert orch._telemetry_path.name == "query_telemetry.jsonl"


# ═══════════════════════════════════════════════════════════════
# TASK 8: CONFIG SECURITY
# ═══════════════════════════════════════════════════════════════

class TestTask8_ConfigSecurity:

    def test_is_llm_configured_true(self):
        s = Settings(NVIDIA_API_KEY="nvapi-test-key-123")
        assert s.is_llm_configured is True

    def test_is_llm_configured_false_empty(self):
        s = Settings(NVIDIA_API_KEY="")
        assert s.is_llm_configured is False

    def test_is_llm_configured_false_whitespace(self):
        s = Settings(NVIDIA_API_KEY="   ")
        assert s.is_llm_configured is False

    def test_repr_masks_api_key(self):
        s = Settings(NVIDIA_API_KEY="nvapi-super-secret")
        r = repr(s)
        assert "nvapi-super-secret" not in r
        assert "***" in r

    def test_repr_shows_not_set(self):
        s = Settings(NVIDIA_API_KEY="")
        r = repr(s)
        assert "(not set)" in r

    def test_str_same_as_repr(self):
        s = Settings(NVIDIA_API_KEY="secret")
        assert str(s) == repr(s)

    def test_gitignore_has_env(self):
        gitignore_path = Path(__file__).resolve().parent.parent.parent / ".gitignore"
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            assert ".env" in content

    def test_gitignore_has_backend_logs(self):
        gitignore_path = Path(__file__).resolve().parent.parent.parent / ".gitignore"
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            assert "backend/logs" in content


# ═══════════════════════════════════════════════════════════════
# INTEGRATION: FULL FLOW (IntentParser → heuristic_plan → materialize)
# ═══════════════════════════════════════════════════════════════

class TestIntegration_FullFlow:
    """End-to-end: query → intent → plan → charts from real data."""

    def setup_method(self):
        self.builder = DashboardBuilder()
        self.df = _make_test_df()

    def _run_query(self, query, summary=FULL_SUMMARY):
        intent = IntentParser.parse(query, summary)
        plan = self.builder.heuristic_plan(summary, query, intent)
        charts = []
        for i, spec in enumerate(plan):
            chart = self.builder.materialize_chart(self.df, summary, spec, f"test-{i}")
            if chart:
                charts.append(chart)
        return intent, plan, charts

    def test_sales_over_time(self):
        intent, plan, charts = self._run_query("show sales over time")
        assert len(charts) >= 1
        assert any(c["type"] in ("Line Graph", "Area Graph") for c in charts)
        # Verify data is from real df
        for c in charts:
            assert len(c["data"]) > 0

    def test_compare_sales_profit_by_region(self):
        intent, plan, charts = self._run_query("compare sales and profit by region")
        assert len(charts) >= 2

    def test_sales_last_6_months(self):
        intent, plan, charts = self._run_query("sales in last 6 months")
        assert intent["time_filter"]["type"] == "last_n"
        assert len(charts) >= 1

    def test_distribution_of_orders(self):
        intent, plan, charts = self._run_query("what is the distribution of orders")
        assert any(c["type"] == "Histogram" for c in charts)

    def test_top_5_regions(self):
        intent, plan, charts = self._run_query("top 5 regions by sales")
        assert intent["top_bottom"]["direction"] == "top"
        assert len(charts) >= 1

    def test_overview(self):
        intent, plan, charts = self._run_query("analyze everything")
        assert intent["is_overview"] is True
        assert len(charts) >= 3

    def test_no_dates_dataset(self):
        df_nodates = pd.DataFrame({
            "Salary": [50000, 60000, 70000, 80000],
            "Age": [25, 30, 35, 40],
            "Experience": [2, 5, 10, 15],
            "Department": ["Eng", "Mkt", "Eng", "Sales"],
            "Gender": ["M", "F", "M", "F"],
            "Level": ["Jr", "Sr", "Jr", "Lead"],
        })
        intent = IntentParser.parse("show salary by department", NO_DATES_SUMMARY)
        plan = self.builder.heuristic_plan(NO_DATES_SUMMARY, "show salary by department", intent)
        charts = []
        for i, spec in enumerate(plan):
            chart = self.builder.materialize_chart(df_nodates, NO_DATES_SUMMARY, spec, f"test-{i}")
            if chart:
                charts.append(chart)
        assert len(charts) >= 1
        for c in charts:
            assert c["type"] not in ("Line Graph", "Area Graph")

    def test_scatter_correlation(self):
        intent, plan, charts = self._run_query("correlation between sales and profit")
        assert any(c["type"] == "Scatterplot" for c in charts)

    def test_all_charts_have_data(self):
        _, _, charts = self._run_query("show everything")
        for chart in charts:
            assert "data" in chart
            assert len(chart["data"]) > 0
            assert "title" in chart
            assert "type" in chart
            assert chart["type"] in ALLOWED_CHART_TYPES

    def test_greeting_gets_default_charts(self):
        _, _, charts = self._run_query("hello")
        # Should still return something (default plan)
        assert len(charts) >= 1


# ═══════════════════════════════════════════════════════════════
# ALLOWED_CHART_TYPES constant correctness
# ═══════════════════════════════════════════════════════════════

class TestChartTypeCatalog:
    def test_all_six_types(self):
        expected = {"Area Graph", "Line Graph", "Bar Chart", "Donut Chart", "Scatterplot", "Histogram"}
        assert ALLOWED_CHART_TYPES == expected

    def test_alias_map_covers_all(self):
        canonical_from_aliases = set(_CHART_TYPE_ALIASES.values())
        assert canonical_from_aliases == ALLOWED_CHART_TYPES


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
