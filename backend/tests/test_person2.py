"""Tests for Person 2 — IntentParser, ConversationMemory, and validation."""
import sys
import os
from pathlib import Path

# Add backend to path
_backend = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_backend))

import pytest
from app.agents.orchestrator import IntentParser, ConversationMemory


# ───────── Sample dataset summary used across tests ─────────
SAMPLE_SUMMARY = {
    "numeric_columns": ["Sales", "Profit", "Orders", "Cost"],
    "date_columns": ["Date"],
    "categorical_columns": ["Region", "Category", "Product"],
    "primary_kpi": "Sales",
    "col_meta": {
        "Region": {"name": "Region", "sample_values": ["North", "South", "East", "West"]},
        "Category": {"name": "Category", "sample_values": ["Electronics", "Furniture", "Office Supplies"]},
    },
    "row_count": 5000,
    "quality_score": 94,
}

SUMMARY_NO_DATES = {
    "numeric_columns": ["Salary", "Age", "Experience"],
    "date_columns": [],
    "categorical_columns": ["Department", "Gender"],
    "primary_kpi": "Salary",
    "col_meta": {},
    "row_count": 200,
}


# ═══════════════════ IntentParser Tests ═══════════════════

class TestIntentParserTimeFilter:
    def test_this_year(self):
        intent = IntentParser.parse("show sales this year", SAMPLE_SUMMARY)
        assert intent["time_filter"]["type"] == "this_year"

    def test_current_year(self):
        intent = IntentParser.parse("current year revenue", SAMPLE_SUMMARY)
        assert intent["time_filter"]["type"] == "this_year"

    def test_last_year(self):
        intent = IntentParser.parse("compare with last year", SAMPLE_SUMMARY)
        assert intent["time_filter"]["type"] == "last_year"

    def test_last_n_months(self):
        intent = IntentParser.parse("sales in the last 6 months", SAMPLE_SUMMARY)
        assert intent["time_filter"]["type"] == "last_n"
        assert intent["time_filter"]["n"] == 6
        assert intent["time_filter"]["unit"] == "month"

    def test_past_3_quarters(self):
        intent = IntentParser.parse("profit past 3 quarters", SAMPLE_SUMMARY)
        assert intent["time_filter"]["type"] == "last_n"
        assert intent["time_filter"]["n"] == 3
        assert intent["time_filter"]["unit"] == "quarter"

    def test_ytd(self):
        intent = IntentParser.parse("year to date orders", SAMPLE_SUMMARY)
        assert intent["time_filter"]["type"] == "ytd"

    def test_ytd_abbreviation(self):
        intent = IntentParser.parse("show YTD sales", SAMPLE_SUMMARY)
        assert intent["time_filter"]["type"] == "ytd"

    def test_last_quarter(self):
        intent = IntentParser.parse("last quarter profit", SAMPLE_SUMMARY)
        assert intent["time_filter"]["type"] == "last_quarter"

    def test_this_quarter(self):
        intent = IntentParser.parse("this quarter growth", SAMPLE_SUMMARY)
        assert intent["time_filter"]["type"] == "this_quarter"

    def test_specific_year(self):
        intent = IntentParser.parse("sales in 2024", SAMPLE_SUMMARY)
        assert intent["time_filter"]["type"] == "year"
        assert intent["time_filter"]["value"] == 2024

    def test_yoy(self):
        intent = IntentParser.parse("year over year comparison", SAMPLE_SUMMARY)
        assert intent["time_filter"]["type"] == "yoy"

    def test_mom(self):
        intent = IntentParser.parse("MoM growth", SAMPLE_SUMMARY)
        assert intent["time_filter"]["type"] == "mom"

    def test_no_time_filter(self):
        intent = IntentParser.parse("show sales by region", SAMPLE_SUMMARY)
        assert intent["time_filter"] is None


class TestIntentParserMetrics:
    def test_single_metric(self):
        intent = IntentParser.parse("show sales by region", SAMPLE_SUMMARY)
        assert "Sales" in intent["metrics"]

    def test_multiple_metrics(self):
        intent = IntentParser.parse("compare sales and profit", SAMPLE_SUMMARY)
        assert "Sales" in intent["metrics"]
        assert "Profit" in intent["metrics"]

    def test_default_to_primary_kpi(self):
        intent = IntentParser.parse("show me everything", SAMPLE_SUMMARY)
        assert intent["metrics"] == ["Sales"]


class TestIntentParserDimensions:
    def test_category_dimension(self):
        intent = IntentParser.parse("sales by region", SAMPLE_SUMMARY)
        assert "Region" in intent["dimensions"]

    def test_multiple_dimensions(self):
        intent = IntentParser.parse("sales by region and category", SAMPLE_SUMMARY)
        assert "Region" in intent["dimensions"]
        assert "Category" in intent["dimensions"]


class TestIntentParserComparison:
    def test_vs_comparison(self):
        intent = IntentParser.parse("sales vs profit", SAMPLE_SUMMARY)
        assert intent["comparison"] is not None
        cols = intent["comparison"]["columns"]
        assert "Sales" in cols
        assert "Profit" in cols

    def test_compare_keyword(self):
        intent = IntentParser.parse("compare sales and profit by region", SAMPLE_SUMMARY)
        assert intent["comparison"] is not None


class TestIntentParserTopBottom:
    def test_top_5(self):
        intent = IntentParser.parse("top 5 regions by sales", SAMPLE_SUMMARY)
        assert intent["top_bottom"] is not None
        assert intent["top_bottom"]["direction"] == "top"
        assert intent["top_bottom"]["n"] == 5

    def test_bottom_10(self):
        intent = IntentParser.parse("bottom 10 products by profit", SAMPLE_SUMMARY)
        assert intent["top_bottom"]["direction"] == "bottom"
        assert intent["top_bottom"]["n"] == 10

    def test_highest_default_n(self):
        intent = IntentParser.parse("highest sales by region", SAMPLE_SUMMARY)
        assert intent["top_bottom"]["direction"] == "top"
        assert intent["top_bottom"]["n"] == 5  # default


class TestIntentParserOverview:
    def test_overview(self):
        intent = IntentParser.parse("analyze everything", SAMPLE_SUMMARY)
        assert intent["is_overview"] is True

    def test_not_overview(self):
        intent = IntentParser.parse("show sales by region", SAMPLE_SUMMARY)
        assert intent["is_overview"] is False


class TestIntentParserEdgeCases:
    def test_no_date_columns(self):
        intent = IntentParser.parse("salary trend over time", SUMMARY_NO_DATES)
        # Should still parse but time filter applied to nothing
        assert intent["metrics"] == ["Salary"]

    def test_empty_query(self):
        intent = IntentParser.parse("", SAMPLE_SUMMARY)
        assert intent["metrics"] == ["Sales"]  # default KPI

    def test_greeting(self):
        intent = IntentParser.parse("hello", SAMPLE_SUMMARY)
        assert intent["metrics"] == ["Sales"]  # default KPI


# ═══════════════════ ConversationMemory Tests ═══════════════════

class TestConversationMemory:
    def test_add_and_retrieve(self):
        mem = ConversationMemory()
        intent = {"metrics": ["Sales"], "dimensions": ["Region"], "time_filter": None, "filters": [], "comparison": None}
        mem.add("ds-1", "show sales by region", intent)
        ctx = mem.get_context("ds-1")
        assert len(ctx) == 1
        assert ctx[0]["query"] == "show sales by region"

    def test_followup_resolution(self):
        mem = ConversationMemory()
        intent1 = {"metrics": ["Sales"], "dimensions": ["Region"], "time_filter": {"type": "this_year"}, "filters": [], "comparison": None}
        mem.add("ds-1", "show sales by region this year", intent1)

        # Follow-up: "now break that down by category"
        intent2 = {"metrics": [], "dimensions": ["Category"], "time_filter": None, "filters": [], "comparison": None, "raw_query": "now break that down by category"}
        resolved = mem.resolve_followup("ds-1", intent2, "now break that down by category")

        assert resolved["metrics"] == ["Sales"]  # carried from previous
        assert resolved["dimensions"] == ["Category"]  # new dimension
        assert resolved["time_filter"] == {"type": "this_year"}  # carried

    def test_non_followup(self):
        mem = ConversationMemory()
        intent1 = {"metrics": ["Sales"], "dimensions": [], "time_filter": None, "filters": [], "comparison": None}
        mem.add("ds-1", "show sales", intent1)

        intent2 = {"metrics": ["Profit"], "dimensions": ["Region"], "time_filter": None, "filters": [], "comparison": None, "raw_query": "show profit breakdown by region for last quarter"}
        resolved = mem.resolve_followup("ds-1", intent2, "show profit breakdown by region for last quarter")

        assert resolved["metrics"] == ["Profit"]  # new, not carried
        assert resolved["dimensions"] == ["Region"]

    def test_memory_limit(self):
        mem = ConversationMemory()
        for i in range(25):
            mem.add("ds-1", f"query {i}", {"metrics": [f"m{i}"]})
        ctx = mem.get_context("ds-1")
        assert len(ctx) == 20  # capped at 20


# ═══════════════════ LLM Validation Tests ═══════════════════

class TestLLMValidation:
    def setup_method(self):
        from app.core.llm_client import NemotronLLMClient
        self.client = NemotronLLMClient()

    def test_normalize_chart_type_exact(self):
        assert self.client._normalize_chart_type("Bar Chart") == "Bar Chart"

    def test_normalize_chart_type_alias(self):
        assert self.client._normalize_chart_type("bar") == "Bar Chart"
        assert self.client._normalize_chart_type("pie chart") == "Donut Chart"
        assert self.client._normalize_chart_type("scatter") == "Scatterplot"
        assert self.client._normalize_chart_type("area") == "Area Graph"

    def test_normalize_chart_type_unknown(self):
        assert self.client._normalize_chart_type("Sankey Diagram") is None
        assert self.client._normalize_chart_type("Choropleth") is None

    def test_fuzzy_match_column_exact(self):
        cols = ["Sales", "Profit", "Region"]
        assert self.client._fuzzy_match_column("Sales", cols) == "Sales"
        assert self.client._fuzzy_match_column("sales", cols) == "Sales"

    def test_fuzzy_match_column_close(self):
        cols = ["Sales", "Profit", "Region"]
        assert self.client._fuzzy_match_column("Saels", cols) == "Sales"  # typo

    def test_fuzzy_match_column_no_match(self):
        cols = ["Sales", "Profit"]
        assert self.client._fuzzy_match_column("Xyz123", cols) is None

    def test_validate_good_spec(self):
        spec = {"type": "Bar Chart", "x_axis": "Region", "y_axis": "Sales", "title": "Sales by Region", "insight_tooltip": "Shows sales across regions"}
        result = self.client._validate_and_repair_spec(spec, SAMPLE_SUMMARY)
        assert result is not None
        assert result["type"] == "Bar Chart"
        assert result["x_axis"] == "Region"
        assert result["y_axis"] == "Sales"

    def test_validate_repairs_chart_type(self):
        spec = {"type": "bar", "x_axis": "Region", "y_axis": "Sales", "title": "Test", "insight_tooltip": "test"}
        result = self.client._validate_and_repair_spec(spec, SAMPLE_SUMMARY)
        assert result["type"] == "Bar Chart"

    def test_validate_repairs_column_name(self):
        spec = {"type": "Bar Chart", "x_axis": "region", "y_axis": "sales", "title": "Test", "insight_tooltip": "test"}
        result = self.client._validate_and_repair_spec(spec, SAMPLE_SUMMARY)
        assert result["x_axis"] == "Region"
        assert result["y_axis"] == "Sales"

    def test_validate_drops_unknown_type(self):
        spec = {"type": "Sankey Diagram", "x_axis": "Region", "y_axis": "Sales", "title": "Test", "insight_tooltip": "test"}
        result = self.client._validate_and_repair_spec(spec, SAMPLE_SUMMARY)
        assert result is None

    def test_validate_repairs_missing_yaxis(self):
        spec = {"type": "Bar Chart", "x_axis": "Region", "y_axis": "NonexistentColumn", "title": "Test", "insight_tooltip": "test"}
        result = self.client._validate_and_repair_spec(spec, SAMPLE_SUMMARY)
        # Should fallback to primary_kpi
        assert result is not None
        assert result["y_axis"] == "Sales"

    def test_validate_generates_title(self):
        spec = {"type": "Bar Chart", "x_axis": "Region", "y_axis": "Sales", "title": "", "insight_tooltip": "test"}
        result = self.client._validate_and_repair_spec(spec, SAMPLE_SUMMARY)
        assert result["title"]  # auto-generated


# ═══════════════════ Heuristic Plan Tests ═══════════════════

class TestHeuristicPlan:
    def setup_method(self):
        from app.core.dashboard_builder import dashboard_builder
        self.builder = dashboard_builder

    def test_trend_query(self):
        plans = self.builder.heuristic_plan(SAMPLE_SUMMARY, "show sales trend over time")
        assert any(p["type"] in ("Line Graph", "Area Graph") for p in plans)

    def test_breakdown_query(self):
        plans = self.builder.heuristic_plan(SAMPLE_SUMMARY, "sales by region")
        assert any(p["type"] == "Bar Chart" for p in plans)

    def test_comparison_with_intent(self):
        intent = {"metrics": ["Sales", "Profit"], "dimensions": ["Region"], "comparison": {"columns": ["Sales", "Profit"]}, "time_filter": None, "filters": [], "top_bottom": None, "is_overview": False}
        plans = self.builder.heuristic_plan(SAMPLE_SUMMARY, "compare sales and profit", intent)
        assert len(plans) >= 2

    def test_overview_with_intent(self):
        intent = {"metrics": [], "dimensions": [], "comparison": None, "time_filter": None, "filters": [], "top_bottom": None, "is_overview": True}
        plans = self.builder.heuristic_plan(SAMPLE_SUMMARY, "analyze everything", intent)
        assert len(plans) >= 3

    def test_top_n_with_intent(self):
        intent = {"metrics": ["Sales"], "dimensions": ["Region"], "comparison": None, "time_filter": None, "filters": [], "top_bottom": {"direction": "top", "n": 5}, "is_overview": False}
        plans = self.builder.heuristic_plan(SAMPLE_SUMMARY, "top 5 regions by sales", intent)
        assert any("Top" in p.get("title", "") for p in plans)

    def test_no_dates_no_time_chart(self):
        plans = self.builder.heuristic_plan(SUMMARY_NO_DATES, "show salary trend")
        # Should not contain Line or Area chart since no date columns
        for p in plans:
            assert p["type"] not in ("Line Graph", "Area Graph")

    def test_backward_compat_no_intent(self):
        # heuristic_plan should still work without intent arg
        plans = self.builder.heuristic_plan(SAMPLE_SUMMARY, "show sales by region")
        assert len(plans) >= 1

    def test_max_5_charts(self):
        intent = {"metrics": [], "dimensions": [], "comparison": None, "time_filter": None, "filters": [], "top_bottom": None, "is_overview": True}
        plans = self.builder.heuristic_plan(SAMPLE_SUMMARY, "give me everything possible", intent)
        assert len(plans) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
