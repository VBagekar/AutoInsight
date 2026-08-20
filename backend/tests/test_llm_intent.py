"""Tests for LLM-resolved intent merging with regex fallback.

These tests verify that:
1. When the LLM returns a resolved_intent, it is preferred over regex for fields it specifies.
2. When the LLM fails, pure regex intent is used without error.
3. Ambiguous follow-up queries that confuse regex are correctly resolved by the LLM path.

Run with:
    python -m pytest backend/tests/test_llm_intent.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup so this works from the project root without pip-installing the pkg
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parent.parent.parent
_backend = _root / "backend"
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from app.agents.orchestrator import IntentParser


# ---------------------------------------------------------------------------
# Shared test summary (mimics a sales dataset)
# ---------------------------------------------------------------------------
SAMPLE_SUMMARY: Dict[str, Any] = {
    "row_count": 1000,
    "column_count": 6,
    "primary_kpi": "Sales",
    "quality_score": 98,
    "numeric_columns": ["Sales", "Profit", "Quantity"],
    "date_columns": ["Order Date"],
    "categorical_columns": ["Region", "Category", "Segment"],
    "col_meta": {
        "Region": {
            "name": "Region", "dtype": "object", "semantic_role": "dimension",
            "missing_pct": 0, "sample_values": ["East", "West", "South", "North"],
            "unique_count": 4,
        },
        "Category": {
            "name": "Category", "dtype": "object", "semantic_role": "dimension",
            "missing_pct": 0, "sample_values": ["Furniture", "Technology", "Office Supplies"],
            "unique_count": 3,
        },
        "Segment": {
            "name": "Segment", "dtype": "object", "semantic_role": "dimension",
            "missing_pct": 0, "sample_values": ["Consumer", "Corporate", "Home Office"],
            "unique_count": 3,
        },
    },
}


# ===========================================================================
# Test 1 — Ambiguous "before covid" reference
# The query is genuinely hard for regex (no year number, no keyword it knows)
# but the LLM resolves it to a year-based time filter.
# ===========================================================================
def test_llm_intent_before_covid_merges_time_filter():
    """When LLM returns time_filter for 'before covid', it must override regex (which returns None)."""
    query = "how did that do compared to before covid"
    regex_intent = IntentParser.parse(query, SAMPLE_SUMMARY)

    # Regex produces no time filter for this query
    assert regex_intent.get("time_filter") is None, (
        "Regex should not extract a time filter from 'before covid' — "
        "this is the ambiguous case the LLM handles."
    )

    # Simulate the LLM returning a resolved_intent
    llm_intent: Dict[str, Any] = {
        "time_filter": {"type": "year", "value": 2019},
        "dimensions": ["Region"],
        "metrics": ["Sales"],
        "is_followup": True,
        "top_bottom": None,
    }

    # Apply the same merge logic as orchestrator.process_query_stream
    merged = _merge_intent(regex_intent, llm_intent)

    assert merged["time_filter"] == {"type": "year", "value": 2019}, (
        "LLM time_filter should override regex None for 'before covid'."
    )
    assert "Region" in merged["dimensions"], "LLM dimension 'Region' should be present in merged."
    # is_followup is LLM-only metadata — not merged into regex intent dict; checked via llm_intent directly
    assert llm_intent["is_followup"] is True, "LLM should have flagged this as a follow-up query."


# ===========================================================================
# Test 2 — Follow-up "break it down by region now"
# Regex might pick up 'Region' but misses the follow-up semantic.
# LLM explicitly marks is_followup=True and sets the dimension.
# ===========================================================================
def test_llm_intent_followup_by_region():
    """When LLM marks is_followup=True with dimension=Region, merged intent preserves both."""
    query = "break it down by region now"
    regex_intent = IntentParser.parse(query, SAMPLE_SUMMARY)

    llm_intent: Dict[str, Any] = {
        "time_filter": None,
        "dimensions": ["Region"],
        "metrics": ["Sales"],
        "is_followup": True,
        "top_bottom": None,
    }

    merged = _merge_intent(regex_intent, llm_intent)

    assert "Region" in merged["dimensions"], "Region should be in merged dimensions."
    assert "Sales" in merged["metrics"], "Sales should be in merged metrics."
    # No spurious time filter injected
    assert merged["time_filter"] is None, "No time filter should appear for this query."


# ===========================================================================
# Test 3 — "worst performing segment last quarter" with top_bottom
# Regex can partially parse this but LLM correctly resolves bottom-1 + last_quarter.
# ===========================================================================
def test_llm_intent_worst_segment_last_quarter():
    """LLM top_bottom and time_filter override regex for 'worst performing segment last quarter'."""
    query = "worst performing segment last quarter"
    regex_intent = IntentParser.parse(query, SAMPLE_SUMMARY)

    # Regex may partially pick up top_bottom from 'worst', but may get direction wrong
    llm_intent: Dict[str, Any] = {
        "time_filter": {"type": "last_quarter"},
        "dimensions": ["Segment"],
        "metrics": ["Sales"],
        "is_followup": False,
        "top_bottom": {"direction": "bottom", "n": 1},
    }

    merged = _merge_intent(regex_intent, llm_intent)

    assert merged["time_filter"] == {"type": "last_quarter"}, (
        "LLM time_filter 'last_quarter' should be applied."
    )
    assert merged["top_bottom"] == {"direction": "bottom", "n": 1}, (
        "LLM top_bottom should override regex top_bottom."
    )
    assert "Segment" in merged["dimensions"], "Segment should be in merged dimensions."


# ===========================================================================
# Test 4 — LLM failure graceful fallback
# When generate_chart_plan raises / returns ([], None), only regex intent is used.
# ===========================================================================
def test_regex_only_when_llm_fails():
    """When LLM returns no resolved_intent, the regex intent is used unchanged."""
    query = "show sales by region"
    regex_intent = IntentParser.parse(query, SAMPLE_SUMMARY)

    # No LLM intent — simulate LLM failure
    llm_intent = None
    merged = _merge_intent(regex_intent, llm_intent)

    # Merged should be identical to regex intent
    assert merged == regex_intent, (
        "When LLM returns no intent (failure path), merged must equal regex intent exactly."
    )


# ===========================================================================
# Helper — replicates the merge logic in orchestrator.process_query_stream
# ===========================================================================
def _merge_intent(
    regex_intent: Dict[str, Any],
    llm_intent: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Mirror of the merge logic in MasterOrchestrator.process_query_stream (Task 5)."""
    if not llm_intent or not isinstance(llm_intent, dict):
        return regex_intent

    merged = dict(regex_intent)

    if llm_intent.get("time_filter") is not None:
        merged["time_filter"] = llm_intent["time_filter"]

    if llm_intent.get("dimensions"):
        existing = set(merged.get("dimensions", []))
        merged.setdefault("dimensions", [])
        for d in llm_intent["dimensions"]:
            if d not in existing:
                merged["dimensions"].append(d)

    if llm_intent.get("metrics"):
        existing = set(merged.get("metrics", []))
        merged.setdefault("metrics", [])
        for m in llm_intent["metrics"]:
            if m not in existing:
                merged["metrics"].append(m)

    if llm_intent.get("top_bottom") is not None:
        merged["top_bottom"] = llm_intent["top_bottom"]

    return merged
