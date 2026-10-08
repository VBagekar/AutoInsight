"""Tolerant comparison of model outputs and expected tabular results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import pandas as pd


@dataclass
class Comparison:
    exact: bool
    numeric_accuracy: float
    ordering_match: bool = True
    column_match: bool = True
    reasons: list[str] = field(default_factory=list)


def _frame(value: Any) -> pd.DataFrame | None:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict) and "rows" in value:
        return pd.DataFrame(value["rows"])
    return None


def compare(actual: Any, expected: Any, *, tolerance: float = 1e-6,
            ordered: bool = False) -> Comparison:
    """Compare scalars or tables, ignoring row order unless ``ordered`` is set."""
    if actual is None or expected is None:
        return Comparison(actual == expected, 1.0 if actual == expected else 0.0,
                          reasons=[] if actual == expected else ["value mismatch"])
    af, ef = _frame(actual), _frame(expected)
    if af is None or ef is None:
        try:
            ok = math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)
            return Comparison(ok, 1.0 if ok else 0.0, reasons=[] if ok else ["numeric mismatch"])
        except (TypeError, ValueError):
            ok = actual == expected
            return Comparison(ok, 1.0 if ok else 0.0, reasons=[] if ok else ["value mismatch"])
    column_match = list(af.columns) == list(ef.columns)
    if not column_match:
        return Comparison(False, 0.0, column_match=False, reasons=["column mismatch"])
    if not ordered:
        # Stable string keys make mixed null/numeric columns sortable.
        key = lambda df: df.astype(object).where(df.notna(), "<NULL>").astype(str).sort_values(
            by=list(df.columns), kind="mergesort"
        ).index
        af, ef = af.iloc[list(key(af))].reset_index(drop=True), ef.iloc[list(key(ef))].reset_index(drop=True)
    ordering_match = af.shape == ef.shape
    if ordering_match:
        for col in ef.columns:
            for actual_cell, expected_cell in zip(af[col], ef[col]):
                if pd.isna(actual_cell) and pd.isna(expected_cell):
                    continue
                if pd.api.types.is_number(actual_cell) and pd.api.types.is_number(expected_cell):
                    if not math.isclose(float(actual_cell), float(expected_cell),
                                        rel_tol=tolerance, abs_tol=tolerance):
                        ordering_match = False
                        break
                elif actual_cell != expected_cell:
                    ordering_match = False
                    break
            if not ordering_match:
                break
    numeric = []
    for col in ef.columns:
        if pd.api.types.is_numeric_dtype(ef[col]):
            av, ev = pd.to_numeric(af[col], errors="coerce"), pd.to_numeric(ef[col], errors="coerce")
            numeric.extend((abs(float(a) - float(e)) <= tolerance * max(1, abs(float(e))))
                           for a, e in zip(av, ev) if pd.notna(a) and pd.notna(e))
    accuracy = sum(numeric) / len(numeric) if numeric else float(ordering_match)
    return Comparison(ordering_match and column_match, accuracy, ordering_match, column_match,
                      [] if ordering_match else ["row/value mismatch"])
