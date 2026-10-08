"""Offline evaluation harness for the AI analyst pipeline."""

from .cases import CASES, EvalCase, build_cases
from .comparator import Comparison, compare
from .metrics import Metrics, compute_metrics

__all__ = ["CASES", "EvalCase", "build_cases", "Comparison", "compare", "Metrics", "compute_metrics"]
