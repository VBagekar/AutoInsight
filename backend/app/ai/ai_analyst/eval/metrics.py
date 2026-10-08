"""Aggregate metrics for evaluation reports."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable
import statistics


@dataclass
class Metrics:
    total: int
    exact_match: float
    execution_success: float
    numeric_accuracy: float
    verifier_catch_rate: float
    mean_latency_ms: float
    route_accuracy: float = 0.0
    plan_validity: float = 0.0
    sql_first_attempt_success: float = 0.0
    mean_retries: float = 0.0
    execution_accuracy: float = 0.0
    verifier_false_positive_rate: float = 0.0
    correction_success: float = 0.0
    grounding_failure_rate: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    tokens: int = 0
    error_rate: float = 0.0
    fallback_rate: float = 0.0
    stage_latency_p50_ms: dict[str, float] = None
    stage_latency_p95_ms: dict[str, float] = None

    def to_dict(self) -> dict:
        value = asdict(self)
        value["stage_latency_p50_ms"] = value["stage_latency_p50_ms"] or {}
        value["stage_latency_p95_ms"] = value["stage_latency_p95_ms"] or {}
        return value


def compute_metrics(results: Iterable[dict]) -> Metrics:
    rows = list(results)
    n = len(rows)
    if not n:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    latencies = [float(r.get("latency_ms", 0)) for r in rows]
    def rate(key, default=False):
        return sum(bool(r.get(key, default)) for r in rows) / n
    retries = [float(r.get("retries", 0)) for r in rows]
    stages: dict[str, list[float]] = {}
    for row in rows:
        mapping = row.get("stage_latency_ms") or row.get("stage_latencies") or {}
        for stage, latency in mapping.items():
            try:
                stages.setdefault(stage, []).append(float(latency))
            except (TypeError, ValueError):
                continue
    p50 = {stage: statistics.median(values) for stage, values in stages.items()}
    p95 = {stage: sorted(values)[max(0, int(0.95 * (len(values) - 1)))]
           for stage, values in stages.items()}
    return Metrics(
        n,
        sum(bool(r.get("exact_match")) for r in rows) / n,
        sum(bool(r.get("execution_success")) for r in rows) / n,
        sum(float(r.get("numeric_accuracy", 0)) for r in rows) / n,
        sum(bool(r.get("verifier_caught")) for r in rows) / n,
        sum(float(r.get("latency_ms", 0)) for r in rows) / n,
        rate("route_correct", True), rate("plan_valid", True),
        rate("sql_first_attempt", True), statistics.mean(retries),
        rate("execution_correct", None), rate("verifier_false_positive"),
        rate("correction_success"), rate("grounding_failure"),
        statistics.median(latencies), sorted(latencies)[max(0, int(0.95 * (n - 1)))],
        sum(int(r.get("tokens", 0)) for r in rows), rate("error"), rate("fallback"),
        p50, p95,
    )


def category_metrics(results: Iterable[dict]) -> dict[str, Metrics]:
    groups: dict[str, list[dict]] = {}
    for row in results:
        for category in row.get("categories", row.get("tags", ["uncategorized"])):
            groups.setdefault(category, []).append(row)
    return {category: compute_metrics(rows) for category, rows in groups.items()}
