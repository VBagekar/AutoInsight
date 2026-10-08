from pathlib import Path
import pandas as pd

from app.ai.ai_analyst.eval.cases import CASES, INTENTS
from app.ai.ai_analyst.eval.comparator import compare
from app.ai.ai_analyst.eval.datasets import dataset_fingerprint, seed_datasets
from app.ai.ai_analyst.eval.metrics import compute_metrics
from app.ai.ai_analyst.eval.report import make_report, write_report
from app.ai.ai_analyst.eval.runner import run
from app.ai.ai_analyst.eval.metrics import category_metrics


def test_case_count_and_all_intents():
    assert len(CASES) >= 40
    assert {case.intent for case in CASES} == set(INTENTS)
    assert all(case.dataset and case.expected is not None for case in CASES)


def test_datasets_are_seeded_and_fresh():
    assert dataset_fingerprint() == dataset_fingerprint()
    left, right = seed_datasets(), seed_datasets()
    left["sales"].loc[0, "amount"] = 999
    assert right["sales"].loc[0, "amount"] == 10
    assert left["sales"].isna().sum().sum() > 0


def test_comparator_tolerance_order_and_columns():
    assert compare(1.0000001, 1.0).exact
    assert compare(pd.DataFrame({"x": [2, 1]}), pd.DataFrame({"x": [1, 2]})).exact
    assert not compare(pd.DataFrame({"x": [1]}), pd.DataFrame({"y": [1]})).column_match
    assert not compare(pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"x": [2, 1]}), ordered=True).exact


def test_metrics_and_reports():
    rows = [{"case_id": "a", "exact_match": True, "execution_success": True,
             "numeric_accuracy": 1, "verifier_caught": False, "latency_ms": 4}]
    metrics = compute_metrics(rows)
    assert metrics.total == 1 and metrics.mean_latency_ms == 4
    assert metrics.route_accuracy == 1 and metrics.latency_p95_ms == 4
    assert category_metrics([{**rows[0], "categories": ["aggregation"]}])["aggregation"].total == 1
    base = Path("ai_eval_test_report.json")
    try:
        paths = write_report(make_report(rows), base)
        assert all(path.exists() for path in paths)
        assert "Exact Match" in paths[1].read_text()
    finally:
        for path in (base, base.with_suffix(".md")):
            path.unlink(missing_ok=True)


def test_stage_latency_percentiles_and_report():
    report = make_report([
        {"case_id": "a", "exact_match": True, "execution_success": True,
         "latency_ms": 30, "stage_latency_ms": {"router": 10, "sql": 20}},
        {"case_id": "b", "exact_match": True, "execution_success": True,
         "latency_ms": 50, "stage_latency_ms": {"router": 30, "sql": 40}},
    ])
    assert report["metrics"]["stage_latency_p50_ms"] == {"router": 20, "sql": 30}
    assert report["metrics"]["stage_latency_p95_ms"] == {"router": 10, "sql": 20}
    base = Path("ai_eval_stage_report.json")
    try:
        _, markdown = write_report(report, base)
        text = markdown.read_text()
        assert "Stage latency" in text and "router" in text
    finally:
        base.unlink(missing_ok=True)
        base.with_suffix(".md").unlink(missing_ok=True)


def test_runner_resume_and_dry_run():
    state, output = Path("ai_eval_state.jsonl"), Path("ai_eval_resume.json")
    dry_state, dry_output = Path("ai_eval_dry.jsonl"), Path("ai_eval_dry.json")
    try:
        first = run(limit=3, state=state, output=output)
        assert first["metrics"]["total"] == 3
        before = len(state.read_text().splitlines())
        second = run(limit=3, state=state, output=output)
        assert len(state.read_text().splitlines()) == before
        assert second["metrics"]["total"] == 3
        dry = run(limit=1, state=dry_state, output=dry_output, dry_run=True)
        assert dry["metrics"]["execution_success"] == 0
    finally:
        for path in (state, output, output.with_suffix(".md"), dry_state,
                     dry_output, dry_output.with_suffix(".md")):
            path.unlink(missing_ok=True)
