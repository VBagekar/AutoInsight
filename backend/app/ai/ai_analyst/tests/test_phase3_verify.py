import os
from types import SimpleNamespace

import pandas as pd
import pytest

from app.ai.ai_analyst.verify.checks import run_checks
from app.ai.ai_analyst.verify.corrective import corrective_loop
from app.ai.ai_analyst.verify.grounding import (
    deterministic_answer, numeric_grounding, value_grounding,
)
from app.ai.ai_analyst.verify.schemas import Verification
from app.ai.ai_analyst.verify import verifier
import app.ai.ai_analyst.service as service


def result(rows, sql="SELECT 1", truncated=False, total_rows=None):
    frame = pd.DataFrame(rows)
    return SimpleNamespace(df=frame, sql=sql, truncated=truncated,
                           total_rows=total_rows if total_rows is not None else len(frame))


CARD = {"tables": [{"name": "sales", "columns": [
    {"name": "sale_date", "kind": "datetime", "min": "2023-01-01", "max": "2024-01-01"},
    {"name": "amount", "kind": "numeric", "min": 1, "max": 1000, "mean": 100,
     "median": 10, "std": 2},
    {"name": "change_percent", "kind": "numeric", "min": -1, "max": 1,
     "mean": 0, "median": 0, "std": 1},
]},], "join_hints": []}


@pytest.mark.parametrize("sql, rows, expected", [
    ("SELECT * FROM sales WHERE x=1", [], "EMPTY_RESULT"),
    ("SELECT * FROM sales", [{"x": 1}], "TRUNCATED"),
    ("SELECT x, y FROM sales", [{"x": None, "y": 1}], "ALL_NULL_COLUMN"),
    ("SELECT * FROM sales WHERE sale_date >= '2024-01-01' AND sale_date < '2025-01-01'",
     [{"x": 1}], "DATE_COVERAGE"),
    ("SELECT x FROM sales LIMIT 5", [{"x": 1}], "TOPN_SHORT"),
    ("SELECT SUM(change_percent) FROM sales", [{"x": 1}], "SUM_OF_RATIO"),
    ("SELECT AVG(amount) FROM sales", [{"x": 1}], "SKEWED_AVG"),
    ("SELECT SUM(s.amount) FROM sales s JOIN other o ON s.id=o.id", [{"x": 1}], "JOIN_FANOUT"),
])
def test_every_gate_one_check(sql, rows, expected):
    finding = run_checks("question", None, sql, result(rows, sql, truncated=expected == "TRUNCATED"), CARD)
    assert expected in {item.code for item in finding}


def test_checks_accept_dict_result_shape():
    findings = run_checks("q", None, "SELECT x FROM sales", {"df": pd.DataFrame({"x": [None]})}, CARD)
    assert any(f.code == "ALL_NULL_COLUMN" for f in findings)


def test_verifier_uses_configured_task_and_delimited_privacy_prompt(monkeypatch):
    captured = []
    def fake_call(task, messages, model):
        captured.extend(messages)
        return Verification(verdict="correct")
    monkeypatch.setattr(verifier, "call_json", fake_call)
    card = {"tables": [{"name": "sales", "n_rows": 1, "n_cols": 0, "source": "test",
                        "columns": [], "sample_rows": []}], "join_hints": []}
    out = verifier.verify("q", SimpleNamespace(restated_question="q"), "SELECT 1",
                           result([{"answer": 1}]), [], card)
    assert out.verdict == "correct"
    assert captured[1]["content"].startswith("<DATA_BLOCK>")
    assert "secret sample row" not in captured[1]["content"]


def test_privacy_omits_sample_rows_from_prompt_card(monkeypatch):
    monkeypatch.setattr(service, "AI_SEND_SAMPLE_ROWS", False)
    card = {"tables": [{"sample_rows": [{"secret": "sample row"}],
                        "columns": [{"name": "x", "top_values": [["secret", 1]]}]}]}
    safe = service._privacy_card(card)
    assert safe["tables"][0]["sample_rows"] == []
    assert card["tables"][0]["sample_rows"]


def test_verifier_incorrect_then_correct_and_feedback():
    initial = SimpleNamespace(result=result([{"answer": 1}], "SELECT wrong"))
    verdicts = iter([Verification(verdict="incorrect", fix_hint="use amount"),
                     Verification(verdict="correct")])
    attempts = []
    def verify_one(out, findings):
        attempts.append(out.result.sql)
        return next(verdicts)
    fixed = SimpleNamespace(result=result([{"answer": 2}], "SELECT amount"))
    outcome = corrective_loop(initial, verify=verify_one,
                              regenerate=lambda out, v, f: fixed, limit=2)
    assert outcome.status == "pass" and outcome.corrections == 1
    assert attempts == ["SELECT wrong", "SELECT amount"]


def test_verifier_exhaustion_identical_sql_and_ambiguous():
    initial = SimpleNamespace(result=result([{"x": 1}], "SELECT x"))
    bad = Verification(verdict="incorrect", fix_hint="fix")
    exhausted = corrective_loop(initial, verify=lambda *_: bad,
                                regenerate=lambda *_: SimpleNamespace(result=result([{"x": 1}], "SELECT y")), limit=1)
    assert exhausted.status == "fail" and exhausted.confidence == "low"
    identical = corrective_loop(initial, verify=lambda *_: bad,
                                regenerate=lambda *_: SimpleNamespace(result=result([{"x": 1}], " select   x; ")), limit=2)
    assert identical.status == "fail" and "same query" in identical.caveat
    ambiguous = corrective_loop(initial,
                                verify=lambda *_: Verification(verdict="ambiguous"),
                                regenerate=lambda *_: pytest.fail("must not regenerate"))
    assert ambiguous.status == "ambiguous" and ambiguous.confidence == "medium"


def test_verifier_disabled_skips_gate_two(monkeypatch):
    monkeypatch.setenv("AI_VERIFIER_ENABLED", "false")
    called = []
    outcome = corrective_loop(SimpleNamespace(result=result([{"x": 1}])), verify=lambda *_: called.append(1),
                              regenerate=lambda *_: None)
    assert outcome.status == "pass" and outcome.confidence == "high" and not called


def test_numeric_grounding_regenerate_then_fallback():
    out = result([{"total": 100}], total_rows=1)
    assert numeric_grounding("The total is 100.", out)[0]
    assert not numeric_grounding("The total is 999.", out)[0]
    assert deterministic_answer(out).startswith("The result is")


def test_value_grounding_is_case_evidence_and_safe():
    calls = []
    def fake_run(session, query):
        calls.append(query)
        return {"df": pd.DataFrame({"category": ["Electronics", "Books"]})}
    evidence = value_grounding("s", "SELECT * FROM sales WHERE category = 'electronics'", fake_run)
    assert evidence["category"] == ["Electronics", "Books"]
    assert '"category"' in calls[0] and '"sales"' in calls[0]
    value_grounding("s", "SELECT * FROM sales; DROP TABLE users", fake_run)
    assert ";" not in calls[-1]
