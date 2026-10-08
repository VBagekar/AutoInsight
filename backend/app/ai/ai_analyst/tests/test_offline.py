import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.ai.ai_analyst.agent.executor import UnsafeSQL, check_sql
from app.ai.ai_analyst.agent.llm_json import call_json, extract_json
from app.ai.ai_analyst.agent.schemas import Route
from app.ai.ai_analyst.service import MissingAPIKeyError, AnalystService


def test_read_only_sql_guard():
    assert check_sql("SELECT count(*) FROM sales;") == "SELECT count(*) FROM sales"
    for sql in ("DROP TABLE sales", "SELECT * FROM read_csv_auto('x.csv')", "SELECT 1; SELECT 2"):
        try:
            check_sql(sql)
        except UnsafeSQL:
            pass
        else:
            raise AssertionError(sql)


def test_json_extraction_from_thinking_response():
    assert extract_json('<think>ignore</think>```json\n{"intent":"chitchat"}\n```') == {"intent": "chitchat"}


def test_fake_llm_drives_typed_route(monkeypatch):
    monkeypatch.setattr(
        "app.ai.ai_analyst.agent.llm_json.call_llm",
        lambda task, messages: '{"intent": "chitchat", "reason": "hello"}',
    )
    route = call_json("router", [{"role": "user", "content": "hi"}], Route)
    assert route.intent == "chitchat"


def test_missing_key_is_lazy(monkeypatch):
    monkeypatch.setattr("app.ai.ai_analyst.service.NVIDIA_API_KEY", None)
    try:
        AnalystService().ask("missing", "hello", "development-user")
    except MissingAPIKeyError:
        pass
    else:
        raise AssertionError("missing key should fail only when asking")
