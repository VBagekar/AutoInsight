from __future__ import annotations
import json
from pathlib import Path
from app.ai.ai_analyst.agent.llm_json import call_json
from app.ai.ai_analyst.agent.context import load_prompt
from app.ai.ai_analyst.profile.datacard import render_card_text
from .schemas import Verification


def _preview(result) -> str:
    df = getattr(result, "df", None)
    if df is None: return "[]"
    return json.dumps(df.head(20).iloc[:, :12].to_dict(orient="records"), default=str)


def verify(question, plan, sql, result, findings, card, *, call=None) -> Verification:
    prompt = load_prompt("verifier")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": (
            "<DATA_BLOCK>\nQUESTION\n" + question +
            "\nRESTATED_QUESTION\n" + getattr(plan, "restated_question", "") +
            "\nPLAN\n" + (plan.model_dump_json() if hasattr(plan, "model_dump_json") else json.dumps(plan, default=str)) +
            "\nSQL\n" + sql + "\nRESULT_PREVIEW\n" + _preview(result) +
            "\nGATE_1_FINDINGS\n" + json.dumps([f.model_dump() for f in findings]) +
            "\nDATA_CARD\n" + render_card_text(card) + "\n</DATA_BLOCK>"
        )},
    ]
    return (call or call_json)("verifier", messages, Verification)
