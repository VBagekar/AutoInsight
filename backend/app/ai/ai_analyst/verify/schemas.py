from typing import Literal
from pydantic import BaseModel, Field

FindingLevel = Literal["info", "warn", "error"]
Verdict = Literal["correct", "incorrect", "ambiguous"]
IssueType = Literal[
    "wrong_column", "wrong_filter", "wrong_grain", "wrong_aggregation",
    "missing_condition", "misread_question", "other",
]


class Finding(BaseModel):
    level: FindingLevel
    code: str
    message: str


class VerifierIssue(BaseModel):
    type: IssueType
    detail: str


class Verification(BaseModel):
    verdict: Verdict
    issues: list[VerifierIssue] = Field(default_factory=list)
    fix_hint: str = ""


class VerificationResult(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    verification: Verification | None = None
    status: Literal["pass", "ambiguous", "fail"] = "pass"
    confidence: Literal["high", "medium", "low"] = "high"
    corrections: int = 0
    timing_ms: dict[str, float] = Field(default_factory=dict)
    caveat: str | None = None
