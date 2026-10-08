from __future__ import annotations
import math, re
from typing import Any

_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:\s*[kKmM])?%?")


def _number(text: str) -> float:
    raw = text.replace(",", "").replace(" ", "")
    suffix = raw[-1:].lower() if raw[-1:].lower() in "km" else ""
    if suffix: raw = raw[:-1]
    value = float(raw.rstrip("%"))
    return value * (1000 if suffix == "k" else 1_000_000 if suffix == "m" else 1)


def _values(result) -> list[float]:
    df = result.get("df") if isinstance(result, dict) else getattr(result, "df", None)
    vals = []
    if df is not None:
        for value in df.to_numpy().flat:
            try:
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    vals.append(float(value))
            except (TypeError, ValueError): pass
    total = result.get("row_count") if isinstance(result, dict) else getattr(result, "total_rows", None)
    if total is not None: vals.append(float(total))
    return vals


def numeric_grounding(answer: str, result, question: str = "") -> tuple[bool, list[str]]:
    ignored = {_number(x) for x in _NUMBER.findall(question)}
    available = _values(result)
    missing = []
    for token in _NUMBER.findall(answer or ""):
        value = _number(token)
        if 1900 <= value <= 2100 and (value in ignored or re.search(rf"\b{int(value)}\b", question)):
            continue
        candidates = [value / 100] if token.endswith("%") else [value]
        if any(abs(candidate - actual) <= max(abs(actual) * .005, .01)
               for candidate in candidates for actual in available):
            continue
        missing.append(token)
    return not missing, missing


def deterministic_answer(result, question: str = "") -> str:
    df = result.get("df") if isinstance(result, dict) else getattr(result, "df", None)
    total = result.get("row_count") if isinstance(result, dict) else getattr(result, "total_rows", None)
    if df is None or df.empty:
        return "No matching rows were found."
    if len(df) == 1 and len(df.columns) == 1:
        return f"The result is {df.iloc[0, 0]}."
    preview = df.head(5).to_dict(orient="records")
    suffix = f" ({total:,} rows total)." if isinstance(total, int) else "."
    return "The query returned: " + repr(preview) + suffix


def extract_filter_values(sql: str) -> list[tuple[str, str]]:
    pattern = r"(?:[\w.\"`]+\.)?([A-Za-z_]\w*)\s*=\s*'((?:''|[^'])*)'"
    return [(column, value.replace("''", "'")) for column, value in re.findall(pattern, sql or "", re.I)]


def value_grounding(session_id: str, sql: str, run_query) -> dict[str, list[Any]]:
    """Fetch distinct alternatives for string filters, never interpolating raw values."""
    evidence = {}
    table_match = re.search(r"\bfrom\s+([A-Za-z_]\w*)\b", sql or "", re.I)
    if not table_match:
        return evidence
    table = table_match.group(1)
    for column, _ in extract_filter_values(sql):
        if not re.fullmatch(r"[A-Za-z_]\w*", column):
            raise ValueError(f"Unsafe value-grounding column identifier: {column!r}")
        query = f'SELECT DISTINCT "{column}" FROM "{table}" LIMIT 20'
        rows = run_query(session_id, query)
        df = rows.get("df") if isinstance(rows, dict) else getattr(rows, "df", None)
        evidence[column] = [] if df is None else df.iloc[:, 0].dropna().tolist()[:20]
    return evidence
