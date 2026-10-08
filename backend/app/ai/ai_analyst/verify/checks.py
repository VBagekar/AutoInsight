"""Deterministic, best-effort checks run before the model verifier."""
from __future__ import annotations
import re
from typing import Any
from .schemas import Finding


def _df(result):
    return result.get("df") if isinstance(result, dict) else getattr(result, "df", None)


def _sql(result, sql=None):
    return sql or (result.get("sql", "") if isinstance(result, dict) else getattr(result, "sql", ""))


def _columns(card: dict) -> list[tuple[str, dict]]:
    return [(c["name"], c) for t in card.get("tables", []) for c in t.get("columns", [])]


def _f(level, code, message): return Finding(level=level, code=code, message=message)


def check_empty_result(result, **_) -> Finding | None:
    df = _df(result)
    if df is not None and len(df.index) == 0:
        return _f("warn", "EMPTY_RESULT", "The query returned zero rows.")


def check_truncated(result, **_) -> Finding | None:
    if getattr(result, "truncated", result.get("truncated", False) if isinstance(result, dict) else False):
        return _f("info", "TRUNCATED", "The result reached the row cap and may be incomplete.")


def check_all_null(result, **_) -> Finding | None:
    df = _df(result)
    if df is not None:
        nulls = [str(c) for c in df.columns if df[c].isna().all()]
        if nulls:
            return _f("error", "ALL_NULL_COLUMN", "Result column(s) are entirely NULL: " + ", ".join(nulls))


def check_date_coverage(sql, card, **_) -> Finding | None:
    sql = sql or ""
    date_cols = [(name, c) for name, c in _columns(card)
                 if c.get("kind") == "datetime" or "date" in name.lower() or "time" in name.lower()]
    for name, c in date_cols:
        if c.get("min") is None or c.get("max") is None:
            continue
        # Restrict the check to columns referenced by the query; otherwise an
        # unrelated date column can produce a misleading warning.
        if not re.search(rf"\b{re.escape(name)}\b", sql, re.I):
            continue
        max_date = str(c["max"])[:10]
        literals = re.findall(r"'(\d{4}(?:-\d{2}-\d{2})?)'", sql)
        years = [int(x) for x in re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", sql)]
        requested_end = max((x + "-12-31" if len(x) == 4 else x)
                            for x in literals if x[:4].isdigit()) if literals else None
        if not requested_end and years:
            requested_end = f"{max(years)}-12-31"
        if requested_end and requested_end > max_date:
            return _f("warn", "DATE_COVERAGE",
                      f"Date filter for {name} extends beyond the covered range "
                      f"({c['min']} to {c['max']}).")


def check_topn_short(result, sql, **_) -> Finding | None:
    m = re.search(r"\blimit\s+(\d+)", sql or "", re.I)
    df = _df(result)
    if m and df is not None and len(df.index) < int(m.group(1)):
        return _f("info", "TOPN_SHORT", f"LIMIT {m.group(1)} requested but only {len(df.index)} rows were returned.")


def check_sum_ratio(sql, **_) -> Finding | None:
    if re.search(r"\bsum\s*\(\s*[\"`]?[\w.]*?(percent|pct|rate|ratio|change)", sql or "", re.I):
        return _f("warn", "SUM_OF_RATIO", "SUM is applied to a ratio, rate, percent, or change column.")


def check_skewed_avg(sql, card, **_) -> Finding | None:
    for name, c in _columns(card):
        if re.search(r"\bavg\s*\(\s*[\"`]?"+re.escape(name)+r"\b", sql or "", re.I):
            try:
                mean, median, std, maximum = map(float, (c.get("mean"), c.get("median"), c.get("std"), c.get("max")))
                if abs(mean - median) > 3 * max(std, 1e-9) or (abs(median) > 1e-9 and maximum > 20 * abs(median)):
                    return _f("warn", "SKEWED_AVG", f"AVG({name}) may be misleading because the column is heavily skewed.")
            except (TypeError, ValueError):
                pass


def check_join_fanout(sql, **_) -> Finding | None:
    if re.search(r"\bjoin\b", sql or "", re.I) and re.search(r"\bsum\s*\(", sql or "", re.I):
        return _f("warn", "JOIN_FANOUT", "A JOIN with SUM may duplicate rows; compare with the base-table total.")


CHECKS = (check_empty_result, check_truncated, check_all_null, check_date_coverage,
          check_topn_short, check_sum_ratio, check_skewed_avg, check_join_fanout)


def run_checks(question: str, plan: Any, sql: str, result: Any, card: dict) -> list[Finding]:
    findings = []
    for check in CHECKS:
        finding = check(question=question, plan=plan, sql=sql, result=result, card=card)
        if finding: findings.append(finding)
    return findings
