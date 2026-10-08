"""JSON and Markdown report generation."""
from __future__ import annotations
import json
from pathlib import Path
from .metrics import compute_metrics, category_metrics


def make_report(results: list[dict], *, model_set: str = "offline") -> dict:
    return {"model_set": model_set, "metrics": compute_metrics(results).to_dict(),
            "categories": {k: v.to_dict() for k, v in category_metrics(results).items()},
            "results": results, "real_numbers_pending": True}


def write_report(report: dict, output: str | Path) -> tuple[Path, Path]:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path if path.suffix == ".json" else path.with_suffix(".json")
    md_path = json_path.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    m = report["metrics"]
    lines = ["# AI Analyst Evaluation", "", f"**Model set:** `{report['model_set']}`",
             "", "> Real model numbers are pending an API key; this report is dry-run/offline.",
             "",
             "| Metric | Value |", "|---|---:|"]
    lines += [f"| {k.replace('_', ' ').title()} | {v:.4f} |" if isinstance(v, float)
              else f"| {k.replace('_', ' ').title()} | {v} |"
              for k, v in m.items() if not k.startswith("stage_latency_")]
    if m.get("stage_latency_p50_ms") or m.get("stage_latency_p95_ms"):
        lines += ["", "## Stage latency", "", "| Stage | P50 (ms) | P95 (ms) |",
                  "|---|---:|---:|"]
        stages = sorted(set(m.get("stage_latency_p50_ms", {})) |
                        set(m.get("stage_latency_p95_ms", {})))
        lines += [f"| {stage} | {m.get('stage_latency_p50_ms', {}).get(stage, 0):.2f} | "
                  f"{m.get('stage_latency_p95_ms', {}).get(stage, 0):.2f} |"
                  for stage in stages]
    lines += ["", "## Cases", "", "| Case | Intent | Exact | Success | Latency (ms) |",
              "|---|---|---:|---:|---:|"]
    lines += [f"| {r.get('case_id')} | {r.get('intent')} | {str(bool(r.get('exact_match'))).lower()} | "
              f"{str(bool(r.get('execution_success'))).lower()} | {r.get('latency_ms', 0):.2f} |"
              for r in report["results"]]
    lines += ["", "## Per category", "", "| Category | Cases | Route accuracy | P95 latency |",
              "|---|---:|---:|---:|"]
    lines += [f"| {k} | {v['total']} | {v['route_accuracy']:.3f} | {v['latency_p95_ms']:.2f} |"
              for k, v in report["categories"].items()]
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path
