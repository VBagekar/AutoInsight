"""Module entry point: python -m app.ai.ai_analyst.eval run|report."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .runner import run
from .report import write_report


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.ai.ai_analyst.eval")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config")
    run_parser.add_argument("--cases")
    run_parser.add_argument("--out", default="eval-report.json")
    run_parser.add_argument("--model-set", default="offline")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--dry-run", action="store_true")
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--input", default="eval-report.json")
    report_parser.add_argument("--out", default="eval-report.json")
    args = parser.parse_args()
    if args.command == "run":
        result = run(config=args.config, cases=args.cases, out=args.out,
                     model_set=args.model_set, limit=args.limit, dry_run=args.dry_run)
    else:
        result = json.loads(Path(args.input).read_text())
        write_report(result, args.out)
    print(json.dumps(result.get("metrics", {}), sort_keys=True))


if __name__ == "__main__":
    main()
