# AI Analyst evaluation report

## Status

The Phase 4 dry-run harness is wired and reproducible. Real model numbers are
pending an API key and a model-enabled run; dry-run numbers are not quality
measurements. This report is intentionally checked in at the AI module
documentation path required by section 8.3.

Run from `backend/`:

```bash
python -m app.ai.ai_analyst.eval run \
  --config app/ai/ai_analyst/eval/configs/A.yaml \
  --cases app/ai/ai_analyst/eval/cases/core.yaml \
  --out eval-report.json --dry-run
python -m app.ai.ai_analyst.eval report \
  --input eval-report.json --out eval-report.json
```

Reports include overall and per-category routing, plan validity, SQL
first-attempt success/retries, execution accuracy, verifier catch and
false-positive rates, correction and grounding failures, total p50/p95
latency, per-stage p50/p95 latency when stage mappings are supplied, tokens,
errors, and fallbacks.
