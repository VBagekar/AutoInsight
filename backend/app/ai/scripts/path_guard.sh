#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-main}"
ALLOWED_PREFIXES=("backend/app/ai/")
SHARED_EXCEPTIONS=("backend/app/main.py" "backend/requirements.txt" "backend/.env.example")

fail=0
while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  ok=0
  for prefix in "${ALLOWED_PREFIXES[@]}"; do [[ "$file" == "$prefix"* ]] && ok=1; done
  for exception in "${SHARED_EXCEPTIONS[@]}"; do [[ "$file" == "$exception" ]] && ok=1; done
  if [[ "$ok" -eq 0 ]]; then
    echo "OUTSIDE ALLOWED SET: $file"
    fail=1
  fi
done < <(git diff --name-only "$BASE"...HEAD; git diff --name-only; git ls-files --others --exclude-standard)

for exception in "${SHARED_EXCEPTIONS[@]}"; do
  deleted=$(git diff --numstat "$BASE"...HEAD -- "$exception" | awk '{sum += $2} END {print sum + 0}')
  if [[ "$deleted" -gt 0 ]]; then
    echo "NON-ADDITIVE CHANGE in $exception ($deleted deleted lines)"
    fail=1
  fi
done

if [[ "$fail" -eq 0 ]]; then
  echo "path guard: OK"
else
  echo "path guard: FAILED"
  exit 1
fi
