# AI Analyst Integration Notes

## Phase 0: Discovery

Discovery was performed without modifying the existing application, frontend, or
POC. The POC at `../../../../ai-analyst-poc/` remains read-only.

### Proposed integration location

- **`<AI_DIR>`:** `backend/app/ai/`
- **`<PKG>`:** `ai_analyst`
- **Proposed path:** Path A, an in-process package at
  `backend/app/ai/ai_analyst/`
- There is no existing isolated AI/RAG directory. The existing AI and data
  features are spread across `backend/app/api/`, `backend/app/agents/`, and
  `backend/app/core/`; those directories contain shared upload, profiling,
  preprocessing, forecasting, and dashboard behavior and are not safe to
  reuse as `<AI_DIR>`.

### 1. Backend stack

- Language: Python.
- Framework: FastAPI.
- Entry point: `backend/app/main.py`.
- Router registration: `main.py` imports routers from `app.api` and mounts them
  with `app.include_router(..., prefix="/api", tags=[...])`.
- Existing routes include upload, query, forecast, and preprocessing routers.

Evidence:

- `backend/app/main.py`
- `backend/app/api/upload.py`
- `backend/app/api/query.py`
- `backend/app/api/forecast.py`
- `backend/app/api/preprocess.py`

### 2. Runtime and dependency management

- Runtime command: `python --version` reported **Python 3.14.7**.
- Backend dependency manager: `backend/requirements.txt`; no backend
  `pyproject.toml`, Poetry configuration, or setup configuration was found.
- `backend/requirements.txt` declares:
  - `fastapi>=0.109.0`
  - `pydantic>=2.5.0`
  - `pydantic-settings>=2.1.0`
  - pandas, openpyxl, numpy, duckdb, openai, and related runtime packages.
- The declared pydantic major version is **v2**. The POC lock-style
  requirements also use pydantic v2 and DuckDB, so no version-major conflict
  was found from manifests.
- The repository's JavaScript package manager surface is the root
  `package.json`/`package-lock.json`, but it is not the backend dependency
  manager.

Evidence/commands:

```text
python3 --version
find AutoInsight -maxdepth 3 -type f ...
backend/requirements.txt
```

### 3. Authentication

- No authentication dependency, bearer/JWT handling, current-user dependency,
  user model, or auth middleware was found in `backend/app` or `src`.
- `main.py` currently configures only CORS middleware.
- Existing route handlers accept requests without an auth dependency and have no
  current-user id available.

Evidence:

- `backend/app/main.py`
- `backend/app/api/*.py`
- repository search for `auth`, `authorization`, `bearer`, `jwt`,
  `current_user`, `user_id`, `oauth`, and `token`.

Impact: the new module cannot yet reuse an existing authentication mechanism.
This must be resolved or explicitly supplied by the owner before implementing
the requirement that all new routes require existing auth; no replacement auth
will be introduced by this work.

### 4. Dataset receipt and storage

- `POST /api/upload` accepts one CSV/XLS/XLSX `UploadFile`, reads it into
  memory, and passes bytes and the client filename to
  `master_orchestrator.process_file_and_generate_initial_dashboard`.
- `MasterOrchestrator` stores datasets in an in-memory `datasets` dictionary;
  the stored entry contains the cleaned dataframe, filename, summary, and
  dashboard-related state.
- `GET /api/dataset/{dataset_id}/preview`,
  `GET /api/dataset/{dataset_id}/download`, and
  `POST /api/dataset/{dataset_id}/preprocess` read and mutate that in-memory
  state.
- No object storage or persistent application dataset database was found.
- The new chatbot therefore cannot safely point at a persistent shared file
  store today without changing the existing upload flow. A DatasetSource
  adapter could call an existing read function only if a suitable stable
  function is identified later; none was found in Phase 0.

Evidence:

- `backend/app/api/upload.py`
- `backend/app/api/preprocess.py`
- `backend/app/agents/orchestrator.py`
- `backend/app/core/dataset_profiler.py`

### 5. Existing AI/RAG chatbot and adjacent AI code

No vector store or embedding/RAG package was found. The existing AI/data
pipeline consists of:

- `backend/app/api/query.py` — legacy streaming natural-language query route
  (`POST /api/query`).
- `backend/app/agents/orchestrator.py` — upload orchestration, in-memory
  dataset state, conversation memory, intent parsing, query stream, chart
  planning, dashboard and report orchestration.
- `backend/app/agents/forecast_agent.py` — forecasting agent and route support.
- `backend/app/core/llm_client.py` — NVIDIA/OpenAI-compatible client, prompts,
  fallback behavior, chart planning, preprocessing action generation, and
  narrative generation.
- `backend/app/core/dataset_profiler.py` — local profiling and summary logic.
- `backend/app/core/data_cleaner.py` — upload cleaning behavior.
- `backend/app/core/dashboard_builder.py` — locally computed dashboard payloads.
- `backend/app/core/preprocessing_actions.py` — preprocessing command executor.
- `backend/app/api/upload.py` — upload, preview, and download routes.
- `backend/app/api/preprocess.py` — preprocessing route.
- `backend/app/api/forecast.py` and `backend/app/agents/forecast_agent.py` —
  forecast route/agent.
- `backend/tests/test_llm_intent.py`,
  `backend/tests/test_person2.py`, and
  `backend/tests/test_person2_exhaustive.py` — existing AI/orchestration tests.
- `backend/app/config.py` — NVIDIA settings and `MAX_UPLOAD_SIZE_MB`.
- `backend/.env.example` — `NVIDIA_API_KEY`, `NVIDIA_BASE_URL`, and
  `NVIDIA_MODEL`.
- `backend/requirements.txt` — FastAPI, pandas, DuckDB, OpenAI, and related
  dependencies used by the existing backend.

These are not classified for removal in Phase 0. Phase 1 must inventory each
symbol/file with whole-repository usage and authorship evidence before any
deletion is considered.

### 6. Callers and legacy API contracts

The frontend caller is `src/api/client.js`, used by
`src/components/Dashboard.jsx`.

Legacy contracts observed:

- `POST /api/upload`: multipart field `file`; optional query `sheet_name`.
  Expects a JSON object containing at least `dataset_id`, `summary`, and
  dashboard fields such as `charts`, `kpi_summary`, `ai_insights`, and
  `forecast`.
- `POST /api/query`: JSON `{query, dataset_id}`; consumes
  `text/event-stream` events in `data: <JSON>\n\n` form. Event payloads include
  `type: "thinking"`, `type: "payload"`/`data`, and `type: "error"` with
  `content`/`error_type`.
- `GET /api/dataset/{dataset_id}/preview?page=<n>&page_size=<n>`: expects a
  paginated preview JSON object.
- `GET /api/dataset/{dataset_id}/download`: expects CSV bytes and a
  `Content-Disposition` filename.
- `POST /api/dataset/{dataset_id}/preprocess`: JSON `{command}`; expects
  `confirmation_message`, `explanation`, and `updated_summary`.
- `POST /api/forecast`: JSON `{historical_values, periods}`; expects forecast
  JSON.
- `GET /api/health`: expects `llm_configured`, `llm_reachable`, and `model`.

R12 consequence: these routes must remain working until the owner approves a
cutover, and the frontend must not be edited by this integration.

Evidence:

- `src/api/client.js`
- `src/components/Dashboard.jsx`
- `backend/app/api/*.py`

### 7. Ownership

`git config user.name` reported `shravn95`.

The following existing candidate areas have multiple authors and therefore
cannot be treated as owner-only removal targets without further review:

```text
backend/app       Manas Doye (3), Sujal (1), SujalSakla (1), VBagekar (15)
backend/app/main.py
                  Manas Doye (2), VBagekar (2)
backend/app/api   Manas Doye (2), VBagekar (5)
backend/app/core  Manas Doye (3), Sujal (1), SujalSakla (1), VBagekar (11)
backend/app/agents
                  Manas Doye (3), Sujal (1), SujalSakla (1), VBagekar (7)
backend/requirements.txt
                  Manas Doye (1), VBagekar (1)
src               Manas Doye (3), VBagekar (7)
```

The current AutoInsight branch is `main`; no feature branch was created during
read-only Phase 0.

Evidence/commands:

```text
git -C AutoInsight config user.name
git -C AutoInsight log --format='%an' -- <path> | sort | uniq -c
git -C AutoInsight branch --show-current
```

### 8. Conventions

- Route prefix: existing backend routers use `/api`, with route paths such as
  `/upload`, `/query`, `/dataset/{dataset_id}/...`, `/forecast`, and `/health`.
- Error format: existing routes use FastAPI `HTTPException`, which produces
  FastAPI's `{"detail": "..."}` shape; streaming query errors are SSE JSON
  objects with `type`, `error_type`, and `content`.
- Config/env loading: `pydantic-settings` `BaseSettings` reads
  `backend/.env` through `backend/app/config.py`; API keys are masked by
  `Settings.__repr__`.
- Logging: standard-library `logging` configured in `main.py`; the existing
  orchestrator also emits structured query telemetry.
- Tests: pytest tests live in `backend/tests`; POC tests live in
  `ai-analyst-poc/tests` and use `pytest.ini` with `pythonpath = .`.
- Frontend tooling: Vite and oxlint scripts are in the root `package.json`.
  No backend lint configuration was found.

### 9. CI and execution surfaces

- No `.github` workflow, Dockerfile, compose file, Makefile, or backend build
  configuration was found in the repository.
- Root scripts are `run_all.bat`, `run_backend.bat`, and `run_frontend.bat`.
- Root `package.json` scripts are `dev`, `backend`, `build`, `lint`, and
  `preview`; no backend pytest script is defined.
- New files under `backend/app/ai/` would be imported only when explicitly
  registered, but backend pytest discovery may collect tests if they are placed
  under the existing test paths. No CI workflow currently indicates otherwise.

### POC read-only findings

The POC source tree contains Steps 1-6 modules under `analyst/`, four prompts,
four offline test files, `models.yaml`, and `data/samples/sales.csv` (1,000
data rows plus header). The expected synthesizer and chat loop are in
`analyst/agent/synthesizer.py` and `analyst/cli.py`.

The checked-in POC has unresolved Git conflict markers in
`analyst/config.py`, `analyst/llm.py`, `analyst/profile/datacard.py`,
`analyst/cli.py`, `analyst/prompts/planner.md`, `models.yaml`, and
`tests/test_profile.py`. This is recorded as a source-of-truth ambiguity; the
POC was not modified or “resolved” during discovery.

### Shared-file exceptions considered

No shared file has been changed in Phase 0. If D1 is approved for Path A,
router registration would require the one additive import and registration
line permitted for `backend/app/main.py`. Any dependency or env additions will
be additive and recorded here with exact lines.

## Phase 1: Removal Manifest (inventory only; no deletions)

Whole-repository searches were run with `rg` from the AutoInsight repository
root. Authorship counts are from `git log --format='%an' -- <path> | sort |
uniq -c`. No item below is approved for deletion by this manifest alone.

| Item | Type | Used by (grep evidence) | Authors | Frontend/other callers | Verdict |
|---|---|---|---|---|---|
| `backend/app/api/query.py` | route file; `POST /api/query` | Imported by `backend/app/main.py`; imports and calls `master_orchestrator.process_query_stream` | Manas Doye (1), VBagekar (1) | `src/api/client.js` calls `/query`; `Dashboard.jsx` consumes its SSE contract | KEEP-COMPAT |
| `backend/app/api/upload.py` | route file; upload/preview/download routes | Imported by `backend/app/main.py`; calls `master_orchestrator` | Manas Doye (2), VBagekar (3) | `src/api/client.js` and `Dashboard.jsx` call `/upload`, `/dataset/*/preview`, `/dataset/*/download` | KEEP-COMPAT |
| `backend/app/api/preprocess.py` | route file; `POST /api/dataset/{id}/preprocess` | Imported by `backend/app/main.py`; imports orchestrator, `nemotron_client`, `PreprocessingExecutor`, and `dataset_profiler` | VBagekar (1) | `src/api/client.js` and `Dashboard.jsx` call the route | KEEP-COMPAT |
| `backend/app/api/forecast.py` | route file; `POST /api/forecast` | Imported by `backend/app/main.py`; imports `forecasting_agent` | VBagekar (1) | `src/api/client.js` exposes `fetchForecast`; orchestrator also calls `forecasting_agent` | KEEP-COMPAT |
| `backend/app/agents/orchestrator.py` | shared service, intent parser, memory, upload/query orchestration | Imported by all legacy upload/query/preprocess routes; imports profiler, cleaner, dashboard builder, LLM client, and forecast agent; tests import `IntentParser`, `ConversationMemory`, and telemetry | Manas Doye (2), Sujal (1), VBagekar (6) | Legacy frontend contracts depend on its upload and query outputs | KEEP-SHARED |
| `backend/app/agents/forecast_agent.py` | service/agent | Imported by `api/forecast.py` and `orchestrator.py` | Manas Doye (1), VBagekar (1) | Forecast is exposed to the frontend and used in upload/dashboard results | KEEP-SHARED |
| `backend/app/core/llm_client.py` | shared LLM client and prompt/action/chart/report logic | Imported by `orchestrator.py` and `api/preprocess.py`; module singleton `nemotron_client` is used in both | Manas Doye (2), Sujal (1), VBagekar (2) | Supports legacy query, upload dashboard, and preprocessing behavior | KEEP-SHARED |
| `backend/app/core/dataset_profiler.py` | shared profiler | Imported by `orchestrator.py` and `api/preprocess.py`; singleton used in both | Manas Doye (2), VBagekar (5) | Its summaries are returned to the frontend and used by dashboard/preprocess flows | KEEP-SHARED |
| `backend/app/core/data_cleaner.py` | shared upload cleaning | Imported by `orchestrator.py` | Manas Doye (1), VBagekar (3) | Feeds the legacy upload response and dataset state | KEEP-SHARED |
| `backend/app/core/dashboard_builder.py` | shared dashboard computation | Imported by `orchestrator.py` | Manas Doye (2), Sujal (1), VBagekar (1) | Its chart payload is consumed by `Dashboard.jsx` | KEEP-SHARED |
| `backend/app/core/preprocessing_actions.py` | shared preprocessing executor | Imported by `api/preprocess.py` | VBagekar (2) | Route response is consumed by `Dashboard.jsx` | KEEP-SHARED |
| `backend/app/config.py` | shared settings/config | Imported by `main.py`, tests, and core/agent code through application settings | Manas Doye (2), Sujal (1), VBagekar (1) | Supplies current health/config behavior and upload limit | KEEP-SHARED |
| `backend/app/main.py` | app entry point/router registration | Imports and mounts every existing router; owns `/api/health` and CORS/lifespan | Manas Doye (2), VBagekar (2) | All frontend API calls enter through this app | KEEP-SHARED (R4 exception) |
| `backend/tests/test_llm_intent.py` | test file | Imports `IntentParser` from orchestrator | Manas Doye (1) | No direct frontend caller; protects shared legacy behavior | KEEP-SHARED |
| `backend/tests/test_person2.py` | test file | Imports `IntentParser`, `ConversationMemory` from orchestrator | Manas Doye (1) | No direct frontend caller; protects shared legacy behavior | KEEP-SHARED |
| `backend/tests/test_person2_exhaustive.py` | test file | Imports orchestrator, `NemotronLLMClient`, dashboard builder, config | Manas Doye (1) | No direct frontend caller; protects shared legacy behavior | KEEP-SHARED |
| `backend/requirements.txt` | dependency manifest | Runtime imports include FastAPI, pandas, openpyxl, DuckDB, OpenAI, and scientific packages | Manas Doye (1), VBagekar (1) | Shared backend dependency surface | KEEP-SHARED (R4 exception) |
| `backend/.env.example` | env example | Declares NVIDIA settings consumed by `backend/app/config.py` | Manas Doye (1), VBagekar (1) | Documents current app setup; key is used by existing LLM features | KEEP-SHARED (R4 exception) |
| `backend/app/agents/viz_planner.py` | historical file | `git log --all` shows it was already deleted; no current file or current import | historical authorship only | No current caller found | KEEP-SHARED / no action |

### Phase 1 conclusion

Every current route is frontend-called and therefore KEEP-COMPAT. Every
service/core/config/test/manifest surface is shared or has multiple authors.
There are no safe `REMOVE` items at this point. Any later removal would first
need an owner-approved compatibility adapter and a new whole-repository
reference check; D2 is still required before Phase 5.

## Decisions

- **D1 approved:** Path A, with `<AI_DIR>` `backend/app/ai/` and package
  `ai_analyst`.
- **D2 approved:** `APPROVE REMOVAL`; no items are authorized for deletion
  because the manifest contains no `REMOVE` verdicts.
- **Authentication exception approved by owner:** the repository has no auth
  dependency. Until real auth is supplied outside this task, Phase 2 may use a
  clearly documented, module-local development identity mechanism, and the
  production auth blocker must remain explicit in this file and the module
  README. This mechanism must not be presented as production authentication.
- **POC parity conflict resolution approved by owner:** use the `HEAD`
  sections of the committed POC conflict markers (NVIDIA provider setup,
  day-first date expectation, and full chat loop) when copying behavior into
  new files. The POC itself remains unmodified.

## Phase 2: AI Analyst integration

The approved Path A package is `backend/app/ai/ai_analyst/`. It ports POC Steps
1–6 and adds the HTTP-safe `AnalystService`, explicit `session_id`, owner
metadata, 24-hour TTL cleanup, per-session locks, upload allowlist and 25 MiB
limit, read-only DuckDB SQL checks (1,000-row library cap, with API responses
capped at 200 rows, and timeout), typed errors,
rate limiting, lazy missing-key handling, telemetry, and privacy metadata.

Routes are mounted under `/api/ai/`; legacy routes are unchanged. The
owner dependency is currently a clearly temporary `development_identity`
(`AI_ANALYST_DEV_USER`) because this repository has no auth. It is not
production authentication: production deployment is blocked until it is
replaced with the host's authenticated-user dependency. No verifier/evaluation
or removal phase is included.

Exact Phase 2 environment defaults are `AI_SESSION_TTL_HOURS=24`,
`AI_MAX_UPLOAD_MB=25`, `AI_LLM_TIMEOUT_S=60`,
`AI_SEND_SAMPLE_ROWS=true`, and `AI_SEND_TOP_VALUES=true`; request limiting
defaults to 20 per minute. Privacy flags are applied to every LLM prompt,
while the stored card remains complete. Prompt data blocks include an
untrusted-data injection warning. Telemetry contains event metadata only.

## Phase 4 Step 8: evaluation tooling

Added the dev-only `ai_analyst/eval/` harness. It has 41 deterministic cases
covering all five intents plus aggregation, join, date, null, casing, and
adversarial behavior. Seeded fixtures include join/null-heavy `sales`,
`customers`, and `targets` tables. Comparisons support scalar/table numeric
tolerance, order and column mismatch diagnostics. Metrics include exact match,
execution success, numeric accuracy, verifier catch rate, and mean latency.
`eval run` supports `--model-set`, `--limit`, `--dry-run`, and JSONL resume state,
and emits JSON and Markdown reports without real keys.

The corrective follow-up adds YAML case catalogues with explicit dataset and
expectation schemas, 15 incorrect plus 15 correct verifier pairs, seeded CSV
generation (42), A/B/C configuration splits, full overall/category metrics,
rate-limit-safe failure records, and the standalone `python -m
app.ai.ai_analyst.eval run|report` entry point. `backend/app/ai/docs/EVAL_REPORT.md`
records that real model numbers remain pending until a key is available.

## Phase 5: safe compatibility outcome

Phase 5 performed the safe portion only. The Phase 1 manifest has no `REMOVE`
verdicts: all legacy frontend endpoints remain `KEEP-COMPAT`, and shared
routes/services/core files remain `KEEP-SHARED`. A whole-repository reference
search was run before considering deletion and found active frontend callers in
`src/api/client.js` and `src/components/Dashboard.jsx`; therefore **zero
files and zero endpoints were removed**.

`backend/app/ai/FRONTEND_NOTES.md` records the exact request/response contracts,
the current callers, and why `AnalystService` cannot safely preserve the
dashboard, mutable dataset, preprocessing, forecast, health, or legacy SSE
shapes. `ai_analyst/legacy_contracts.json` and
`tests/test_phase5_compat.py` provide a no-key contract/reference check that
legacy routes remain registered and frontend callers remain untouched. No
adapter was added where the shape cannot be preserved. Phase 6 is not
implemented.
