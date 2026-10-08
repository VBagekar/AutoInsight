# Phase 5 frontend compatibility notes

Phase 1 classified every existing frontend-facing endpoint as `KEEP-COMPAT`.
Phase 5 is therefore documentation and verification only: it performs **zero
removals**, does not add an adapter, and does not change `src/` or any legacy
route/service/core file.

The callers are `src/api/client.js` and `src/components/Dashboard.jsx`. Their
current request and response contracts are captured below and in the
machine-readable `ai_analyst/legacy_contracts.json` inventory.

## KEEP-COMPAT endpoint contracts

### `POST /api/upload`

* **Caller/request:** `uploadDataset(file)` sends multipart field `file`
  (CSV/XLS/XLSX). The optional `sheet_name` query parameter is supported by
  the route, although the current frontend does not send it.
* **Response:** JSON object with `status`, `dataset_id`, `dashboard_title`,
  `summary`, `cleaning_report`, `charts`, `kpi_summary`, `forecast`,
  `ai_insights`, `available_sheets`, and `plan_source`. `summary`, chart,
  forecast, and KPI values are consumed directly by `Dashboard.jsx`.
* **AnalystService:** cannot preserve this contract. It stores an isolated
  session/card and deliberately does not create the legacy in-memory
  orchestrator dataset, dashboard plan, or forecast payload.

### `POST /api/query`

* **Caller/request:** `streamAIQuery(query, datasetId, ...)` sends JSON
  `{query: string, dataset_id: string}`.
* **Response:** `text/event-stream`; each event is `data: <JSON>\n\n`.
  Thinking events are `{type: "thinking", content: string}`, successful
  results are `{type: "payload", data: <legacy dashboard payload>}`, and
  failures are `{type: "error", error_type: string, content: string}`.
* **AnalystService:** cannot preserve the dashboard/SSE payload contract.
  Its `/api/ai/sessions/{id}/chat` endpoint returns an analyst answer and
  table-oriented evidence, not legacy chart updates. No shape-changing
  adapter is safe, so none is performed and this route remains registered.

### `GET /api/dataset/{dataset_id}/preview`

* **Caller/request:** `fetchDatasetPreview(id, page, pageSize)` sends
  `page` (default 1) and `page_size` (default 50, maximum 500).
* **Response:** JSON `{columns: string[], rows: object[], total_rows: number,
  page: number, page_size: number, total_pages: number,
  cleaning_summary: object|null}`.
* **AnalystService:** cannot preserve legacy dataset IDs or the mutable
  cleaned-DataFrame view. The AI card is a different privacy and storage
  contract; no adapter or removal is performed.

### `GET /api/dataset/{dataset_id}/download`

* **Caller/request:** `downloadDataset(id)` sends no body.
* **Response:** CSV bytes (`text/csv`) with
  `Content-Disposition: attachment; filename="<basename>_cleaned.csv"` and
  exposed `Content-Disposition` header.
* **AnalystService:** cannot preserve the legacy cleaned/preprocessed export
  because its session database is not the legacy DataFrame. No adapter or
  removal is performed.

### `POST /api/dataset/{dataset_id}/preprocess`

* **Caller/request:** `preprocessDataset(id, command)` sends JSON
  `{command: string}`.
* **Response:** JSON `{explanation: string, confirmation_message: string,
  updated_summary: object}`. The route performs one of the six existing
  preprocessing actions and mutates the legacy dataset.
* **AnalystService:** cannot preserve mutation/action execution or
  `updated_summary` semantics. The analyst module is read-only after ingest;
  no adapter or removal is performed.

### `POST /api/forecast`

* **Caller/request:** `fetchForecast(historicalValues, periods)` sends JSON
  `{historical_values: number[], periods: number}`.
* **Response:** forecast JSON with `forecast_type`, `historical_mean`,
  `projected_growth_rate`, `trend_direction`, `forecast_points` (each point
  has `period`, `value`, `lower_bound`, `upper_bound`), and `ai_summary`;
  it may be `null` for fewer than two usable values.
* **AnalystService:** does not implement forecasting and cannot preserve this
  response. No adapter or removal is performed.

### `GET /api/health`

* **Caller/request:** `fetchHealth()` sends no body.
* **Response:** JSON `{llm_configured: boolean, llm_reachable: boolean,
  model: string}`. On a non-OK response or network failure the frontend
  supplies all-false/empty fallback values.
* **AnalystService:** its `/api/ai/health` reports analyst-module status and
  is not a substitute for the host LLM health contract. No adapter or
  removal is performed.

The root endpoint and all routes above remain mounted by `backend/app/main.py`.
The compatibility test checks the route registrations and the unchanged
legacy caller strings without requiring an LLM key.
