# Project Log

**Project:** UI (React + Vite frontend, Python FastAPI backend)
**Date Started:** 2026-08-10

## Initial Repository Snapshot (2026-08-10)

### Frontend (React + Vite)
- `package.json` – project metadata & scripts
- `vite.config.js` – Vite configuration
- `index.html` – entry HTML
- `src/main.jsx` – React bootstrap
- `src/App.jsx` – root component
- `src/App.css` / `src/index.css` – global styles
- Components:
  - `src/components/LandingPage.jsx` + `LandingPage.css`
  - `src/components/Dashboard.jsx` + `Dashboard.css`
  - `src/components/HeroDashboardShowcase.jsx` + `HeroDashboardShowcase.css`
  - `src/components/ChartComponents.jsx`
- Assets: `src/assets/react.svg`, `hero.png`, `vite.svg`
- API client: `src/api/client.js`
- Mock data: `src/data/mockData.js`
- Scripts: `run_frontend.bat`, `run_all.bat`

### Backend (FastAPI)
- `backend/requirements.txt` – Python dependencies
- `backend/.env` & `.env.example` – environment config
- App entry: `backend/app/main.py`
- Config: `backend/app/config.py`
- Core modules:
  - `backend/app/core/llm_client.py`
  - `backend/app/core/data_cleaner.py`
  - `backend/app/core/dataset_profiler.py`
  - `backend/app/core/dashboard_builder.py`
- API routes:
  - `backend/app/api/upload.py`
  - `backend/app/api/query.py`
  - `backend/app/api/forecast.py`
- Agents:
  - `backend/app/agents/orchestrator.py`
  - `backend/app/agents/viz_planner.py`
  - `backend/app/agents/forecast_agent.py`
- Scripts: `run_backend.bat`

### Misc
- `README.md` – project overview
- `setup-and-person1-tasks.md` – setup notes / task list
- `.gitignore`, `.oxlintrc.json`

---

## Log Entries

| Date | Action | Details |
|------|--------|---------|
| 2026-08-10 | Refactor data cleaner with column‑aware strategies | Implemented median → forward‑fill for numeric time‑series columns, skip imputation & flag categorical columns with >40 % missing, winsorize outliers at 1st/99th percentile (optional treat_outliers flag), and extended cleaning report with imputation_strategy_used, high_missing_flagged, and outlier_treatment per column. |
| 2026-08-10 | Person 1 – Task 2: confidence‑scored KPI/geo detection | Added configurable kpi_keywords / geo_keywords parameters to profile_csv(). KPI detection now combines keyword match (0.6) and statistical signal (unique_ratio < 0.95 and not a date column) (0.4) into a kpi_confidence (0–1) per numeric column; both signals give 1.0. Geo detection keeps keyword match = 1.0, plus fallback pattern match (2/3‑letter codes or common country names) = 0.5. Each column's col_meta now includes kpi_confidence (numeric) or geo_confidence (categorical) alongside semantic_role. Date detection unchanged. |
| 2026-08-10 | Person 1 – Task 3: structured cleaning summary + data quality breakdown contract | Extended profile_csv() with optional `cleaning_report` parameter. When supplied, a `cleaning_summary` object is added containing rows_before, rows_after, duplicates_removed, plain‑English imputation_details, high_missing_flagged, and outlier_treatment_details. Also added `quality_score_breakdown` with three sub‑scores (0‑100): completeness = 100 – average missing %; consistency = 100 × (1 – fraction of numeric columns with outlier treatment); type_confidence = average of per‑column kpi_confidence / geo_confidence / date‑confidence (date = 1.0). Existing overall quality_score retained. No changes to data_cleaner.py, upload.py, KPI/geo or date detection logic. |
| 2026-08-10 | Person 1 – Task 4: wire cleaning_report into profiler call | The orchestrator already produced a `cleaning_report` but never passed it to `profile_csv()`. Fixed by adding `cleaning_report=cleaning_report` keyword argument on the single profiler call (line 35). This makes the new `cleaning_summary` and `quality_score_breakdown` appear in the API response. No other logic changed. |
| 2026-08-11 | Person 1 – Task 5: multi-sheet Excel support | **Found:** `orchestrator.py` used `pd.read_excel(BytesIO(file_bytes))` which only reads the first sheet silently; no sheet detection or selection was exposed. **Added:** (1) `pd.ExcelFile(...).sheet_names` to detect all sheets; (2) optional `sheet_name` query parameter on `POST /api/upload` (default `None` → first sheet); (3) `available_sheets: List[str]` in upload response for Excel files (empty list for CSV); (4) `sheet_name` stored in `self.datasets[dataset_id]` so downstream agents know which sheet was loaded. CSV handling unchanged. Verified with synthetic 3-sheet `.xlsx`: default loads Sheet1, `?sheet_name=Sheet2` loads Sheet2, `?sheet_name=Sheet3` loads Sheet3; all responses include `available_sheets: ["Sheet1","Sheet2","Sheet3"]`. |
| 2026-08-11 | Person 1 – Task 6: PII/sensitive-column detection with optional masking | Added heuristic PII detection to `profile_csv()` in `dataset_profiler.py`. Detection uses two signals: (1) column name patterns (regex for email, phone, ssn, credit_card, passport, driver_license, address, name, dob, ip_address) with base confidences 0.6–0.95; (2) value-pattern checks on up to 50 sampled rows per column (regex for email format, phone formats, SSN `ddd-dd-dddd`, credit card 13–16 digits with optional spaces/dashes, IPv4). Combined confidence = max(name_conf, value_conf × min(1, match_ratio×2)). Results returned in new `pii_flags: List[{column, reason, confidence}]` (confidence 0.0–1.0). Optional `mask_pii: bool = False` parameter: when True, columns with confidence ≥ 0.7 get a `masked_preview` dict in response with 5 sampled values cosmetically masked (e.g., `j***@***.com`, `***-***-1234`, `***-**-6789`, `**** **** **** 1111`, `***.***.***.1`). Masking is preview-only; original DataFrame is untouched for cleaning/charts/downstream use. No changes to data_cleaner.py, upload.py, or orchestrator.py. |
*Add future entries below this line.*