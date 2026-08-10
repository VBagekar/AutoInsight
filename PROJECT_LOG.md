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
*Add future entries below this line.*