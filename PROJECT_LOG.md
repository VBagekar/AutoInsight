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
| 2026-08-10 | Created project log file | Initial snapshot recorded (this file). |

*Add future entries below this line.*