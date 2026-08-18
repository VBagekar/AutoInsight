# AutoInsight Analytics

Upload a CSV or Excel dataset, then ask a question in plain language. Nexus cleans and profiles the data locally, computes the chart series locally, and renders an interactive dashboard from those real values.

## What works

- CSV and Excel upload with duplicate removal and missing-value treatment
- Data profiling: schema, date/dimension/KPI detection, correlations, and quality score
- Data-backed initial dashboard (trend, category breakdown, composition, and correlation where applicable)
- Query-driven dashboard updates, tooltips, detailed report, and local forecast
- Optional NVIDIA Nemotron-3 Super 120B planner. It receives only compact schema metadata, not the uploaded file or raw rows; all values displayed in charts remain locally computed and field-validated.

## Run locally

1. Install backend packages:

   ```powershell
   python -m pip install -r backend/requirements.txt
   ```

2. Optional: copy `backend/.env.example` to `backend/.env` and set `NVIDIA_API_KEY`. Without a key, the deterministic local planner remains fully functional.

3. In one terminal start the API:

   ```powershell
   python backend/app/main.py
   ```

4. In another terminal start the frontend:

   ```powershell
   npm install
   npm run dev
   ```

Open the Vite URL displayed in the terminal, then use **Sample CSV** or upload your own file.

## Deployment note

Dataset state is held in application memory for this local single-user deliverable. Before a multi-user/cloud deployment, move uploaded data to encrypted object storage and persist dataset metadata/session ownership in a database.
