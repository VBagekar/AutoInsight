import pandas as pd
import numpy as np
import io
from typing import Dict, Any, List

class LocalDatasetProfiler:
    def __init__(self):
        pass

    def profile_csv(self, file_contents: bytes, filename: str, dataframe: pd.DataFrame | None = None) -> Dict[str, Any]:
        """
        Parses a CSV dataset locally using Pandas/NumPy, extracting a rich, 
        compact JSON summary to send to NVIDIA Nemotron without sending raw files.
        """
        df = dataframe.copy() if dataframe is not None else pd.read_csv(io.BytesIO(file_contents))
        
        row_count, col_count = df.shape
        columns_info: List[Dict[str, Any]] = []
        numeric_cols = []
        date_cols = []
        categorical_cols = []
        detected_kpis = []

        # Keywords for business KPI detection
        kpi_keywords = ['revenue', 'sales', 'profit', 'amount', 'price', 'cost', 'orders', 'users', 'margin', 'clv', 'quantity']
        geo_keywords = ['city', 'state', 'country', 'region', 'zip', 'lat', 'lon', 'location']

        for col in df.columns:
            col_lower = str(col).lower()
            dtype_str = str(df[col].dtype)
            missing_pct = round(float(df[col].isnull().mean() * 100), 2)
            unique_count = int(df[col].nunique())
            sample_vals = df[col].dropna().unique()[:3].tolist()
            
            # Format sample values for JSON compatibility
            sample_vals_clean = [str(v) for v in sample_vals]

            col_meta = {
                "name": str(col),
                "type": dtype_str,
                "missing_pct": missing_pct,
                "unique_count": unique_count,
                "sample_values": sample_vals_clean
            }

            # Date detection.  We infer real datetime-like columns too, not just names.
            parsed_dates = None
            if not pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
                parsed_dates = pd.to_datetime(df[col], errors='coerce')
            date_like = parsed_dates is not None and float(parsed_dates.notna().mean()) >= 0.8
            if 'date' in col_lower or 'time' in col_lower or 'year' in col_lower or 'month' in col_lower or date_like:
                date_cols.append(str(col))
                col_meta["semantic_role"] = "date"
            # Numeric analysis
            elif pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(str(col))
                col_meta["mean"] = float(df[col].mean()) if not df[col].isnull().all() else 0.0
                col_meta["std"] = float(df[col].std()) if not df[col].isnull().all() else 0.0
                col_meta["min"] = float(df[col].min()) if not df[col].isnull().all() else 0.0
                col_meta["max"] = float(df[col].max()) if not df[col].isnull().all() else 0.0
                col_meta["median"] = float(df[col].median()) if not df[col].isnull().all() else 0.0

                if any(kw in col_lower for kw in kpi_keywords):
                    detected_kpis.append(str(col))
                    col_meta["semantic_role"] = "kpi"
            elif any(kw in col_lower for kw in geo_keywords):
                categorical_cols.append(str(col))
                col_meta["semantic_role"] = "geographical"
            else:
                categorical_cols.append(str(col))
                col_meta["semantic_role"] = "dimension"

            columns_info.append(col_meta)

        # Correlation Matrix for Numeric Columns
        correlation_matrix = {}
        if len(numeric_cols) > 1:
            try:
                corr_df = df[numeric_cols].corr().fillna(0).round(2)
                correlation_matrix = corr_df.to_dict()
            except Exception:
                pass

        # Extract Aggregated Series for Actual Chart Rendering
        extracted_chart_data = {}
        primary_kpi = detected_kpis[0] if detected_kpis else (numeric_cols[0] if numeric_cols else None)
        
        if date_cols and primary_kpi:
            try:
                date_col = date_cols[0]
                df_temp = df.copy()
                df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                df_temp = df_temp.dropna(subset=[date_col])
                df_temp['period'] = df_temp[date_col].dt.strftime('%b %Y')
                ts_grouped = df_temp.groupby('period', sort=False)[primary_kpi].sum().reset_index()
                extracted_chart_data["time_series"] = [
                    {"name": str(r['period']), "value": round(float(r[primary_kpi]), 2), primary_kpi: round(float(r[primary_kpi]), 2)}
                    for _, r in ts_grouped.head(12).iterrows()
                ]
            except Exception:
                pass

        if categorical_cols and primary_kpi:
            try:
                cat_col = categorical_cols[0]
                cat_grouped = df.groupby(cat_col)[primary_kpi].sum().reset_index().sort_values(by=primary_kpi, ascending=False).head(8)
                extracted_chart_data["categorical"] = [
                    {"name": str(r[cat_col]), "value": round(float(r[primary_kpi]), 2), "count": round(float(r[primary_kpi]), 2)}
                    for _, r in cat_grouped.iterrows()
                ]
            except Exception:
                pass

        if len(numeric_cols) >= 2:
            try:
                col1, col2 = numeric_cols[0], numeric_cols[1]
                sample_scatter = df[[col1, col2]].dropna().head(30)
                extracted_chart_data["scatter"] = [
                    {"x": round(float(r[col1]), 2), "y": round(float(r[col2]), 2), "z": 100}
                    for _, r in sample_scatter.iterrows()
                ]
            except Exception:
                pass

        # High-level Summary object
        summary = {
            "filename": filename,
            "row_count": row_count,
            "column_count": col_count,
            "columns": columns_info,
            "numeric_columns": numeric_cols,
            "date_columns": date_cols,
            "categorical_columns": categorical_cols,
            "detected_kpis": detected_kpis if detected_kpis else numeric_cols[:4],
            "primary_kpi": primary_kpi,
            "correlation_matrix": correlation_matrix,
            "extracted_chart_data": extracted_chart_data,
            "quality_score": round(float((1 - df.isnull().mean().mean()) * 100), 1)
        }

        return summary

dataset_profiler = LocalDatasetProfiler()
