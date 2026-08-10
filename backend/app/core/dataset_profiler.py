import pandas as pd
import numpy as np
import io
import re
from typing import Dict, Any, List, Optional

class LocalDatasetProfiler:
    def __init__(self):
        pass

    def profile_csv(self, file_contents: bytes, filename: str, dataframe: pd.DataFrame | None = None,
                    kpi_keywords: Optional[List[str]] = None,
                    geo_keywords: Optional[List[str]] = None,
                    cleaning_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

        # Default keyword lists
        default_kpi_keywords = ['revenue', 'sales', 'profit', 'amount', 'price', 'cost', 'orders', 'users', 'margin', 'clv', 'quantity']
        default_geo_keywords = ['city', 'state', 'country', 'region', 'zip', 'lat', 'lon', 'location']
        kpi_kw = [kw.lower() for kw in (kpi_keywords or default_kpi_keywords)]
        geo_kw = [kw.lower() for kw in (geo_keywords or default_geo_keywords)]

        # Pre-detect date columns (used for KPI statistical signal)
        for col in df.columns:
            col_lower = str(col).lower()
            parsed_dates = None
            if not pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
                parsed_dates = pd.to_datetime(df[col], errors='coerce')
            date_like = parsed_dates is not None and float(parsed_dates.notna().mean()) >= 0.8
            if 'date' in col_lower or 'time' in col_lower or 'year' in col_lower or 'month' in col_lower or date_like:
                date_cols.append(str(col))

        for col in df.columns:
            col_lower = str(col).lower()
            dtype_str = str(df[col].dtype)
            missing_pct = round(float(df[col].isnull().mean() * 100), 2)
            unique_count = int(df[col].nunique())
            sample_vals = df[col].dropna().unique()[:3].tolist()
            sample_vals_clean = [str(v) for v in sample_vals]

            col_meta = {
                "name": str(col),
                "type": dtype_str,
                "missing_pct": missing_pct,
                "unique_count": unique_count,
                "sample_values": sample_vals_clean
            }

            # Date detection (already collected)
            if str(col) in date_cols:
                col_meta["semantic_role"] = "date"
            # Numeric analysis
            elif pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(str(col))
                col_meta["mean"] = float(df[col].mean()) if not df[col].isnull().all() else 0.0
                col_meta["std"] = float(df[col].std()) if not df[col].isnull().all() else 0.0
                col_meta["min"] = float(df[col].min()) if not df[col].isnull().all() else 0.0
                col_meta["max"] = float(df[col].max()) if not df[col].isnull().all() else 0.0
                col_meta["median"] = float(df[col].median()) if not df[col].isnull().all() else 0.0

                # KPI confidence scoring
                keyword_match = any(kw in col_lower for kw in kpi_kw)
                # statistical signal: not ID-like (unique ratio not ~1) and not a date column
                unique_ratio = unique_count / row_count if row_count else 1.0
                statistical_signal = (unique_ratio < 0.95) and (str(col) not in date_cols)

                if keyword_match and statistical_signal:
                    kpi_conf = 1.0
                elif keyword_match:
                    kpi_conf = 0.6
                elif statistical_signal:
                    kpi_conf = 0.4
                else:
                    kpi_conf = 0.0

                col_meta["kpi_confidence"] = round(kpi_conf, 2)
                col_meta["semantic_role"] = "kpi" if kpi_conf > 0.0 else "measure"
                if kpi_conf > 0.0:
                    detected_kpis.append((str(col), kpi_conf))
            # Categorical / geo analysis
            elif any(kw in col_lower for kw in geo_kw):
                categorical_cols.append(str(col))
                col_meta["semantic_role"] = "geographical"
                col_meta["geo_confidence"] = 1.0
            else:
                # fallback pattern detection for geo
                pattern_match = False
                if df[col].dtype == object:
                    # Check first few non-null unique values
                    for val in sample_vals_clean:
                        if re.fullmatch(r'[A-Z]{2}|[A-Z]{3}', val):
                            pattern_match = True
                            break
                        # simple country name list (lowercase)
                        if val.lower() in {"united states", "usa", "canada", "mexico", "uk", "germany", "france", "india", "china", "japan", "australia", "brazil"}:
                            pattern_match = True
                            break
                if pattern_match:
                    categorical_cols.append(str(col))
                    col_meta["semantic_role"] = "geographical"
                    col_meta["geo_confidence"] = 0.5
                else:
                    categorical_cols.append(str(col))
                    col_meta["semantic_role"] = "dimension"
                    col_meta["geo_confidence"] = 0.0

            columns_info.append(col_meta)

        # Build detected_kpis list ordered by confidence descending
        if detected_kpis:
            detected_kpis.sort(key=lambda x: x[1], reverse=True)
            detected_kpis = [col for col, _ in detected_kpis]
        else:
            # fallback to first few numeric columns
            detected_kpis = numeric_cols[:4]

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
            "detected_kpis": detected_kpis,
            "primary_kpi": primary_kpi,
            "correlation_matrix": correlation_matrix,
            "extracted_chart_data": extracted_chart_data,
            "quality_score": round(float((1 - df.isnull().mean().mean()) * 100), 1)
        }

        # If a cleaning report is supplied, attach a human‑readable cleaning summary
        # and a detailed quality‑score breakdown.
        if cleaning_report is not None:
            # ---- cleaning_summary ----
            missing_handled = cleaning_report.get("missing_values_handled", {})
            imputation_used = cleaning_report.get("imputation_strategy_used", {})
            high_missing = cleaning_report.get("high_missing_flagged", [])
            outlier_treat = cleaning_report.get("outlier_treatment", {})

            imputation_details = []
            for col, miss_cnt in missing_handled.items():
                if col in high_missing:
                    pct = round(miss_cnt / cleaning_report["initial_rows"] * 100, 1)
                    imputation_details.append(f"{col}: {pct}% missing — left as-is, flagged for review")
                else:
                    strategy = imputation_used.get(col, "median")
                    friendly = {
                        "forward_fill": "forward-filled",
                        "median": "median-imputed",
                        "mode": "mode-imputed",
                        "skipped_high_missing": "left as-is, flagged for review"
                    }.get(strategy, strategy)
                    imputation_details.append(f"{col}: {friendly} ({miss_cnt} missing values)")

            high_missing_details = [
                f"{col}: {round(cleaning_report['missing_values_handled'][col] / cleaning_report['initial_rows'] * 100, 1)}% missing — left as-is, flagged for review"
                for col in high_missing
            ]

            outlier_details = []
            for col, info in outlier_treat.items():
                cnt = info.get("count", 0)
                action = info.get("action", "none")
                if cnt and action != "none":
                    if action == "winsorized_1_99":
                        outlier_details.append(f"{col}: {cnt} outliers capped at 1st/99th percentile")
                    else:
                        outlier_details.append(f"{col}: {cnt} outliers treated ({action})")

            cleaning_summary = {
                "rows_before": cleaning_report.get("initial_rows"),
                "rows_after": cleaning_report.get("cleaned_rows"),
                "duplicates_removed": cleaning_report.get("duplicates_removed"),
                "imputation_details": imputation_details,
                "high_missing_flagged": high_missing_details,
                "outlier_treatment_details": outlier_details
            }
            summary["cleaning_summary"] = cleaning_summary

            # ---- quality_score_breakdown ----
            # completeness: 100 - average missing %
            avg_missing = sum(col["missing_pct"] for col in columns_info) / len(columns_info) if columns_info else 0
            completeness = max(0.0, 100.0 - avg_missing)

            # consistency: proportion of numeric columns without outlier treatment
            treated_cols = sum(1 for info in outlier_treat.values() if info.get("action") != "none" and info.get("count", 0) > 0)
            total_numeric = len(numeric_cols) if numeric_cols else 1
            consistency = max(0.0, 100.0 * (1.0 - treated_cols / total_numeric))

            # type_confidence: average of kpi_confidence, geo_confidence, date confidence (1.0)
            conf_vals = []
            for col_meta in columns_info:
                if "kpi_confidence" in col_meta:
                    conf_vals.append(col_meta["kpi_confidence"])
                elif "geo_confidence" in col_meta:
                    conf_vals.append(col_meta["geo_confidence"])
                elif col_meta.get("semantic_role") == "date":
                    conf_vals.append(1.0)
                else:
                    conf_vals.append(0.0)
            type_confidence = (sum(conf_vals) / len(conf_vals) * 100) if conf_vals else 0.0

            summary["quality_score_breakdown"] = {
                "completeness": round(completeness, 1),
                "consistency": round(consistency, 1),
                "type_confidence": round(type_confidence, 1)
            }

        return summary

dataset_profiler = LocalDatasetProfiler()
