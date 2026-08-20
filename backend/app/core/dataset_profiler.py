import pandas as pd
import numpy as np
import io
import re
from typing import Dict, Any, List, Optional

class LocalDatasetProfiler:
    def __init__(self):
        self._pii_name_patterns = [
            (r'(email|e_mail|mail)', 'email', 0.9),
            (r'(phone|tel|mobile|cell)', 'phone', 0.9),
            (r'(ssn|social_security|socialsecurity)', 'ssn', 0.95),
            (r'(credit_card|creditcard|cc_num|card_number|cc\b|card\b)', 'credit_card', 0.9),
            (r'(passport|passport_num)', 'passport', 0.85),
            (r'(driver_license|driverlicense|dl_num)', 'driver_license', 0.85),
            (r'(address|street|zip|postal)', 'address', 0.7),
            (r'(name|first_name|last_name|full_name|fname|lname)', 'name', 0.6),
            (r'(dob|date_of_birth|birth)', 'dob', 0.8),
            (r'(ip_address|ip\b)', 'ip_address', 0.7),
        ]

        self._pii_value_patterns = [
            (re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'), 'email', 0.95),
            (re.compile(r'^(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$'), 'phone', 0.9),
            (re.compile(r'^\d{3}-\d{2}-\d{4}$'), 'ssn', 0.95),
            (re.compile(r'^(\d{4}[- ]?){3}\d{4}$|^\d{13,16}$'), 'credit_card', 0.85),
            (re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'), 'ip_address', 0.8),
        ]

    def _detect_pii(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        pii_flags = []
        for col in df.columns:
            col_lower = str(col).lower()
            max_confidence = 0.0
            reasons = []

            for pattern, ptype, conf in self._pii_name_patterns:
                if re.search(pattern, col_lower):
                    if conf > max_confidence:
                        max_confidence = conf
                    reasons.append(f"column name matches {ptype} pattern")

            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                sample_vals = df[col].dropna().astype(str).head(50)
                for regex, ptype, conf in self._pii_value_patterns:
                    matches = sample_vals.apply(lambda x: bool(regex.match(x))).sum()
                    if matches > 0:
                        match_ratio = matches / len(sample_vals)
                        val_conf = conf * min(1.0, match_ratio * 2)
                        if val_conf > max_confidence:
                            max_confidence = val_conf
                        reasons.append(f"{matches}/{len(sample_vals)} values match {ptype} format")

            if max_confidence > 0.0:
                pii_flags.append({
                    "column": str(col),
                    "reason": "; ".join(reasons),
                    "confidence": round(max_confidence, 2)
                })
        return pii_flags

    def _mask_value(self, value: str, ptype: str) -> str:
        if ptype == 'email':
            parts = value.split('@')
            if len(parts) == 2:
                return f"{parts[0][0]}***@***.{parts[1].split('.')[-1]}"
            return "***@***.***"
        elif ptype == 'phone':
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 4:
                return f"***-***-{digits[-4:]}"
            return "***-***-****"
        elif ptype == 'ssn':
            return "***-**-" + value[-4:] if len(value) >= 4 else "***-**-****"
        elif ptype == 'credit_card':
            return "**** **** **** " + value[-4:] if len(value) >= 4 else "**** **** **** ****"
        elif ptype == 'ip_address':
            parts = value.split('.')
            return f"***.***.***.{parts[-1]}" if len(parts) == 4 else "***.***.***.***"
        else:
            return value[0] + "***" if len(value) > 0 else "***"

    def _get_pii_type(self, reasons: str) -> str:
        for ptype in ['email', 'phone', 'ssn', 'credit_card', 'ip_address', 'passport', 'driver_license', 'address', 'name', 'dob']:
            if ptype in reasons.lower():
                return ptype
        return 'unknown'

    def profile_csv(self, file_contents: bytes, filename: str, dataframe: pd.DataFrame | None = None,
                    kpi_keywords: Optional[List[str]] = None,
                    geo_keywords: Optional[List[str]] = None,
                    cleaning_report: Optional[Dict[str, Any]] = None,
                    mask_pii: bool = False,
                    was_sampled: bool = False,
                    sample_size: int = 0,
                    total_rows: int = 0) -> Dict[str, Any]:
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

        # Pre-detect date columns (fast sampled check)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(str(col))
            elif not pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
                col_lower = str(col).lower()
                is_date_name = any(kw in col_lower for kw in ["date", "time", "year", "month", "day", "timestamp", "period"])
                sample = df[col].dropna().head(20)
                try:
                    parsed_dates = pd.to_datetime(sample, errors='coerce', format='mixed')
                    if parsed_dates.notna().mean() >= 0.8 or (is_date_name and parsed_dates.notna().mean() >= 0.5):
                        date_cols.append(str(col))
                except Exception:
                    pass

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
                is_id_like = bool(re.search(r'(^|_)(id|key|code|zip|postal|row|index|num|number)($|_)', col_lower))
                is_discrete_date = bool(re.search(r'(^|_)(year|yr|month|day|quarter|qtr|week|hour|minute)($|_)', col_lower))

                keyword_match = any(kw in col_lower for kw in kpi_kw)
                unique_ratio = unique_count / row_count if row_count else 1.0
                statistical_signal = (unique_ratio < 0.95) and (str(col) not in date_cols)

                if is_id_like:
                    kpi_conf = 0.0
                    col_meta["semantic_role"] = "identifier"
                elif is_discrete_date:
                    kpi_conf = 0.0
                    col_meta["semantic_role"] = "temporal_discrete"
                elif keyword_match and statistical_signal:
                    kpi_conf = 1.0
                    col_meta["semantic_role"] = "kpi"
                elif keyword_match:
                    kpi_conf = 0.8
                    col_meta["semantic_role"] = "kpi"
                elif statistical_signal and unique_count > 5:
                    kpi_conf = 0.4
                    col_meta["semantic_role"] = "measure"
                else:
                    kpi_conf = 0.0
                    col_meta["semantic_role"] = "measure"

                col_meta["kpi_confidence"] = round(kpi_conf, 2)
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
                if (pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object) and df[col].notna().any():
                    for val in sample_vals_clean:
                        if re.fullmatch(r'[A-Z]{2}|[A-Z]{3}', val):
                            pattern_match = True
                            break
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

        # PII Detection
        pii_flags = self._detect_pii(df)

        # Build masked preview if requested
        masked_preview = {}
        if mask_pii:
            for flag in pii_flags:
                if flag["confidence"] >= 0.7:
                    col_name = flag["column"]
                    ptype = self._get_pii_type(flag["reason"])
                    sample_vals = df[col_name].dropna().astype(str).head(5).tolist()
                    masked_preview[col_name] = [self._mask_value(v, ptype) for v in sample_vals]

        # Build detected_kpis list ordered by confidence descending
        if detected_kpis:
            detected_kpis.sort(key=lambda x: x[1], reverse=True)
            detected_kpis = [col for col, _ in detected_kpis]
        else:
            # fallback to non-id numeric columns
            detected_kpis = [c for c in numeric_cols if not re.search(r'(^|_)(id|key|code|zip|postal|row)($|_)', c.lower())] or numeric_cols[:4]

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
        col_meta_dict = {c["name"]: c for c in columns_info}
        effective_row_count = total_rows if total_rows > 0 else row_count
        summary = {
            "filename": filename,
            "row_count": effective_row_count,
            "column_count": col_count,
            "columns": columns_info,
            "col_meta": col_meta_dict,
            "numeric_columns": numeric_cols,
            "date_columns": date_cols,
            "categorical_columns": categorical_cols,
            "detected_kpis": detected_kpis,
            "primary_kpi": primary_kpi,
            "correlation_matrix": correlation_matrix,
            "extracted_chart_data": extracted_chart_data,
            "quality_score": round(float((1 - df.isnull().mean().mean()) * 100), 1),
            "pii_flags": pii_flags,
        }

        # Sample records for LLM semantic domain understanding
        try:
            sample_df = df.head(5).copy()
            for col in sample_df.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_df[col]):
                    sample_df[col] = sample_df[col].astype(str)
            summary["sample_records"] = sample_df.replace({np.nan: None}).to_dict(orient="records")
        except Exception:
            summary["sample_records"] = []

        if was_sampled:
            summary["was_sampled"] = True
            summary["sample_size"] = sample_size
        if masked_preview:
            summary["masked_preview"] = masked_preview

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
