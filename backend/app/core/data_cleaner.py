import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

class AutomatedDataCleaner:
    def clean_dataset(self, df: pd.DataFrame, treat_outliers: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Performs automated cleaning: duplicate removal, missing value imputation,
        outlier treatment, and column standardization.
        """
        initial_rows = len(df)

        # 1. Remove duplicates
        df_cleaned = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df_cleaned)

        # Detect datetime columns
        datetime_cols: List[str] = []
        for col in df_cleaned.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_cleaned[col]):
                try:
                    parsed = pd.to_datetime(df_cleaned[col], errors='coerce')
                    if parsed.notna().mean() > 0.8:
                        datetime_cols.append(col)
                except Exception:
                    pass

        row_index = np.arange(len(df_cleaned))

        # 2. Handle missing values with column-aware strategies
        missing_report = {}
        imputation_strategy_used = {}
        high_missing_flagged = []

        for col in df_cleaned.columns:
            null_count = df_cleaned[col].isnull().sum()
            if null_count == 0:
                continue
            missing_report[col] = int(null_count)
            missing_rate = null_count / len(df_cleaned)

            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                # Decide imputation strategy
                use_forward_fill = False
                if datetime_cols:
                    # Correlation with row order as proxy for time-series
                    numeric_series = df_cleaned[col].astype(float)
                    corr = numeric_series.corr(pd.Series(row_index, index=df_cleaned.index), method='spearman')
                    if pd.notna(corr) and abs(corr) > 0.5:
                        use_forward_fill = True
                if use_forward_fill:
                    df_cleaned[col] = df_cleaned[col].ffill()
                    imputation_strategy_used[col] = "forward_fill"
                else:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                    imputation_strategy_used[col] = "median"
            else:
                # Categorical column
                if missing_rate > 0.4:
                    high_missing_flagged.append(col)
                    imputation_strategy_used[col] = "skipped_high_missing"
                    # leave NaNs as‑is
                else:
                    mode_val = df_cleaned[col].mode()
                    fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                    df_cleaned[col] = df_cleaned[col].fillna(fill_val)
                    imputation_strategy_used[col] = "mode"

        # 3. Outlier detection and treatment (winsorization)
        outlier_treatment = {}
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            std = df_cleaned[col].std()
            outlier_count = 0
            if std > 0:
                z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / std)
                outliers = z_scores > 3
                outlier_count = int(outliers.sum())
            if treat_outliers and outlier_count > 0:
                lower = df_cleaned[col].quantile(0.01)
                upper = df_cleaned[col].quantile(0.99)
                df_cleaned[col] = df_cleaned[col].clip(lower, upper)
                action = "winsorized_1_99"
            else:
                action = "none"
            outlier_treatment[col] = {"count": outlier_count, "action": action}

        report = {
            "initial_rows": initial_rows,
            "cleaned_rows": len(df_cleaned),
            "duplicates_removed": duplicates_removed,
            "missing_values_handled": missing_report,
            "imputation_strategy_used": imputation_strategy_used,
            "high_missing_flagged": high_missing_flagged,
            "outlier_treatment": outlier_treatment,
            "status": "Success"
        }

        return df_cleaned, report

data_cleaner = AutomatedDataCleaner()
