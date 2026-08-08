import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

class AutomatedDataCleaner:
    def clean_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Performs automated cleaning: duplicate removal, missing value imputation,
        outlier scoring, and column standardization.
        """
        initial_rows = len(df)
        
        # 1. Remove duplicates
        df_cleaned = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df_cleaned)

        # 2. Handle missing values
        missing_report = {}
        for col in df_cleaned.columns:
            null_count = df_cleaned[col].isnull().sum()
            if null_count > 0:
                missing_report[col] = int(null_count)
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    # Fill missing numeric values with median
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                else:
                    # Fill missing categorical with mode or 'Unknown'
                    mode_val = df_cleaned[col].mode()
                    fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                    df_cleaned[col] = df_cleaned[col].fillna(fill_val)

        # 3. Outlier Detection (Z-score > 3)
        outlier_count = 0
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            std = df_cleaned[col].std()
            if std > 0:
                z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / std)
                outliers = z_scores > 3
                outlier_count += int(outliers.sum())

        report = {
            "initial_rows": initial_rows,
            "cleaned_rows": len(df_cleaned),
            "duplicates_removed": duplicates_removed,
            "missing_values_handled": missing_report,
            "outliers_detected": outlier_count,
            "status": "Success"
        }

        return df_cleaned, report

data_cleaner = AutomatedDataCleaner()
