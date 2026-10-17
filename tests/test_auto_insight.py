import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from auto_data_pipeline import (
    UniversalNullNormalizer,
    DataPipelineConfig,
    EnterpriseDataPipeline,
)
from auto_report_pipeline import (
    MatplotlibSeabornEngine,
    SeniorDataScientistReportPipeline,
)


class TestAutoInsightSuite(unittest.TestCase):
    """Unit and integration test suite for AutoInsight pipelines and Matplotlib/Seaborn report engine."""

    def test_null_normalizer_and_mojibake_cleanup(self):
        valid_values = ["Not Returned", "Not Discounted", "Cash", "Online", "-45.5"]
        for val in valid_values:
            self.assertFalse(UniversalNullNormalizer.is_missing_scalar(val))

        dirty_mojibake = pd.Series(["Double\u00c2\u00a0Overhead Camshaft", "  Overhead   Camshaft ", "NULL", "<missing>"])
        cleaned = UniversalNullNormalizer.normalize_series(dirty_mojibake)
        self.assertEqual(cleaned.iloc[0], "Double Overhead Camshaft")
        self.assertEqual(cleaned.iloc[1], "Overhead Camshaft")
        self.assertTrue(pd.isna(cleaned.iloc[2]))
        self.assertTrue(pd.isna(cleaned.iloc[3]))

    def test_identifier_filtering_and_report_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_csv = tmp_path / "car_sales_sample.csv"
            out_dir = tmp_path / "pipeline_output"

            rows = []
            for i in range(60):
                rows.append({
                    "Car_id": f"C_{i:04d}",
                    "Date": f"2024-01-{(i % 28) + 1:02d}",
                    "Customer Name": f"Customer_{i % 15}",
                    "Gender": "Male" if i % 2 == 0 else "Female",
                    "Annual Income": 650000 + (i * 15000),
                    "Company": ["Chevrolet", "Ford", "Dodge", "Toyota", "Mercedes-B"][i % 5],
                    "Engine": "Double\u00c2\u00a0Overhead Camshaft" if i % 2 == 0 else "Overhead Camshaft",
                    "Transmission": "Auto" if i % 2 == 0 else "Manual",
                    "Price ($)": 22000 + (i * 450),
                    "Phone": 7000000 + i,
                    "Body Style": ["SUV", "Sedan", "Hatchback", "Passenger", "Hardtop"][i % 5],
                    "Dealer_Region": ["Austin", "Janesville", "Scottsdale"][i % 3],
                })
            pd.DataFrame(rows).to_csv(raw_csv, index=False)

            pipeline = EnterpriseDataPipeline(output_dir=str(out_dir))
            cleaned_df, _ = pipeline.run_pipeline(str(raw_csv))

            # Verify 'Price ($)' became 'price' (no trailing underscore)
            self.assertIn("price", cleaned_df.columns)
            self.assertNotIn("price_", cleaned_df.columns)

            # Verify 'phone' is NOT treated as an analytical numeric measure
            self.assertNotIn("phone", pipeline.numeric_cols)
            primary = EnterpriseDataPipeline.pick_primary_measure(pipeline.numeric_cols)
            self.assertEqual(primary, "price")

            # Verify Senior Data Scientist Report was generated
            report_html = out_dir / "eda_reports" / "senior_data_scientist_report.html"
            report_json = out_dir / "eda_reports" / "senior_data_scientist_report.json"
            self.assertTrue(report_html.exists())
            self.assertTrue(report_json.exists())

            rep_data = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(rep_data["primary_measure"], "price")
            self.assertGreaterEqual(len(rep_data["visual_insights"]), 5)
            self.assertIn("conclusion", rep_data)


if __name__ == "__main__":
    unittest.main()
