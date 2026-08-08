import numpy as np
import pandas as pd
from typing import Dict, Any, List

class ForecastingAgent:
    def forecast_metric(self, historical_values: List[float], periods: int = 4) -> Dict[str, Any]:
        """
        Generates predictive forecast points with best-case, expected, and worst-case confidence intervals.
        """
        if not historical_values:
            historical_values = [4000, 4200, 4500, 4800, 5200, 5600]

        vals = np.array(historical_values)
        n = len(vals)
        x = np.arange(n)
        
        # Fit linear trend
        slope, intercept = np.polyfit(x, vals, 1)
        
        # Std dev for confidence bounds
        residuals = vals - (slope * x + intercept)
        std_err = np.std(residuals) if len(residuals) > 1 else vals.mean() * 0.05

        forecast_points = []
        for i in range(1, periods + 1):
            future_x = n - 1 + i
            expected = slope * future_x + intercept
            upper = expected + (1.96 * std_err * np.sqrt(i))
            lower = expected - (1.96 * std_err * np.sqrt(i))
            
            forecast_points.append({
                "period": f"Period +{i}",
                "expected": round(float(expected), 2),
                "best_case": round(float(upper), 2),
                "worst_case": round(float(max(0, lower)), 2)
            })

        return {
            "forecast_type": "Linear Trend with 95% Confidence Intervals",
            "historical_mean": round(float(vals.mean()), 2),
            "projected_growth_rate": f"{round(float((slope / vals.mean()) * 100), 2)}% per period",
            "forecast_points": forecast_points,
            "ai_summary": f"Revenue is projected to grow by {round(float((slope / vals.mean()) * 100), 1)}% next quarter."
        }

forecasting_agent = ForecastingAgent()
