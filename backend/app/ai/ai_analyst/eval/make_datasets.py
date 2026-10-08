"""Generate deterministic evaluation CSV fixtures (seed 42)."""
from pathlib import Path
import numpy as np
import pandas as pd


def make_datasets(output: str | Path | None = None) -> Path:
    out = Path(output or Path(__file__).parent / "datasets")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    customers = pd.DataFrame({"customer_id": np.arange(1, 201),
                              "customer_name": [f"Customer {i}" for i in range(1, 201)],
                              "segment": rng.choice(["consumer", "business", None], 200)})
    orders = pd.DataFrame({"order_id": np.arange(1, 2001),
                           "customer_id": rng.integers(1, 211, 2000),
                           "region": rng.choice(["North", "South", "West", None], 2000),
                           "amount": rng.normal(100, 30, 2000).round(2),
                           "order_date": pd.date_range("2024-01-01", periods=2000, freq="D")})
    sales = orders.copy()
    sales["category"] = rng.choice(["Books", "Games", "Home", None], len(sales))
    messy = pd.DataFrame({"date": ["2024-01-02", "02/03/2024", "bad", None],
                          "amount": ["1,200", "N/A", "—", "42.5"],
                          "label": ["ok", "unknown", "", None]})
    for name, frame in {"sales": sales, "customers": customers, "orders": orders, "messy": messy}.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    return out


if __name__ == "__main__":
    print(make_datasets())
