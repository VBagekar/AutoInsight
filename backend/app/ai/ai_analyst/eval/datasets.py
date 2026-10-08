"""Small deterministic datasets used by offline evaluation cases."""
from __future__ import annotations

import pandas as pd


def seed_datasets() -> dict[str, pd.DataFrame]:
    """Return fresh copies of the fixed, join/null-heavy evaluation tables."""
    sales = pd.DataFrame([
        {"order_id": 1, "customer_id": "A", "region": "North", "category": "Books", "sale_date": "2024-01-02", "amount": 10.0},
        {"order_id": 2, "customer_id": "B", "region": "south", "category": "Books", "sale_date": "2024-01-15", "amount": 20.0},
        {"order_id": 3, "customer_id": "A", "region": "North", "category": "Games", "sale_date": "2024-02-01", "amount": None},
        {"order_id": 4, "customer_id": "C", "region": None, "category": "games", "sale_date": "2024-02-20", "amount": 40.0},
        {"order_id": 5, "customer_id": None, "region": "West", "category": None, "sale_date": None, "amount": 50.0},
        {"order_id": 6, "customer_id": "D", "region": "North", "category": "Books", "sale_date": "2024-03-05", "amount": 60.0},
    ])
    customers = pd.DataFrame([
        {"customer_id": "A", "customer_name": "Ada", "segment": "consumer"},
        {"customer_id": "B", "customer_name": "Bo", "segment": "business"},
        {"customer_id": "C", "customer_name": "Cy", "segment": None},
        {"customer_id": "X", "customer_name": "Xiu", "segment": "consumer"},
    ])
    targets = pd.DataFrame([
        {"region": "North", "target": 100.0},
        {"region": "South", "target": 80.0},
        {"region": "West", "target": None},
    ])
    return {"sales": sales, "customers": customers, "targets": targets}


def dataset_fingerprint() -> str:
    """Stable fingerprint useful for detecting accidental fixture drift."""
    import hashlib
    payload = "".join(
        f"{name}:{df.to_csv(index=False, na_rep='<NULL>')}"
        for name, df in seed_datasets().items()
    )
    return hashlib.sha256(payload.encode()).hexdigest()
