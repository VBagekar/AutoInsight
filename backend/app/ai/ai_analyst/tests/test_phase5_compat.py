import json
from pathlib import Path

from app.main import app
from fastapi.routing import APIRoute


AI_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = AI_DIR.parents[2]


def test_phase5_inventory_is_keep_compat_and_zero_removals():
    inventory = json.loads((AI_DIR / "ai_analyst" / "legacy_contracts.json").read_text())
    assert inventory["phase"] == 5
    assert inventory["removals"] == 0
    assert inventory["endpoints"]
    assert all(item["verdict"] == "KEEP-COMPAT" for item in inventory["endpoints"])
    assert all(not item["analyst_service_preserves"] for item in inventory["endpoints"])


def test_legacy_routes_remain_registered_alongside_ai_routes():
    routes = set()
    for route in app.routes:
        if isinstance(route, APIRoute):
            routes.update((method, route.path) for method in route.methods)
        elif hasattr(route, "original_router"):
            prefix = route.include_context.prefix
            routes.update(
                (method, prefix + child.path)
                for child in route.original_router.routes
                if isinstance(child, APIRoute)
                for method in child.methods
            )
    expected = {
        ("POST", "/api/upload"),
        ("POST", "/api/query"),
        ("GET", "/api/dataset/{dataset_id}/preview"),
        ("GET", "/api/dataset/{dataset_id}/download"),
        ("POST", "/api/dataset/{dataset_id}/preprocess"),
        ("POST", "/api/forecast"),
        ("GET", "/api/health"),
    }
    assert expected <= routes
    assert ("POST", "/api/ai/sessions/{session_id}/chat") in routes


def test_frontend_legacy_callers_still_use_only_legacy_contracts():
    client = (REPO_ROOT / "src/api/client.js").read_text()
    dashboard = (REPO_ROOT / "src/components/Dashboard.jsx").read_text()
    for path in ("/upload", "/query", "/forecast", "/health", "/dataset/"):
        assert path in client
    for function in ("uploadDataset", "streamAIQuery", "fetchForecast",
                     "fetchDatasetPreview", "preprocessDataset", "downloadDataset",
                     "fetchHealth"):
        assert function in client
    for function in ("uploadDataset", "streamAIQuery", "fetchDatasetPreview",
                     "preprocessDataset", "downloadDataset", "fetchHealth"):
        assert function in dashboard
    assert "/api/ai/" not in client
    assert "/api/ai/" not in dashboard
