import os
from pathlib import Path
import yaml
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional in the backend
    def load_dotenv(_path):  # type: ignore
        return False

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Which provider to use: "nvidia" or "openrouter"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia").lower()

SESSIONS_DIR = ROOT / "data" / "sessions"
AI_SESSION_TTL_HOURS = float(os.getenv("AI_SESSION_TTL_HOURS", "24"))
AI_MAX_UPLOAD_MB = int(os.getenv("AI_MAX_UPLOAD_MB", "25"))
AI_LLM_TIMEOUT_S = float(os.getenv("AI_LLM_TIMEOUT_S", "60"))
# Backwards-compatible aliases for early Phase 2 deployments.
AI_SESSION_TTL_SECONDS = int(float(os.getenv("AI_SESSION_TTL_SECONDS", str(AI_SESSION_TTL_HOURS * 3600))))
AI_MAX_UPLOAD_BYTES = int(os.getenv("AI_MAX_UPLOAD_BYTES", str(AI_MAX_UPLOAD_MB * 1024 * 1024)))
AI_QUERY_TIMEOUT_SECONDS = float(os.getenv("AI_QUERY_TIMEOUT_SECONDS", str(AI_LLM_TIMEOUT_S)))
AI_RATE_LIMIT_PER_MINUTE = int(os.getenv("AI_RATE_LIMIT_PER_MINUTE", "20"))
AI_SEND_SAMPLE_ROWS = os.getenv("AI_SEND_SAMPLE_ROWS", "true").lower() in {"1", "true", "yes"}
AI_SEND_TOP_VALUES = os.getenv("AI_SEND_TOP_VALUES", "true").lower() in {"1", "true", "yes"}
AI_SEND_DATA_TO_LLM = AI_SEND_SAMPLE_ROWS or AI_SEND_TOP_VALUES
AI_PRIVACY_DEFAULT = os.getenv("AI_PRIVACY_DEFAULT", "standard")

def load_models() -> dict:
    with open(ROOT / "models.yaml") as f:
        return yaml.safe_load(f)

MODELS = load_models()