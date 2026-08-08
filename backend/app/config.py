import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Resolve .env relative to the backend directory (parent of app/)
_backend_dir = Path(__file__).resolve().parent.parent
_env_path = _backend_dir / ".env"

class Settings(BaseSettings):
    NVIDIA_API_KEY: str = ""
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_MODEL: str = "nvidia/nemotron-3-super-120b-a12b"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = str(_env_path)
        extra = "ignore"

settings = Settings()
