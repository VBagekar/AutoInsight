import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Resolve .env relative to the backend directory (parent of app/)
_backend_dir = Path(__file__).resolve().parent.parent
_env_path = _backend_dir / ".env"

class Settings(BaseSettings):
    NVIDIA_API_KEY: str = ""
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
    MAX_UPLOAD_SIZE_MB: int = 100

    @property
    def is_llm_configured(self) -> bool:
        """Check if the LLM API key is set and available."""
        return bool(self.NVIDIA_API_KEY and self.NVIDIA_API_KEY.strip())

    def __repr__(self) -> str:
        """Prevent accidental logging of the API key."""
        return (
            f"Settings(NVIDIA_BASE_URL={self.NVIDIA_BASE_URL!r}, "
            f"NVIDIA_MODEL={self.NVIDIA_MODEL!r}, "
            f"NVIDIA_API_KEY={'***' if self.NVIDIA_API_KEY else '(not set)'}, "
            f"HOST={self.HOST!r}, PORT={self.PORT})"
        )

    __str__ = __repr__

    class Config:
        env_file = str(_env_path)
        extra = "ignore"

settings = Settings()
