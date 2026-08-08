import sys
import os
from pathlib import Path

# Force UTF-8 encoding for Windows standard streams
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add backend root directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import upload, query, forecast

app = FastAPI(
    title="AutoInsights AI Analytics Platform Engine",
    description="Backend API powered by NVIDIA Nemotron-3 Super 120B & Multi-Agent Analytics",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(upload.router, prefix="/api", tags=["Upload & Profiling"])
app.include_router(query.router, prefix="/api", tags=["AI Query & Reasoning"])
app.include_router(forecast.router, prefix="/api", tags=["Forecasting"])

@app.get("/")
def read_root():
    return {
        "status": "Online",
        "engine": "AutoInsights AI Analytics Engine",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
