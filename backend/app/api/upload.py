from fastapi import APIRouter, UploadFile, File, HTTPException
from app.agents.orchestrator import master_orchestrator

router = APIRouter()

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Receives CSV dataset upload, parses schema & stats locally using Pandas/DuckDB,
    and returns dataset profile + auto-generated initial dashboard.
    """
    if not file.filename or not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Upload a CSV or Excel (.xlsx/.xls) dataset.")

    try:
        contents = await file.read()
        result = master_orchestrator.process_file_and_generate_initial_dashboard(contents, file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")
