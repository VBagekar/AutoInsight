from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.agents.orchestrator import master_orchestrator

router = APIRouter()

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), sheet_name: str | None = Query(default=None)):
    """
    Receives CSV dataset upload, parses schema & stats locally using Pandas/DuckDB,
    and returns dataset profile + auto-generated initial dashboard.
    """
    if not file.filename or not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Upload a CSV or Excel (.xlsx/.xls) dataset.")

    try:
        contents = await file.read()
        result = master_orchestrator.process_file_and_generate_initial_dashboard(contents, file.filename, sheet_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")

@router.get("/dataset/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, page: int = Query(default=1, ge=1), page_size: int = Query(default=50, ge=1, le=500)):
    """
    Returns a paginated preview of the cleaned dataset.
    """
    try:
        result = master_orchestrator.get_dataset_preview(dataset_id, page, page_size)
        return result
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving dataset preview: {str(e)}")
