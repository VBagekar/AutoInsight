from collections import defaultdict, deque
import time
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from .config import AI_RATE_LIMIT_PER_MINUTE
from .service import AnalystError, AnalystService, development_identity
router=APIRouter(prefix="/ai")
service=AnalystService(); requests=defaultdict(deque)
class CreateRequest(BaseModel): privacy: dict|None=None
class ChatRequest(BaseModel): message:str=Field(min_length=1,max_length=10000)
def identity(): return development_identity()
def limited(owner):
    now=time.monotonic(); q=requests[owner]
    while q and now-q[0]>60:q.popleft()
    if len(q)>=AI_RATE_LIMIT_PER_MINUTE: raise HTTPException(429,detail={"code":"RATE_LIMITED","message":"Too many requests."})
    q.append(now)
def fail(e): return HTTPException(e.status_code,detail={"code":e.code,"message":str(e)})
@router.get("/health")
def health(): return {"ok":True,"llm_configured":bool(__import__('app.ai.ai_analyst.config',fromlist=['NVIDIA_API_KEY']).NVIDIA_API_KEY)}
@router.post("/sessions")
async def create(files:list[UploadFile] | None = File(default=None), owner:str=Depends(identity)):
    try:
        limited(owner); result=service.create_session(owner)
        for file in files or []: result=service.ingest(result["session_id"],await file.read(),file.filename or "",owner)
        return result
    except AnalystError as e: raise fail(e) from e
@router.post("/sessions/{session_id}/files")
async def files(session_id:str,file:UploadFile=File(...),owner:str=Depends(identity)):
    try: limited(owner); return service.ingest(session_id,await file.read(),file.filename or "",owner)
    except AnalystError as e: raise fail(e) from e
@router.get("/sessions/{session_id}")
def session(session_id:str,owner:str=Depends(identity)):
    try: return service.get_session(session_id,owner)
    except AnalystError as e: raise fail(e) from e
@router.get("/sessions/{session_id}/card")
def card(session_id:str,owner:str=Depends(identity)):
    try: return service.get_card(session_id,owner)
    except AnalystError as e: raise fail(e) from e
@router.post("/sessions/{session_id}/chat")
def chat(session_id:str,req:ChatRequest,owner:str=Depends(identity)):
    try: limited(owner); return service.ask(session_id,req.message,owner)
    except AnalystError as e: raise fail(e) from e
@router.delete("/sessions/{session_id}",status_code=204)
def delete(session_id:str,owner:str=Depends(identity)):
    try: service.delete_session(session_id,owner)
    except AnalystError as e: raise fail(e) from e
