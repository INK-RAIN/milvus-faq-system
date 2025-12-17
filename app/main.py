from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from app.models.schemas import SearchRequest, SearchResponse, IngestionRequest, IngestionResponse
from app.core.ingestion import ingest_faqs
from app.core.retrieval import search_faq
from app.core.milvus_client import init_settings
from typing import List

app = FastAPI(title="Milvus FAQ Retrieval System")

@app.on_event("startup")
async def startup_event():
    init_settings()

@app.post("/search", response_model=List[SearchResponse])
async def search(request: SearchRequest):
    try:
        results = search_faq(request.query, request.top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", response_model=IngestionResponse)
async def ingest(request: IngestionRequest):
    try:
        count = ingest_faqs(request.faqs)
        return IngestionResponse(message="Successfully ingested FAQs", count=count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
