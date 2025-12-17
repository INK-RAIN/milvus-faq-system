from pydantic import BaseModel
from typing import List, Optional

class FAQItem(BaseModel):
    question: str
    answer: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResponse(BaseModel):
    question: str
    answer: str
    score: float

class IngestionRequest(BaseModel):
    faqs: List[FAQItem]

class IngestionResponse(BaseModel):
    message: str
    count: int
