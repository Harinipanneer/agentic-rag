from pydantic import BaseModel
from typing import Optional, List


class QueryRequest(BaseModel):
    query: str
    k: int = 20
    chunk_type: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    citation: Optional[str] = None
    page_no: Optional[str] = None
    document_name: Optional[str] = None
    sql_query_executed: Optional[str] = None
    source_chunks: Optional[List[str]] = None

class AIResponse(BaseModel):
    query: str
    answer: str
    citation: str
    page_no: str
    document_name: str
    sql_query_executed: Optional[str] = None
    source_chunks: Optional[List[str]] = None 