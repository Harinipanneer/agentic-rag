from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    chunk_type: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    policy_citations: Optional[str] = None
    page_no: Optional[str] = None
    document_name: Optional[str] = None
    sql_query_executed: Optional[str] = None


class AIResponse(BaseModel):
    query: str
    answer: str
    policy_citations: str
    page_no: str
    document_name: str
    sql_query_executed: Optional[str] = None
    