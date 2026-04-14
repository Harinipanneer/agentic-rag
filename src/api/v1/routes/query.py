from fastapi import APIRouter
from src.api.v1.schemas.query_schema import QueryRequest, QueryResponse
from src.api.v1.services.query_service import run_agent

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    print("\n ===== QUERY RECEIVED =====")
    print(f"Query: {request.query}")
    result = run_agent(request.query)
    print(" FINAL RESPONSE:", result)
    return QueryResponse(**result)

