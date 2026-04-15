from fastapi import FastAPI
from src.api.v1.routes.upload import router as upload_router
from src.api.v1.routes.query import router as query_router
app = FastAPI()

@app.get("/")
def root():
    return {"message": "RAG Running"}




app.include_router(upload_router, prefix="/api/v1/admin")

app.include_router(query_router, prefix="/api/v1")