from fastapi import APIRouter, UploadFile, File
from typing import List
import os
import shutil
from pathlib import Path

from src.ingestion.ingestion import run_ingestion

router = APIRouter()

UPLOAD_DIR = "data"

@router.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):  # Use List[UploadFile] here
    print("\n ===== MULTI FILE UPLOAD =====")
    results = []
    try:
        # Step 1: create folder if not exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        for file in files:
            print(f"\n Processing file: {file.filename}")

            # Step 2: Save the file to the server
            file_path = Path(UPLOAD_DIR) / file.filename

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            print(f" File saved at: {file_path}")

            # Step 3: Run ingestion process
            print(" Running ingestion...")
            result = run_ingestion(str(file_path))

            # Add the result for each file processed
            results.append({
                "file": file.filename,
                "result": result
            })

        print(" ALL FILES PROCESSED")

        return {
            "status": "success",
            "files_processed": len(results),
            "data": results
        }

    except Exception as e:
        print(f" ERROR: {e}")
        return {"error": str(e)}