"""HTTP API for the React frontend. Thin wrapper over the existing pipeline —
same calls app.py made (run_with_versioning, storage.list_analyses).

Run:  uvicorn server:app --reload --port 8000
"""
from __future__ import annotations

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from pipeline import run_with_versioning
import storage

UPLOAD_DIR = "data/uploads"
FRONTEND_DIST = os.path.join("frontend", "dist")

app = FastAPI(title="AI Legal Contract Review API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/history")
def history():
    return storage.list_analyses()


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), compare_with: str | None = Form(None)):
    name = os.path.basename(file.filename or "")
    if not name.lower().endswith((".pdf", ".docx")):
        raise HTTPException(400, "Only PDF or DOCX files are supported.")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, name)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    try:
        doc, report, version = await run_in_threadpool(
            run_with_versioning, save_path, compare_with_doc_id=compare_with or None
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))
    return {
        "document": doc.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
        "version": version.model_dump(mode="json") if version is not None else None,
    }


# Serve the built React app (after `npm run build`) from the same server.
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
