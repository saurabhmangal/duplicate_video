"""
main.py  —  FastAPI backend for Duplicate Video Detector
=========================================================
Endpoints:
  POST /api/analyze        — Upload video files, get back analysis results
  GET  /api/health         — Health check
  GET  /                   — Serve the frontend SPA
"""

import os
import shutil
import tempfile
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from video_processor import VideoProcessor

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Duplicate Video Detector",
    description="Detect exact duplicates and partial-match videos using perceptual hashing.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    """Serve the main HTML page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. Place index.html in ../frontend/"}


@app.get("/api/health")
async def health():
    return {"status": "ok", "message": "Duplicate Video Detector is running"}


@app.post("/api/analyze")
async def analyze_videos(
    files: List[UploadFile] = File(...),
    sample_fps: float = Form(1.0),
    dup_threshold: int = Form(8),
    partial_threshold: int = Form(8),
):
    """
    Accept uploaded video files, analyze them for duplicates and partial matches.

    Parameters (form fields):
    - files            : One or more video files
    - sample_fps       : Frames per second to sample (default 1.0)
    - dup_threshold    : Hamming distance threshold for exact duplicates (default 8)
    - partial_threshold: Hamming distance threshold for partial matches (default 8)

    Returns JSON with:
    - videos      : List of processed videos with thumbnails
    - duplicates  : Pairs of exact/near-duplicate videos
    - partials    : Cases where one video is a segment of another
    - errors      : Any files that could not be processed
    - stats       : Summary statistics
    """

    # Validate that at least 2 files were uploaded
    if len(files) < 2:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least 2 video files to compare.",
        )

    # ── Save uploaded files to a temp directory ──────────────
    temp_dir = tempfile.mkdtemp(prefix="dupvid_")
    saved_paths = []
    skipped = []

    ALLOWED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"}

    try:
        for upload in files:
            # Filter by extension
            ext = os.path.splitext(upload.filename or "")[-1].lower()
            if ext not in ALLOWED_EXTS:
                skipped.append(upload.filename)
                continue

            # Save file
            dest = os.path.join(temp_dir, upload.filename or f"{uuid.uuid4()}{ext}")
            # Handle filename collisions
            if os.path.exists(dest):
                base, extension = os.path.splitext(dest)
                dest = f"{base}_{uuid.uuid4().hex[:6]}{extension}"

            content = await upload.read()
            with open(dest, "wb") as f:
                f.write(content)
            saved_paths.append(dest)

        if len(saved_paths) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 2 valid video files. Got {len(saved_paths)} valid files. "
                       f"Skipped: {', '.join(skipped) or 'none'}",
            )

        # ── Run the analysis ──────────────────────────────────
        processor = VideoProcessor(
            sample_fps=max(0.1, min(sample_fps, 5.0)),  # clamp to 0.1–5 fps
            dup_threshold=max(1, min(dup_threshold, 30)),
            partial_threshold=max(1, min(partial_threshold, 30)),
        )

        results = processor.analyze(saved_paths)

        # Add any format-skipped files to errors list
        for fn in skipped:
            results["errors"].append({"file": fn, "error": "Unsupported file format"})

        return JSONResponse(content=results)

    finally:
        # Always clean up temp files
        shutil.rmtree(temp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
