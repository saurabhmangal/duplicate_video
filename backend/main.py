"""
main.py  (v2)
=============
FastAPI backend for Duplicate Video Detector.

Endpoints
---------
POST /api/analyze-folder   → scan a local folder path, kick off analysis job
POST /api/analyze-upload   → upload video files, kick off analysis job
GET  /api/progress/{id}    → SSE stream of real-time progress (0-100 %)
GET  /api/results/{id}     → full JSON results once job is done
DELETE /api/files          → permanently delete specified file paths
GET  /api/video            → stream a local video file (for in-browser preview)
GET  /api/health           → health check
GET  /                     → serve frontend SPA
"""

import asyncio
import json
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from video_processor import VideoProcessor, scan_folder

# ─────────────────────────────────────────────────────────────
# App & middleware
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="Duplicate Video Detector", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

ALLOWED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"}

# ─────────────────────────────────────────────────────────────
# In-memory job store
# ─────────────────────────────────────────────────────────────
# Each job: { status, progress, message, results, error, temp_dir }
_jobs: Dict[str, Dict] = {}
_jobs_lock = threading.Lock()


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


# ─────────────────────────────────────────────────────────────
# Background analysis task
# ─────────────────────────────────────────────────────────────

def _run_analysis(
    job_id: str,
    video_paths: List[str],
    sample_fps: float,
    dup_threshold: int,
    partial_threshold: int,
    temp_dir: Optional[str] = None,
):
    """
    Runs in a background thread.
    Updates _jobs[job_id] with progress; when done sets status='done'.
    """
    _update_job(job_id, status="processing", progress=1, message="Initialising…")

    try:
        processor = VideoProcessor(
            sample_fps          = max(0.1, min(sample_fps, 5.0)),
            dup_threshold       = max(1, min(dup_threshold, 30)),
            partial_threshold   = max(1, min(partial_threshold, 30)),
        )

        def on_progress(pct: float, msg: str):
            _update_job(job_id, progress=round(pct, 1), message=msg)

        results = processor.analyze(video_paths, progress_cb=on_progress)
        _update_job(job_id, status="done", progress=100,
                    message="Analysis complete!", results=results)

    except Exception as exc:
        _update_job(job_id, status="error", error=str(exc),
                    message=f"Error: {exc}")
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def _create_job() -> str:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status"  : "queued",
            "progress": 0,
            "message" : "Queued…",
            "results" : None,
            "error"   : None,
            "temp_dir": None,
        }
    return job_id


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    index = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(index) if os.path.exists(index) else {"message": "Frontend not found"}


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ── Folder path analysis ──────────────────────────────────────

@app.post("/api/analyze-folder")
async def analyze_folder(
    background_tasks: BackgroundTasks,
    folder_path: str = Form(...),
    sample_fps: float = Form(1.0),
    dup_threshold: int = Form(8),
    partial_threshold: int = Form(8),
):
    """Scan a local folder and start an analysis job."""
    path = Path(folder_path)
    if not path.exists():
        raise HTTPException(400, f"Path does not exist: {folder_path}")
    if not path.is_dir():
        raise HTTPException(400, f"Not a directory: {folder_path}")

    videos = scan_folder(str(path))
    if len(videos) < 2:
        raise HTTPException(
            400,
            f"Need at least 2 video files in the folder. Found {len(videos)}."
        )

    job_id = _create_job()
    background_tasks.add_task(
        _run_analysis, job_id, videos, sample_fps, dup_threshold, partial_threshold
    )
    return {"job_id": job_id, "video_count": len(videos)}


# ── File upload analysis ──────────────────────────────────────

@app.post("/api/analyze-upload")
async def analyze_upload(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    sample_fps: float = Form(1.0),
    dup_threshold: int = Form(8),
    partial_threshold: int = Form(8),
):
    """Accept uploaded video files and start an analysis job."""
    if len(files) < 2:
        raise HTTPException(400, "Please upload at least 2 video files.")

    temp_dir    = tempfile.mkdtemp(prefix="dupvid_")
    saved_paths = []
    skipped     = []

    for upload in files:
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTS:
            skipped.append(upload.filename)
            continue
        dest    = os.path.join(temp_dir, upload.filename or f"{uuid.uuid4()}{ext}")
        content = await upload.read()
        with open(dest, "wb") as f:
            f.write(content)
        saved_paths.append(dest)

    if len(saved_paths) < 2:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            400,
            f"Need at least 2 valid video files. Got {len(saved_paths)}. "
            f"Skipped: {', '.join(skipped) or 'none'}"
        )

    job_id = _create_job()
    background_tasks.add_task(
        _run_analysis, job_id, saved_paths,
        sample_fps, dup_threshold, partial_threshold, temp_dir
    )
    return {"job_id": job_id, "video_count": len(saved_paths)}


# ── SSE progress stream ───────────────────────────────────────

@app.get("/api/progress/{job_id}")
async def progress_stream(job_id: str):
    """
    Server-Sent Events stream.
    Frontend connects with  new EventSource('/api/progress/<id>')
    and receives JSON events until status == 'done' or 'error'.
    """
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")

    async def generate():
        while True:
            with _jobs_lock:
                job = dict(_jobs.get(job_id, {}))

            payload = json.dumps({
                "status"  : job.get("status",   "unknown"),
                "progress": job.get("progress", 0),
                "message" : job.get("message",  ""),
                "error"   : job.get("error"),
            })
            yield f"data: {payload}\n\n"

            if job.get("status") in ("done", "error"):
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Results ───────────────────────────────────────────────────

@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == "error":
        raise HTTPException(500, job.get("error", "Unknown error"))
    if job["status"] != "done":
        raise HTTPException(400, f"Job not finished yet (status: {job['status']})")
    return JSONResponse(job["results"])


# ── Delete files ──────────────────────────────────────────────

@app.post("/api/delete-files")
async def delete_files(body: dict):
    """
    Body: { "paths": ["C:/path/to/video.mp4", ...] }
    Permanently deletes each file and returns a summary.
    """
    paths   = body.get("paths", [])
    deleted = []
    errors  = []

    for p in paths:
        try:
            if os.path.isfile(p):
                os.remove(p)
                deleted.append(p)
            else:
                errors.append({"path": p, "error": "File not found"})
        except Exception as exc:
            errors.append({"path": p, "error": str(exc)})

    return {"deleted": deleted, "errors": errors}


# ── Video streaming (for preview) ─────────────────────────────

@app.get("/api/video")
async def serve_video(path: str):
    """
    Stream a local video file so the browser's <video> element can play it.
    Works for folder-path mode where we know the real file location.
    FastAPI's FileResponse handles HTTP Range requests automatically,
    which is required for video seeking to work in browsers.
    """
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "Video file not found")

    # Basic path traversal guard
    abs_path = os.path.realpath(path)
    ext      = os.path.splitext(abs_path)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, "Not a supported video format")

    mime_map = {
        ".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo", ".mov": "video/quicktime",
        ".wmv": "video/x-ms-wmv", ".flv": "video/x-flv",
        ".m4v": "video/mp4", ".mpg": "video/mpeg", ".mpeg": "video/mpeg",
    }
    return FileResponse(abs_path, media_type=mime_map.get(ext, "video/mp4"))


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
