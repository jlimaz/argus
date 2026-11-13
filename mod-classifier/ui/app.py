"""Local web UI: upload an IQ recording, classify each burst with the model routing picks.

    cd mod-classifier && .venv-ui/bin/uvicorn ui.app:app --port 8765     # then open http://localhost:8765

Endpoints: GET / (the page), GET /api/summary (models, test scores, routing table),
GET /api/samples (the bundled test samples), POST /api/classify (multipart: one raw
file, or a .sigmf-meta + .sigmf-data pair; sample_rate_hz, center_frequency_hz and
format for raw files), POST /api/classify_sample (a bundled sample by file name).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ui import service

STATIC = service.UI_DIR / "static"
SAMPLES = service.UI_DIR / "samples"
MAX_UPLOAD_BYTES = 8 * service.MAX_SAMPLES + 1_000_000  # cf32 at MAX_SAMPLES, plus a meta file

app = FastAPI(title="mcls model routing")


@lru_cache(maxsize=1)
def bank() -> service.Bank:
    return service.load_bank()


def run(rec: service.Recording) -> dict:
    result = service.classify(bank(), rec)
    result["spectrogram_png"] = service.spectrogram_png(rec, result["bursts"])
    return result


@app.get("/")
def page() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/summary")
def summary() -> dict:
    return service.summary(bank())


@app.get("/api/samples")
def samples() -> list[dict]:
    path = SAMPLES / "samples.json"
    return json.loads(path.read_text()) if path.is_file() else []


def _number(text: str | None) -> float | None:
    if text is None or not str(text).strip():
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise HTTPException(400, f"{text!r} is not a number") from exc


@app.post("/api/classify")
async def classify(files: list[UploadFile] = File(...), sample_rate_hz: str | None = Form(None),
                   center_frequency_hz: str | None = Form(None), format: str | None = Form(None)) -> JSONResponse:
    contents, total = {}, 0
    for upload in files:
        data = await upload.read()
        total += len(data)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Upload is over {MAX_UPLOAD_BYTES // 1_000_000} MB; send a shorter slice.")
        contents[Path(upload.filename or "upload").name] = data
    try:
        rec = service.read_upload(contents, _number(sample_rate_hz), _number(center_frequency_hz), format or None)
    except service.InputError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(run(rec))


@app.post("/api/classify_sample")
def classify_sample(name: str = Form(...)) -> JSONResponse:
    entries = {s["file"]: s for s in samples()}
    if name not in entries:
        raise HTTPException(404, f"No bundled sample named {name!r}")
    entry = entries[name]
    path = SAMPLES / name
    if entry["format"] == "sigmf":
        meta = path.with_suffix(".sigmf-meta")
        files = {meta.name: meta.read_bytes(), path.name: path.read_bytes()}
        rec = service.read_upload(files, None, None)
    else:
        rec = service.read_upload({path.name: path.read_bytes()}, entry["sample_rate_hz"],
                                  entry["center_frequency_hz"], entry["format"])
    result = run(rec)
    result["expected"] = entry
    return JSONResponse(result)
