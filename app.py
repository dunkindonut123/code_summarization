"""FastAPI web app — Java README summarization comparison."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from engine.pipeline import MODEL_CATALOG, SummarizationPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

pipeline = SummarizationPipeline(top_n=5)
templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting model load (first run may download datasets and weights) ...")
    try:
        pipeline.load()
    except Exception as exc:
        logger.error("Startup load failed: %s", exc)
    yield


app = FastAPI(
    title="Auto-README Java Summarizer",
    description="Compare four summarization models on Java source files.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "models": MODEL_CATALOG,
            "ready": pipeline.ready,
            "loading": pipeline.loading,
            "error": pipeline.error,
        },
    )


@app.get("/api/health")
async def health():
    return {
        "ready": pipeline.ready,
        "loading": pipeline.loading,
        "error": pipeline.error,
        "models": MODEL_CATALOG,
    }


@app.post("/api/summarize")
async def summarize(file: UploadFile = File(...)):
    if not pipeline.ready:
        if pipeline.loading:
            raise HTTPException(status_code=503, detail="Models are still loading. Try again shortly.")
        raise HTTPException(
            status_code=503,
            detail=pipeline.error or "Models failed to load. Check server logs.",
        )

    filename = file.filename or "upload.java"
    if not filename.lower().endswith(".java"):
        raise HTTPException(status_code=400, detail="Please upload a .java file.")

    raw = await file.read()
    try:
        java_source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text.") from exc

    if not java_source.strip():
        raise HTTPException(status_code=400, detail="File is empty.")

    try:
        result = pipeline.compare(java_source, filename=filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Summarization failed")
        raise HTTPException(status_code=500, detail="Summarization failed.") from exc

    return JSONResponse({
        "filename": result.filename,
        "char_count": result.char_count,
        "token_count": result.token_count,
        "statement_count": result.statement_count,
        "method_count": result.method_count,
        "top_n": result.top_n,
        "total_elapsed_ms": round(result.total_elapsed_ms, 1),
        "summaries": [
            {
                "model_id": s.model_id,
                "model": s.model,
                "tier": s.tier,
                "approach": s.approach,
                "accent": s.accent,
                "summary": s.summary,
                "elapsed_ms": round(s.elapsed_ms, 1),
                "methods": [
                    {"name": m.name, "summary": m.summary}
                    for m in s.methods
                ],
            }
            for s in result.summaries
        ],
    })
