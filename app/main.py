import asyncio
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .ocr_client import OCRClient, OCRError
from .pipeline import ExtractionPipeline
from .schemas import ExtractionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    ocr = OCRClient(s)
    app.state.pipeline = ExtractionPipeline(ocr, s)
    app.state.sem = asyncio.Semaphore(s.max_concurrency)
    yield
    await ocr.aclose()


app = FastAPI(
    title="Medicine Label Extractor",
    description="Extracts GTIN / batch / lot / MFG / EXP / serial from medicine images using PaddleOCR-VL 1.6.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _dump(res: ExtractionResponse, debug: bool = False) -> dict:
    d = res.model_dump()
    if not debug:
        d.pop("debug", None)
    return d


async def _run(file: UploadFile, debug: bool = False) -> ExtractionResponse:
    data = await file.read()
    name = file.filename or "upload"
    if not data:
        return ExtractionResponse(source_image=name, errors=["Empty file"])
    async with app.state.sem:
        return await app.state.pipeline.process(name, data, debug=debug)


@app.get("/health")
async def health():
    s = get_settings()
    return {"status": "ok", "ocr_url": s.ocr_url, "token_configured": bool(s.ocr_token)}


@app.post("/extract", responses={200: {"model": ExtractionResponse}})
async def extract(file: UploadFile = File(...)):
    """Extract medicine data from ONE image."""
    return JSONResponse(_dump(await _run(file)))


@app.post("/extract/batch", responses={200: {"model": List[ExtractionResponse]}})
async def extract_batch(files: List[UploadFile] = File(...)):
    """Extract medicine data from MANY images (processed concurrently)."""
    results = await asyncio.gather(*[_run(f) for f in files])
    return JSONResponse([_dump(r) for r in results])


@app.post("/ocr/raw")
async def ocr_raw(file: UploadFile = File(...), angle: int = Query(0)):
    """Debug helper: returns the plain text PaddleOCR-VL read from the image (no parsing)."""
    data = await file.read()
    try:
        text = await app.state.pipeline.ocr_text(file.filename or "upload", data, angle % 360)
    except OCRError as e:
        raise HTTPException(502, str(e))
    return {"source_image": file.filename, "angle": angle, "text": text}
