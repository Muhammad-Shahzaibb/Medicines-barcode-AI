from __future__ import annotations

import mimetypes
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline import process_image

load_dotenv()

app = FastAPI(
    title="Medicine Barcode Extractor API",
    description="Upload a medicine packaging image to extract traceability data via vision LLM.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}


def _resolve_mime(filename: str | None, content_type: str | None) -> str:
    if content_type and content_type in ALLOWED_CONTENT_TYPES:
        return content_type
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed in ALLOWED_CONTENT_TYPES:
            return guessed
    return "image/jpeg"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract")
async def extract_medicines(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    mime = _resolve_mime(file.filename, file.content_type)
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES and mime == "image/jpeg":
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload jpg, jpeg, png, webp, or bmp.",
            )

    result = process_image(
        image_bytes=image_bytes,
        source_name=file.filename,
        mime=mime,
    )
    return result.to_dict()
