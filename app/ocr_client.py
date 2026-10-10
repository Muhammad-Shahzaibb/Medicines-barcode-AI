"""Async client for the locally deployed PaddleOCR-VL 1.6 endpoint.

Equivalent of:
    curl -X POST $OCR_URL -H "Authorization: Bearer $OCR_TOKEN" -F "file=@image.jpg"

The response schema of the server is not fixed, so ``extract_text`` walks any
JSON shape (plain text, {"text": ...}, {"markdown": ...}, PaddleX layout
results, OpenAI-style chat responses, ...) and returns one text blob.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Settings


class OCRError(RuntimeError):
    pass


@dataclass
class OCRResult:
    text: str
    raw: Any


_PRIORITY_KEYS = (
    "block_content", "markdown", "md_results", "text", "full_text", "rec_texts",
    "ocr_text", "content", "transcription", "parsing_res_list",
    "layoutParsingResults", "pages", "results", "result", "data", "output",
    "blocks", "lines", "choices", "message",
)
_SKIP_KEYS = {
    "bbox", "box", "boxes", "polygon", "points", "image", "images", "img", "score",
    "scores", "label", "id", "type", "angle", "width", "height", "coordinates",
    "rec_polys", "rec_boxes", "dt_polys", "rec_scores", "status", "success",
    "code", "model", "usage", "time", "elapsed",
}


def _walk(obj: Any) -> list[str]:
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, (int, float, bool)):
        return []
    if isinstance(obj, list):
        out: list[str] = []
        for item in obj:
            out.extend(_walk(item))
        return out
    if isinstance(obj, dict):
        for key in _PRIORITY_KEYS:
            if key in obj and obj[key] not in (None, "", [], {}):
                got = _walk(obj[key])
                if got:
                    return got
        out = []
        for k, v in obj.items():
            if k in _SKIP_KEYS:
                continue
            out.extend(_walk(v))
        return out
    return []


def extract_text(payload: Any) -> str:
    """Flatten whatever the OCR server returned into a single text string."""
    if isinstance(payload, dict):
        err = payload.get("error")
        if err and not any(k in payload for k in ("text", "markdown", "result", "results", "data")):
            raise OCRError(f"OCR server error: {err}")
    return "\n".join(s.strip("\r") for s in _walk(payload) if s and s.strip())


class OCRClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.s = settings
        self._client = httpx.AsyncClient(
            timeout=settings.ocr_timeout,
            verify=settings.ocr_verify_ssl,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ocr(self, image_bytes: bytes, filename: str = "image.jpg",
                  mime: str = "image/jpeg", retries: int = 2) -> OCRResult:
        headers = {}
        if self.s.ocr_token:
            headers["Authorization"] = f"Bearer {self.s.ocr_token}"
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = await self._client.post(
                    self.s.ocr_url,
                    headers=headers,
                    files={self.s.ocr_file_field: (filename, image_bytes, mime)},
                )
                if resp.status_code in (401, 403):
                    raise OCRError(f"OCR auth failed (HTTP {resp.status_code}) - check OCR_TOKEN")
                if resp.status_code in (502, 503, 504) and attempt < retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    raise OCRError(f"OCR HTTP {resp.status_code}: {resp.text[:300]}")
                try:
                    payload: Any = resp.json()
                except ValueError:
                    payload = resp.text
                return OCRResult(text=extract_text(payload), raw=payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = e
                if attempt < retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
        raise OCRError(f"OCR request failed: {type(last).__name__}: {last}")
