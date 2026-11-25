"""Async client for the OpenAI-compatible vision endpoint (qwen3.8).

Sends the image base64-encoded inside an OpenAI chat-completions request:
    POST $OCR_URL  (e.g. https://101.46.52.232/v2/infer/<uuid>/v1/chat/completions)
    Authorization: Bearer $OCR_TOKEN
    { "model": "$OCR_MODEL",
      "messages": [{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:<mime>;base64,<data>"}},{"type":"text","text":"..."}]}],
      "max_completion_tokens": $OCR_MAX_TOKENS }

The assistant reply text is extracted from choices[0].message.content.
"""
from __future__ import annotations

import asyncio
import base64
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


def _extract_chat_text(payload: Any) -> str:
    """Pull the assistant reply text out of an OpenAI chat-completions response."""
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        err = payload.get("error")
        if err:
            msg = err.get("message", err) if isinstance(err, dict) else err
            raise OCRError(f"Model error: {msg}")
        # Standard OpenAI shape: choices[0].message.content
        choices = payload.get("choices") or []
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message") or {}
                content = message.get("content", "")
                if isinstance(content, list):
                    # multi-part content — join text parts
                    return "\n".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ).strip()
                return str(content).strip()
        # Fallback: any "text" or "content" key at top level
        for key in ("text", "content", "output", "result"):
            if key in payload and payload[key]:
                return str(payload[key]).strip()
    return str(payload).strip()


# Prompt sent with every image — instructs the model to transcribe label text
_OCR_PROMPT = (
    "You are an OCR engine. Transcribe ALL text visible in this medicine / medical-device "
    "label image exactly as printed. Preserve line breaks, numbers, dates, lot/batch codes, "
    "GTIN barcodes, and serial numbers. Do NOT add explanations or summaries — output only "
    "the raw transcribed text."
)


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
        headers = {"Content-Type": "application/json"}
        if self.s.ocr_token:
            headers["Authorization"] = f"Bearer {self.s.ocr_token}"

        b64 = base64.b64encode(image_bytes).decode()
        body = {
            "model": self.s.ocr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {"type": "text", "text": _OCR_PROMPT},
                    ],
                }
            ],
            "max_completion_tokens": self.s.ocr_max_tokens,
        }

        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = await self._client.post(self.s.ocr_url, headers=headers, json=body)
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
                return OCRResult(text=_extract_chat_text(payload), raw=payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = e
                if attempt < retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
        raise OCRError(f"OCR request failed: {type(last).__name__}: {last}")
