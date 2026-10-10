"""Image -> PaddleOCR-VL -> parser. Tries image rotations when the first pass is weak."""
from __future__ import annotations

import io
import mimetypes

from PIL import Image, ImageOps

from .config import Settings
from .extractor import parse_label_text
from .ocr_client import OCRClient, OCRError
from .schemas import ExtractionResponse, Medicine


def _prepare(image_bytes: bytes, angle: int, max_side: int) -> bytes:
    im = Image.open(io.BytesIO(image_bytes))
    im = ImageOps.exif_transpose(im).convert("RGB")
    if angle:
        im = im.rotate(angle, expand=True)  # counter-clockwise
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def _score(meds: list[Medicine]) -> float:
    return max((m.confidence for m in meds), default=0.0)


def _complete(meds: list[Medicine]) -> bool:
    return any(m.exp_date and (m.lot or m.batch_no or m.gtin) and m.confidence >= 0.6 for m in meds)


class ExtractionPipeline:
    def __init__(self, ocr: OCRClient, settings: Settings):
        self.ocr = ocr
        self.s = settings

    async def ocr_text(self, filename: str, data: bytes, angle: int = 0) -> str:
        prepared = _prepare(data, angle, self.s.max_image_side)
        res = await self.ocr.ocr(prepared, filename=filename, mime="image/jpeg")
        return res.text

    async def process(self, filename: str, data: bytes, debug: bool = False) -> ExtractionResponse:
        out = ExtractionResponse(source_image=filename)
        try:
            Image.open(io.BytesIO(data)).verify()
        except Exception:
            out.errors.append("File is not a readable image")
            return out

        angles = self.s.angles if self.s.rotation_search else self.s.angles[:1]
        best_meds: list[Medicine] = []
        best_text, best_angle, best_steps = "", 0, []
        attempts: list[dict] = []
        ocr_failures = 0

        for angle in angles:
            try:
                text = await self.ocr_text(filename, data, angle)
            except OCRError as e:
                ocr_failures += 1
                out.errors.append(f"OCR failed at {angle}°: {e}")
                if "auth" in str(e).lower():
                    break
                continue
            parsed = parse_label_text(text)
            attempts.append({"angle": angle, "chars": len(text), "score": _score(parsed.medicines)})
            out.pipeline_steps.append(
                f"paddleocr_vl: OCR at {angle}° -> {len(text)} chars, {len(parsed.medicines)} candidate(s)")
            better = (_score(parsed.medicines), len(text)) > (_score(best_meds), len(best_text))
            if better or not best_text:
                best_meds, best_text, best_angle, best_steps = parsed.medicines, text, angle, parsed.steps
            if _complete(parsed.medicines):
                break

        out.medicines = best_meds
        out.pipeline_steps.extend(best_steps)
        if best_meds:
            out.pipeline_steps.append(f"selected rotation {best_angle}°")
            if ocr_failures < len(angles):
                # per-angle OCR hiccups are noise once another angle produced a result
                out.errors = [e for e in out.errors if not e.startswith("OCR failed")]
        elif not out.errors:
            out.errors.append("No medicine data (lot / dates / GTIN) could be extracted")
        if debug:
            out.debug = {"rotation_used": best_angle, "ocr_text": best_text, "attempts": attempts}
        return out


def guess_mime(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "image/jpeg"
