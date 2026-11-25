"""Image -> PaddleOCR-VL -> parser. Tries image rotations when the first pass is weak."""
from __future__ import annotations

import io
import mimetypes
from typing import Optional

from PIL import Image, ImageFilter, ImageOps

from .config import Settings
from .extractor import _norm, parse_label_text
from .ocr_client import OCRClient, OCRError
from .schemas import ExtractionResponse, Medicine


def _prepare(image_bytes: bytes, angle: int, max_side: int, enhance: bool = False) -> bytes:
    im = Image.open(io.BytesIO(image_bytes))
    im = ImageOps.exif_transpose(im).convert("RGB")
    if angle:
        im = im.rotate(angle, expand=True)  # counter-clockwise
    if enhance:
        # dot-matrix / low-contrast / small text: grey, stretch contrast, upscale, sharpen
        im = ImageOps.autocontrast(ImageOps.grayscale(im), cutoff=1).convert("RGB")
        scale = 2.0 if max(im.size) < 1600 else 1.0
        if scale > 1:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
        max_side = int(max_side * 1.3)
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

    async def ocr_text(self, filename: str, data: bytes, angle: int = 0, enhance: bool = False) -> str:
        prepared = _prepare(data, angle, self.s.max_image_side, enhance)
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
        attempts: list[dict] = []   # {angle, enhance, text, meds, steps}
        ocr_failures = 0
        total_calls = 0

        async def attempt(angle: int, enhance: bool) -> Optional[dict]:
            nonlocal ocr_failures, total_calls
            total_calls += 1
            tag = f"{angle}°{'+enhanced' if enhance else ''}"
            try:
                text = await self.ocr_text(filename, data, angle, enhance)
            except OCRError as e:
                ocr_failures += 1
                out.errors.append(f"OCR failed at {tag}: {e}")
                return None
            parsed = parse_label_text(text)
            rec = {"angle": angle, "enhance": enhance, "text": text, "meds": parsed.medicines, "steps": parsed.steps}
            attempts.append(rec)
            out.pipeline_steps.append(
                f"paddleocr_vl: OCR at {tag} -> {len(text)} chars, {len(parsed.medicines)} candidate(s)")
            return rec

        def key(r: dict):
            return (_score(r["meds"]), len(r["text"]))

        for angle in angles:
            rec = await attempt(angle, False)
            if rec is None:
                if any("auth" in e.lower() for e in out.errors):
                    break
                continue
            # serialised packs (many SN on one picture) are read at every angle and merged
            if _complete(rec["meds"]) and not any(m.serial_number for m in rec["meds"]):
                break

        best = max(attempts, key=key, default=None)

        # weak result: retry the most promising angles with a contrast-enhanced, upscaled image
        if (best is None or not _complete(best["meds"])) and attempts:
            ranked = sorted(attempts, key=key, reverse=True)[:2]
            for r in ranked:
                await attempt(r["angle"], True)
            best = max(attempts, key=key)

        if best is None:
            if not out.errors:
                out.errors.append("No OCR result")
            return out

        meds = _merge_serials(best["meds"], [r["meds"] for r in attempts if r is not best])
        out.medicines = meds
        out.pipeline_steps.extend(best["steps"])
        if meds:
            out.pipeline_steps.append(
                f"selected rotation {best['angle']}°{' (enhanced)' if best['enhance'] else ''}; {total_calls} OCR call(s)")
            if ocr_failures < total_calls:
                out.errors = [e for e in out.errors if not e.startswith("OCR failed")]
        elif not out.errors:
            out.errors.append("No medicine data (lot / dates / GTIN) could be extracted")
        if debug:
            out.debug = {
                "rotation_used": best["angle"], "enhanced": best["enhance"], "ocr_text": best["text"],
                "attempts": [{"angle": r["angle"], "enhance": r["enhance"], "chars": len(r["text"]),
                              "score": _score(r["meds"]), "text": r["text"]} for r in attempts],
            }
        return out


def _mkey(m: Medicine):
    ident = _norm(m.lot or m.batch_no or "")
    return ((m.gtin or "").lstrip("0") or None, ident or None)


def _merge_serials(best: list[Medicine], others: list[list[Medicine]]) -> list[Medicine]:
    """Different rotations see different boxes of a stack: union their serial numbers per product."""
    if not any(m.serial_number for m in best) and not any(m.serial_number for ms in others for m in ms):
        return best
    result = [m.model_copy() for m in best]
    have = {(_mkey(m), m.serial_number) for m in result if m.serial_number}
    for ms in others:
        for m in ms:
            if not m.serial_number:
                continue
            k = (_mkey(m), m.serial_number)
            if k in have:
                continue
            base = next((b for b in best if _mkey(b)[0] == _mkey(m)[0] and _mkey(b)[1] == _mkey(m)[1]
                         and (_mkey(b)[0] or _mkey(b)[1])), None)
            if base is None:
                if not (_mkey(m)[0] or _mkey(m)[1]):
                    continue
                new = m.model_copy()
            else:
                new = base.model_copy(update={"serial_number": m.serial_number})
            have.add(k)
            result.append(new)
    # drop serial-less duplicates of products that now have serial-bearing rows
    withs = {_mkey(m) for m in result if m.serial_number}
    return [m for m in result if m.serial_number or _mkey(m) not in withs]


def guess_mime(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "image/jpeg"
