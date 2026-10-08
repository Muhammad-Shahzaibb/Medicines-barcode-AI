from __future__ import annotations

import base64
import json
import os
import re
from functools import lru_cache
from io import BytesIO
from typing import Any

from openai import OpenAI
from PIL import Image, ImageOps

from src.gs1_parser import parse_gs1
from src.models import MedicineRecord
from src.normalizer import normalize_date, normalize_gtin

DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_API_KEY = ""
DEFAULT_LLM_MODEL = "gpt-4o"
DEFAULT_LLM_SEED = 42
MAX_IMAGE_SIDE = 2048
NULL_TOKENS = {
    "",
    "null",
    "none",
    "n/a",
    "na",
    "nil",
    "-",
    "--",
    "unknown",
    "not found",
    "not visible",
    "not present",
    "unavailable",
}
INVALID_FIELD_VALUES = {
    "STERILE",
    "STERILER",
    "EXPIRY",
    "EXPIRYDATE",
    "DATE",
    "MADE",
    "FRANCE",
    "CHINA",
    "ITALY",
    "INDIA",
    "SPAIN",
    "KSA",
    "GERMANY",
    "UDI",
    "GTIN",
    "BATCH",
    "BATCHCODE",
    "LOT",
    "CODE",
    "MFG",
    "MFD",
    "EXP",
    "REF",
    "SIZE",
}

SYSTEM_PROMPT = """You are a pharmaceutical and medical-device packaging data extraction expert.
Your task: scan EVERY label, sticker, and printed area in the image and return one JSON object per distinct product label.
Extract ONLY text clearly visible in the image. Never invent, complete, or guess characters.

OUTPUT — return ONLY valid JSON, no markdown fences:
{
  "medicines": [
    {
      "gtin": string or null,
      "batch_no": string or null,
      "lot": string or null,
      "mfg_date": string or null,
      "exp_date": string or null,
      "serial_number": string or null,
      "gs1_text": string or null,
      "visible_text": [string]
    }
  ]
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MULTI-LABEL DETECTION (critical)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Scan the ENTIRE image for separate product labels — boxes, bags, shelf labels, stickers.
- If two or more distinct product labels appear (even if identical products), create a separate medicines[] entry for each.
- A new label starts when you see a new product name, a separate barcode, or a new block of LOT/MFG/EXP fields.
- Do NOT merge multiple distinct labels into one entry.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ISO 15223 ICONS (appear WITHOUT text labels)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
These symbols appear as small printed icons — identify them by shape and assign accordingly:
- FACTORY/MANUFACTURER icon: looks like a small building with a chimney or roof-line (⌂ shape, sometimes drawn as "m̈" wave). This icon means the date next to it is the MANUFACTURE DATE → mfg_date.
- HOURGLASS/USE-BY icon: looks like an hourglass or sand-timer shape (wide top, narrow middle, wide bottom). This icon means the date next to it is the EXPIRY/USE-BY DATE → exp_date.
- LOT BOX icon: the word LOT or [LOT] inside a rectangle/box symbol → lot value.
Never swap factory and hourglass dates. Factory (building) = mfg_date. Hourglass = exp_date.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GTIN:
  ✓ GS1 AI (01) followed by 8–14 digits: e.g. (01)06975486453029 → gtin=06975486453029
  ✓ EAN-13/14 numeric code printed directly below a 1D barcode
  ✓ Label says: GTIN, GTN, EAN
  ✗ NOT GTIN: REF, CAT NO, CAT.NO, Ref., Ref #, catalog code (e.g. IT-31, GS051M, KM-DM033), HIBC (+…), alphanumeric UDI
  ✗ If only REF/CAT/alphanumeric code is visible → gtin = null

BATCH (batch_no):
  ✓ Labels: Batch, Batch No, Batch No., Batch Number, Batch Code, BNO, B.NO, BN, B/N, B.N, B.N.
  ✗ If none of those labels present → batch_no = null (even if LOT exists)
  SPECIAL: If a [LOT] box symbol has sub-label "Batch Code" printed under it → fill BOTH lot AND batch_no with the same value.

LOT (lot):
  ✓ Labels: LOT, Lot, LOT NO., Lot No, LOT #, Lote, GS1 AI (10), [LOT] box symbol
  ✗ If none of those labels present → lot = null (even if Batch exists)
  NOTE: An 8-digit number (YYYYMMDD format) printed beside [LOT] is a LOT NUMBER, not a date.

SERIAL (serial_number):
  ✓ Labels: SN, SNO, S/N, Serial, Serial No, GS1 AI (21)
  ✗ Do not put lot, batch, date, REF, or GTIN into serial_number

MFG DATE (mfg_date) — any of these patterns → manufacturing date:
  ✓ ISO 15223 factory/building icon + date
  ✓ Labels: MFG, MFG., MFG DATE, MFG.DATE, MFD, MD, PRO, P:, GS1 AI (11)
  ✓ Text label "Date of manufacture" or "Manufacturing date"

EXP DATE (exp_date) — any of these patterns → expiry/use-by date:
  ✓ ISO 15223 hourglass icon + date
  ✓ Labels: EXP, EXP., EXP DATE, EXP.DATE, EXPIRY DATE, EXPIRY, Use By, Use-By, CAD, BB, GS1 AI (17)
  ✓ Text label "Shelf life" followed by a date

NEVER swap mfg_date and exp_date.
NEVER derive mfg_date from a lot/batch number that resembles a date.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATE NORMALIZATION → always output YYYY-MM-DD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  20250415       → 2025-04-15
  250415 (YYMMDD)→ 2025-04-15
  2024-06        → 2024-06-01
  07-2023        → 2023-07-01
  07/2023        → 2023-07-01
  11.2025        → 2025-11-01
  30/11/2028     → 2028-11-30
  2029-01-15     → 2029-01-15 (already normalized)
  If day is absent, use 01 as the day.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GS1 HUMAN-READABLE TEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Copy the EXACT printed GS1 string (with parentheses) into gs1_text.
  Common AIs: (01)=GTIN (10)=Lot (11)=MfgDate YYMMDD (17)=ExpDate YYMMDD (21)=Serial
  Example: "(01)06975486453029 (11)250415(17)300414(10)KM2503173"
  This text often appears directly below a 1D barcode as a small-font human-readable string.
  If the GS1 string gives GTIN/lot/dates that are missing from other fields, fill those fields from it.
  If no parenthesized GS1 text exists → gs1_text = null.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISIBLE_TEXT — evidence logging
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  List every printed line/value you used as evidence: LOT, BN, MFG, EXP, GTIN, SN, GS1 string.
  Copy characters exactly as printed. Do not add text not visible in image.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANTI-HALLUCINATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Unclear character → null for that whole field. Prefer null over a wrong value.
  2. Watch carefully: 0 vs O, 0 vs 7, 1 vs I, 5 vs S, 8 vs B, 6 vs G.
  3. Preserve leading zeros exactly as printed.
  4. REF / CAT NO is NOT a GTIN or serial.
  5. Size, quantity, gauge (18G, 6.5mm, 5.0#) are NOT field values.
  6. Country names, addresses, manufacturer names, websites are NOT field values.
  7. Do NOT fill batch_no when only LOT label is present. Do NOT fill lot when only BN/Batch label is present.
  8. Do NOT put expiry date as mfg_date or vice versa.
"""

USER_TEXT = (
    "Examine the ENTIRE image carefully. "
    "Step 1: Count how many distinct product labels/packages are visible (look for separate barcodes, separate LOT/EXP blocks, separate product names). "
    "Step 2: For EACH distinct label, extract one medicines[] entry. "
    "Step 3: Read ALL text including small-font GS1 human-readable strings below barcodes (e.g. '(01)xxx(17)xxx(10)xxx'). "
    "Step 4: Identify factory-building icons (mfg_date) and hourglass icons (exp_date) — they appear WITHOUT text labels. "
    "Step 5: Copy exact printed values for LOT/BATCH/MFG/EXP/GTIN/SN into visible_text. "
    "Step 6: Return ONLY valid JSON — no markdown, no explanation."
)

RETRY_PROMPT = (
    "Your previous response was invalid or empty. Try again. "
    "Look at every part of the image including small text near barcodes. "
    "For factory/building icon → mfg_date. For hourglass icon → exp_date. "
    "For [LOT] box → lot. For B/N or B.N or Batch → batch_no. "
    "Use null for fields that are genuinely not visible. Do not guess. "
    "Return ONLY the JSON object with the medicines array."
)


def _image_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _prepare_image(image_bytes: bytes, mime: str) -> tuple[bytes, str]:
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image) or image
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    longest = max(width, height)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95, optimize=True)
    return buffer.getvalue(), "image/jpeg"


def _parse_llm_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in NULL_TOKENS:
        return None
    return text


def _looks_like_date(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if len(digits) == 8:
        year = int(digits[:4])
        return 1990 <= year <= 2045
    if len(digits) == 6:
        year = 2000 + int(digits[:2])
        return 1990 <= year <= 2045
    return False


def _evidence_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    visible = item.get("visible_text") or item.get("evidence")
    if isinstance(visible, str):
        parts.append(visible)
    elif isinstance(visible, list):
        parts.extend(str(line) for line in visible if line)
    gs1 = item.get("gs1_text") or item.get("gs1")
    if isinstance(gs1, list):
        parts.extend(str(part) for part in gs1 if part)
    elif gs1:
        parts.append(str(gs1))
    return " ".join(parts)


def _compact(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _supported_by_evidence(value: str, evidence: str, is_date: bool = False) -> bool:
    if not evidence or not value:
        return True
    compact_evidence = _compact(evidence)
    if is_date:
        normalized = normalize_date(value)
        if not normalized:
            return False
        year, month, day = normalized.split("-")
        candidates = (
            f"{year}{month}{day}",
            f"{year}{month}",
            f"{month}{year}",
            f"{year[2:]}{month}{day}",
            f"{day}{month}{year}",
            f"{month}{day}{year}",
        )
        return any(candidate in compact_evidence for candidate in candidates)
    return _compact(value) in compact_evidence


def _extract_labeled_value(evidence: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, evidence, re.I)
        if not match:
            continue
        value = match.group(1).strip(" .:-")
        if value and _compact(value) not in INVALID_FIELD_VALUES:
            return value
    return None


def _merge_gs1_text(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("gs1_text") or item.get("gs1")
    if isinstance(raw, list):
        raw = " ".join(str(part) for part in raw if part)
    raw = _clean_str(raw)
    if not raw or not re.search(r"\(\d{2,4}\)", raw):
        return item

    parsed = parse_gs1(raw)
    if parsed.gtin and not _clean_str(item.get("gtin")):
        item["gtin"] = parsed.gtin
    if parsed.lot and not _clean_str(item.get("lot")):
        item["lot"] = parsed.lot
    if parsed.mfg_date and not _clean_str(item.get("mfg_date")):
        item["mfg_date"] = parsed.mfg_date
    if parsed.exp_date and not _clean_str(item.get("exp_date")):
        item["exp_date"] = parsed.exp_date
    if parsed.serial_number and not _clean_str(item.get("serial_number")):
        item["serial_number"] = parsed.serial_number
    return item


def _repair_record(item: dict[str, Any]) -> dict[str, Any]:
    item = _merge_gs1_text(item)
    evidence = _evidence_text(item)
    sn = _clean_str(item.get("serial_number"))
    batch = _clean_str(item.get("batch_no") or item.get("batch"))
    lot = _clean_str(item.get("lot"))
    gtin = _clean_str(item.get("gtin"))
    mfg = _clean_str(item.get("mfg_date"))
    exp = _clean_str(item.get("exp_date"))

    if evidence:
        if gtin and not _supported_by_evidence(gtin, evidence):
            gtin = None
        if batch and not _supported_by_evidence(batch, evidence):
            batch = None
        if lot and not _supported_by_evidence(lot, evidence):
            lot = None
        if sn and not _supported_by_evidence(sn, evidence):
            sn = None
        if mfg and not _supported_by_evidence(mfg, evidence, is_date=True):
            mfg = None
        if exp and not _supported_by_evidence(exp, evidence, is_date=True):
            exp = None

        if not lot:
            lot = _extract_labeled_value(
                evidence,
                (
                    r"\bLOT(?:\s*NO\.?)?[#:\s.]*([A-Z0-9][A-Z0-9\-/]{2,})",
                    r"\bLOT\s*[#:]\s*([A-Z0-9][A-Z0-9\-/]{2,})",
                ),
            )
        if not batch:
            batch = _extract_labeled_value(
                evidence,
                (
                    r"\bB/?N\s*[#:=\s]\s*([A-Z0-9][A-Z0-9\-/]{2,})",
                    r"\bB\.N\.?\s*[#:=\s]\s*([A-Z0-9][A-Z0-9\-/]{2,})",
                    r"\bB\.?N\.?O?\.?\s*[#:=]\s*([A-Z0-9][A-Z0-9\-/]{2,})",
                    r"\bBATCH(?:\s*(?:NO\.?|NUMBER|CODE))?\s*[#:=.\s]\s*([A-Z0-9][A-Z0-9\-/]{2,})",
                ),
            )
        if not exp:
            exp = _extract_labeled_value(
                evidence,
                (
                    r"\bEXP(?:IRY)?(?:\s*DATE\.?)?[#:\s.]*([0-9]{1,4}[\s\-/\.][0-9]{1,4}(?:[\s\-/\.][0-9]{2,4})?)",
                    r"\bCAD[#:\s.]*([0-9]{1,4}[\s\-/\.][0-9]{1,4}(?:[\s\-/\.][0-9]{2,4})?)",
                    r"\bUSE\s+BY[#:\s.]*([0-9]{1,4}[\s\-/\.][0-9]{1,4}(?:[\s\-/\.][0-9]{2,4})?)",
                ),
            )
        if not mfg:
            mfg = _extract_labeled_value(
                evidence,
                (
                    r"\bMFG(?:\.?\s*DATE\.?)?[#:\s.]*([0-9]{1,4}[\s\-/\.][0-9]{1,4}(?:[\s\-/\.][0-9]{2,4})?)",
                    r"\bMFD[#:\s.]*([0-9]{1,4}[\s\-/\.][0-9]{1,4}(?:[\s\-/\.][0-9]{2,4})?)",
                    r"\bPRO[#:\s.]*([0-9]{1,4}[\s\-/\.][0-9]{1,4}(?:[\s\-/\.][0-9]{2,4})?)",
                    r"\bDATE\s+OF\s+MANUFACTURE[#:\s.]*([0-9]{1,4}[\s\-/\.][0-9]{1,4}(?:[\s\-/\.][0-9]{2,4})?)",
                ),
            )
        if not gtin:
            ai_gtin = re.search(r"\(01\)\s*(\d{8,14})", evidence)
            if ai_gtin:
                gtin = ai_gtin.group(1)
        if (
            lot
            and not batch
            and re.search(r"\bbatch\s*code\b", evidence, re.I)
            and _supported_by_evidence(lot, evidence)
        ):
            batch = lot

    if sn and _looks_like_date(sn):
        sn = None

    if sn and batch and sn == batch:
        sn = None
    if sn and lot and sn == lot:
        sn = None
    if gtin and sn and re.sub(r"\D", "", gtin) == re.sub(r"\D", "", sn):
        sn = None

    if gtin and gtin.startswith("+"):
        gtin = None
    if batch and _compact(batch) in INVALID_FIELD_VALUES:
        batch = None
    if lot and _compact(lot) in INVALID_FIELD_VALUES:
        lot = None

    mfg_n = normalize_date(mfg) if mfg else None
    exp_n = normalize_date(exp) if exp else None
    if mfg_n and exp_n and mfg_n > exp_n:
        mfg_n, exp_n = exp_n, mfg_n
    if mfg_n and exp_n and mfg_n == exp_n:
        mfg_n = None

    item["serial_number"] = sn
    item["batch_no"] = batch
    item["lot"] = lot
    item["gtin"] = gtin
    item["mfg_date"] = mfg_n
    item["exp_date"] = exp_n
    return item


def _record_from_dict(item: dict[str, Any]) -> MedicineRecord:
    item = _repair_record(item)
    batch = item.get("batch_no")
    lot = item.get("lot")
    gtin = normalize_gtin(str(item["gtin"]), require_check=False) if item.get("gtin") else None
    filled = sum(
        1
        for v in (gtin, batch, lot, item.get("mfg_date"), item.get("exp_date"), item.get("serial_number"))
        if v
    )
    return MedicineRecord(
        gtin=gtin,
        batch_no=str(batch) if batch else None,
        lot=str(lot) if lot else None,
        mfg_date=item.get("mfg_date"),
        exp_date=item.get("exp_date"),
        serial_number=str(item["serial_number"]) if item.get("serial_number") else None,
        extraction_method="vision_llm",
        confidence=min(0.95, 0.5 + 0.08 * filled),
        source_fields={
            k: "llm"
            for k in ("gtin", "batch_no", "lot", "mfg_date", "exp_date", "serial_number")
            if (k == "gtin" and gtin) or (k != "gtin" and item.get(k))
        },
    )


def _get_llm_settings() -> tuple[str, str, str | None, int]:
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).rstrip("/")
    api_key = os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY)
    model = os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL
    seed = int(os.getenv("LLM_SEED", str(DEFAULT_LLM_SEED)))
    return base_url, api_key, model, seed


def _create_client() -> OpenAI:
    base_url, api_key, _, _ = _get_llm_settings()
    return OpenAI(base_url=base_url, api_key=api_key)


@lru_cache(maxsize=1)
def _resolve_model(base_url: str, api_key: str, configured_model: str | None) -> str:
    if configured_model:
        return configured_model

    client = OpenAI(base_url=base_url, api_key=api_key)
    models = client.models.list()
    if not models.data:
        raise ValueError(f"No models available from LLM server at {base_url}.")
    return models.data[0].id


def _call_vision_llm(
    client: OpenAI,
    model: str,
    image_bytes: bytes,
    mime: str,
    user_text: str,
    seed: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": _image_to_data_url(image_bytes, mime)}},
            ],
        },
    ]
    request = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 4096,
        "seed": seed,
        "response_format": {"type": "json_object"},
    }
    completion = client.chat.completions.create(**request)
    content = completion.choices[0].message.content or "{}"
    try:
        return _parse_llm_json(content)
    except json.JSONDecodeError:
        return {}


def _payload_to_records(payload: dict[str, Any]) -> list[MedicineRecord]:
    medicines = payload.get("medicines", [])
    if isinstance(medicines, dict):
        medicines = [medicines]
    return [_record_from_dict(item) for item in medicines if isinstance(item, dict)]


def extract_with_vision_llm(image_bytes: bytes, mime: str = "image/jpeg") -> list[MedicineRecord]:
    base_url, api_key, configured_model, seed = _get_llm_settings()
    if not base_url:
        raise ValueError("LLM_BASE_URL is not set in environment.")
    if not api_key:
        raise ValueError("LLM_API_KEY is not set in environment.")

    image_bytes, mime = _prepare_image(image_bytes, mime)
    client = _create_client()
    model = _resolve_model(base_url, api_key, configured_model)

    payload = _call_vision_llm(client, model, image_bytes, mime, USER_TEXT, seed)
    records = _payload_to_records(payload)

    if not records:
        retry_payload = _call_vision_llm(client, model, image_bytes, mime, RETRY_PROMPT, seed)
        records = _payload_to_records(retry_payload)

    return records
