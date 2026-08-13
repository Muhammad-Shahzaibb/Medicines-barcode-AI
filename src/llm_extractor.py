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

DEFAULT_LLM_BASE_URL = "http://101.44.222.84:8000/v1"
DEFAULT_LLM_API_KEY = "dummy"
DEFAULT_LLM_SEED = 42
MAX_IMAGE_SIDE = 1536
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
    "UDI",
    "GTIN",
    "BATCH",
    "BATCHCODE",
    "LOT",
    "CODE",
    "MFG",
    "MFD",
    "EXP",
}

SYSTEM_PROMPT = """You are a pharmaceutical and medical-device packaging data extraction expert.
Extract ONLY text that is clearly printed on the image. Never guess, infer, or complete missing characters.

OUTPUT — return ONLY valid JSON:
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

ANTI-HALLUCINATION (highest priority):
1. If a field is not clearly printed, return null. Blurry, cut-off, glare-obscured, or uncertain text → null.
2. Do not invent digits, leading zeros, dates, GTINs, serials, batch, or lot values.
3. Do not copy a value into another field. CAT/REF/product code is not GTIN. LOT is not batch unless a batch label is also present. LOT is not serial. LOT is not a date field.
4. Do not use manufacturer names, addresses, sizes, quantities, websites, PL/license numbers, or HIBC codes as any output field.
5. If you cannot read a character with certainty (0 vs O, 1 vs I, 5 vs S), return null for that whole field.

CORE RULES:
1. One object per distinct medicine/unit label visible in the photo.
2. Normalize dates to YYYY-MM-DD. Month-year only → 1st of month (09-2026 → 2026-09-01). Keep the printed year/month; do not invent a day other than 01 when day is absent.
3. batch_no and lot are separate. Fill a field only if that label (or GS1 AI) is visible.
4. Read rotated/vertical text. Preserve leading zeros exactly as printed.

GTIN — extract when a product identifier is clearly printed:
  - GS1 `(01)` followed by 8–14 digits, even in small/thin font above a 2D barcode.
    Example: (01)16975486451862 → gtin=16975486451862. Always copy this when present.
  - Labels: GTIN, GTN, EAN, or a 13/14-digit numeric code directly under a product barcode.
  NOT GTIN: CAT NO, REF, catalog/SKU (KM-DM033, KM-QP020), HIBC codes starting with +, PL/license numbers,
    or alphanumeric UDI text such as 697548645101PP. The UDI box is not the GTIN unless it is a numeric (01) value.
  If only HIBC, CAT/REF, or alphanumeric UDI is visible and there is no (01)/GTIN/EAN, gtin = null.

BATCH (batch_no): only Batch, BNO, B.NO, BN, B/N, Batch No, Batch Number, Batch Code.
  If that label is absent, batch_no = null even when a LOT value exists.

LOT (lot): only LOT, Lot, Lot No, LOT NO., Lote, (10), [LOT] box symbol.
  If that label is absent, lot = null even when a Batch/BN value exists.
  If BOTH LOT and Batch/Batch Code labels point to the same printed value, set both to that value.
  An 8-digit YYYYMMDD beside [LOT] is a lot number, NOT mfg_date and NOT serial_number.

SERIAL: only SN, SNO, Serial, (21). Unique per unit.
  Do not put expiry, lot, batch, REF, or GTIN into serial_number.

MFG DATE: factory/building icon (ISO 15223) OR MFG, MFD, MD, MFG.DATE, PRO, (11), P:
EXP DATE: hourglass icon (ISO 15223) OR EXP, EXP.DATE, EXPIRY DATE, CAD, (17)
  Factory icon = mfg_date. Hourglass = exp_date. Never swap them.
  Do not derive mfg_date from a lot/batch number even if that number looks like a date.

DATE FORMATS: YYYYMMDD, YYMMDD, YYYY-MM, YYYY-MM-DD, MM/YYYY, MM-YYYY, MM YYYY, DD/MM/YYYY, YYYY MM DD.
  20250415 → 2025-04-15 | 2029-04 → 2029-04-01 | 12/2026 → 2026-12-01 | 09-2026 → 2026-09-01 | 07 - 2023 → 2023-07-01 | 30/11/2028 → 2028-11-30

GS1 HUMAN-READABLE — copy these strings into gs1_text EXACTLY as printed, including parentheses:
  (01)=GTIN  (10)=lot  (11)=mfg YYMMDD  (17)=exp YYMMDD  (21)=serial
  Example gs1_text: "(01)06975486453029 (11)250415(17)300414(10)KM2503173"
  If no parenthesized GS1 text is visible, gs1_text = null. Do not invent AIs or digits.

VISIBLE TEXT — list the exact printed lines/values you used (LOT/BN/MFG/EXP/GS1/SN/GTIN). Copy characters exactly. Do not add lines that are not in the image.

MULTI-BOX: shared PC/GTIN/Lot/EXP is allowed only if printed on each box or clearly the same pack family; each box keeps its own SN. Do not invent extra boxes.

ACCURACY CHECKLIST:
- Prefer null over a guessed character
- 0 vs 7, 0 vs O, 1 vs I, 5 vs S, 8 vs B
- Leading zeros preserved
- Serial is not a date, lot, or GTIN
- batch_no is null when only LOT is printed
- lot is null when only BN/Batch is printed
- mfg_date and exp_date are not swapped
"""

USER_TEXT = (
    "Extract traceability fields that are clearly printed on this image. "
    "First copy the exact LOT/BATCH/MFG/EXP/GTIN/SN/GS1 lines into visible_text. "
    "Copy every visible GS1 string such as (01)/(10)/(11)/(17)/(21) into gs1_text exactly. "
    "Every output field must appear in visible_text or gs1_text. If a field is missing or unreadable, return null. "
    "Do not guess. Return JSON only."
)

RETRY_PROMPT = (
    "The previous response was not valid JSON or contained no records. "
    "Read only clearly visible printed values. Use null for missing fields. "
    "Do not guess. Return JSON only with the medicines array."
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
                (r"\bLOT(?:\s*NO\.?)?[#:\s.]*([A-Z0-9][A-Z0-9\-/]{2,})",),
            )
        if not batch:
            batch = _extract_labeled_value(
                evidence,
                (
                    r"\bB/?N\s*[#:=]\s*([A-Z0-9][A-Z0-9\-/]{2,})",
                    r"\bB\.?N\.?O?\.?\s*[#:=]\s*([A-Z0-9][A-Z0-9\-/]{2,})",
                    r"\bBATCH(?:\s*(?:NO\.?|NUMBER))?\s*[#:=]\s*([A-Z0-9][A-Z0-9\-/]{2,})",
                ),
            )
        if not exp:
            exp = _extract_labeled_value(
                evidence,
                (
                    r"\bEXP(?:IRY)?(?:\s*DATE)?[#:\s.]*([0-9]{1,4}\s*[-/]\s*[0-9]{1,4}(?:\s*[-/]\s*[0-9]{2,4})?)",
                    r"\bCAD[#:\s.]*([0-9]{1,4}\s*[-/]\s*[0-9]{1,4}(?:\s*[-/]\s*[0-9]{2,4})?)",
                ),
            )
        if not mfg:
            mfg = _extract_labeled_value(
                evidence,
                (
                    r"\bMFG(?:\.?\s*DATE)?[#:\s.]*([0-9]{1,4}\s*[-/]\s*[0-9]{1,4}(?:\s*[-/]\s*[0-9]{2,4})?)",
                    r"\bMFD[#:\s.]*([0-9]{1,4}\s*[-/]\s*[0-9]{1,4}(?:\s*[-/]\s*[0-9]{2,4})?)",
                    r"\bPRO[#:\s.]*([0-9]{1,4}\s*[-/]\s*[0-9]{1,4}(?:\s*[-/]\s*[0-9]{2,4})?)",
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
    model = os.getenv("LLM_MODEL") or None
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
        "max_tokens": 2048,
        "seed": seed,
        "response_format": {"type": "json_object"},
        "extra_body": {
            "seed": seed,
            "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    try:
        completion = client.chat.completions.create(**request)
    except Exception:
        request.pop("extra_body", None)
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
