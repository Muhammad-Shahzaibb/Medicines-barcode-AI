from __future__ import annotations

import base64
import json
import os
import re
from functools import lru_cache
from typing import Any

from openai import OpenAI

from src.models import MedicineRecord
from src.normalizer import normalize_date, normalize_gtin

DEFAULT_LLM_BASE_URL = "http://101.44.222.84:8000/v1"
DEFAULT_LLM_API_KEY = "dummy"

SYSTEM_PROMPT = """You are a pharmaceutical and medical-device packaging data extraction expert.
Analyze label images and return structured traceability data with maximum accuracy.

OUTPUT — return ONLY valid JSON:
{
  "medicines": [
    {
      "gtin": string or null,
      "batch_no": string or null,
      "lot": string or null,
      "mfg_date": string or null,
      "exp_date": string or null,
      "serial_number": string or null
    }
  ]
}

CORE RULES:
1. One object per distinct medicine/unit label. Multiple boxes in one photo → multiple objects.
2. Use null when a field is genuinely absent — never guess.
3. Normalize dates to YYYY-MM-DD. Month-year only → use 1st of month (e.g. 09-2026 → 2026-09-01).
4. batch_no and lot are separate fields — never copy one into the other.
5. Use null for batch_no or lot when that specific label/value is not visible on the packaging.
6. Read rotated/vertical text carefully. Preserve leading zeros (00603113 not 70603113).

GTIN — ONLY use values explicitly labeled or encoded as product identifier:
  GTIN, GTN, EAN, UDI, PC: (Product Code), (01), 14-digit code under barcode.
  Do NOT use: CAT NO, REF, LOT, SN/Serial, HIBC codes (+H78476H0L), or license numbers (PL xxxxx).
  HIBC barcodes start with "+" — that is NOT a GTIN. Leave gtin null if only HIBC is visible.

BATCH (batch_no): Batch, BNO, B.NO, BN, Batch No, Batch Number.
  Put the value in batch_no only. If no batch label is visible, set batch_no to null.

LOT (lot): LOT, Lot, Lot No, LOT NO., Lote, (10), [LOT] box symbol.
  Put the value in lot only. If no lot label is visible, set lot to null.
  Lot numbers can be numeric (00603113, KM2503173) or alphanumeric (P2500043).
  An 8-digit YYYYMMDD beside [LOT] is a lot number, NOT a serial number.
  Do NOT duplicate the same value into both batch_no and lot unless both labels appear with the same value.

SERIAL: SN, SNO, Serial, (21) — unique per unit. On multi-box photos each box has a different SN.
  Do NOT put expiry dates or lot numbers in serial_number.
  12-digit numbers labeled SN: are serial numbers, NOT GTIN.

MFG DATE — factory/building icon (ISO 15223) OR text: MFG, MFD, MD, MFG.DATE, (11), P:
EXP DATE — hourglass icon (ISO 15223) OR text: EXP, EXP.DATE, CAD, (17)
  CRITICAL: factory icon date = mfg_date. hourglass icon date = exp_date. Never swap them.
  When three 8-digit dates appear with symbols: [LOT]=lot, factory=mfg, hourglass=exp.

DATE FORMATS: YYYYMMDD, YYMMDD, YYYY-MM, YYYY-MM-DD, MM/YYYY, MM-YYYY, MM YYYY, DD/MM/YYYY.
  Examples: 20250415 → 2025-04-15 | 2029-04 → 2029-04-01 | 12/2026 → 2026-12-01 | 09-2026 → 2026-09-01

GS1 HUMAN-READABLE (when printed near barcode):
  (01)=GTIN (02)=(10)=lot (11)=mfg YYMMDD (17)=exp YYMMDD (21)=serial

MULTI-BOX IMAGES: All boxes often share the same PC:/GTIN, Lot, and EXP but each has a unique SN.
  Return one object per visible box with its own serial_number.

DENSE LABELS: Read the full label including text above/below barcodes. Extract GTIN from (01) even on crowded labels.

ACCURACY CHECKLIST before responding:
- 0 vs 7, 0 vs O, 1 vs I, 5 vs S, 8 vs B
- Leading zeros in lot numbers
- Serial numbers are not dates
- GTIN is not confused with serial number on multi-unit photos
- mfg_date and exp_date are not swapped
"""

RETRY_PROMPT = """The previous extraction failed or was incomplete.
Re-examine this image very carefully. Pay special attention to:
- Rotated/vertical text (LOT, EXP on bottle labels)
- ISO symbols: factory icon = mfg_date, hourglass = exp_date
- PC: label = GTIN (product code), SN: = serial_number (unique per box)
- (01) GTIN and (10)/(11)/(17) GS1 strings on crowded labels
- Leading zeros in lot numbers
- HIBC codes starting with + are NOT gtin
Return JSON only with the medicines array."""


def _image_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _parse_llm_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _looks_like_date(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if len(digits) == 8:
        year = int(digits[:4])
        return 1990 <= year <= 2045
    return False


def _repair_record(item: dict[str, Any]) -> dict[str, Any]:
    sn = str(item["serial_number"]).strip() if item.get("serial_number") else None
    batch = str(item.get("batch_no") or item.get("batch") or "").strip() or None
    lot = str(item.get("lot") or "").strip() or None
    gtin = str(item["gtin"]).strip() if item.get("gtin") else None

    if sn and _looks_like_date(sn) and not item.get("exp_date"):
        item["exp_date"] = normalize_date(sn)
        item["serial_number"] = None

    if sn and batch and sn == batch:
        item["serial_number"] = None
    if sn and lot and sn == lot:
        item["serial_number"] = None

    if gtin:
        gtin_digits = re.sub(r"\D", "", gtin)
        if len(gtin_digits) == 12 and not gtin_digits.startswith("0"):
            if item.get("serial_number") is None:
                item["serial_number"] = gtin_digits
            item["gtin"] = None

    mfg = item.get("mfg_date")
    exp = item.get("exp_date")
    if mfg and exp:
        mfg_n = normalize_date(str(mfg))
        exp_n = normalize_date(str(exp))
        if mfg_n and exp_n and mfg_n > exp_n:
            item["mfg_date"], item["exp_date"] = exp_n, mfg_n

    for date_candidate in (batch, lot):
        if date_candidate and _looks_like_date(date_candidate) and item.get("exp_date") and not item.get("mfg_date"):
            candidate_date = normalize_date(date_candidate)
            exp_date = normalize_date(str(item["exp_date"]))
            if candidate_date and exp_date and candidate_date < exp_date:
                item["mfg_date"] = candidate_date
                break

    item["batch_no"] = batch
    item["lot"] = lot
    return item


def _record_from_dict(item: dict[str, Any]) -> MedicineRecord:
    item = _repair_record(item)
    batch = item.get("batch_no")
    lot = item.get("lot")
    filled = sum(
        1
        for v in (item.get("gtin"), batch, lot, item.get("mfg_date"), item.get("exp_date"), item.get("serial_number"))
        if v
    )
    return MedicineRecord(
        gtin=normalize_gtin(str(item["gtin"])) if item.get("gtin") else None,
        batch_no=str(batch) if batch else None,
        lot=str(lot) if lot else None,
        mfg_date=normalize_date(str(item["mfg_date"])) if item.get("mfg_date") else None,
        exp_date=normalize_date(str(item["exp_date"])) if item.get("exp_date") else None,
        serial_number=str(item["serial_number"]) if item.get("serial_number") else None,
        extraction_method="vision_llm",
        confidence=min(0.95, 0.5 + 0.08 * filled),
        source_fields={
            k: "llm"
            for k in ("gtin", "batch_no", "lot", "mfg_date", "exp_date", "serial_number")
            if item.get(k)
        },
    )


def _get_llm_settings() -> tuple[str, str, str | None]:
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).rstrip("/")
    api_key = os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY)
    model = os.getenv("LLM_MODEL") or None
    return base_url, api_key, model


def _create_client() -> OpenAI:
    base_url, api_key, _ = _get_llm_settings()
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


def _call_vision_llm(client: OpenAI, model: str, image_bytes: bytes, mime: str, user_text: str) -> dict[str, Any]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image_bytes, mime)}},
                ],
            },
        ],
        temperature=0,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content or "{}"
    return _parse_llm_json(content)


def _payload_to_records(payload: dict[str, Any]) -> list[MedicineRecord]:
    medicines = payload.get("medicines", [])
    if isinstance(medicines, dict):
        medicines = [medicines]
    return [_record_from_dict(item) for item in medicines if isinstance(item, dict)]


def _needs_retry(records: list[MedicineRecord]) -> bool:
    if not records:
        return True
    for record in records:
        if record.serial_number and _looks_like_date(record.serial_number):
            return True
        if record.gtin and record.serial_number and re.sub(r"\D", "", record.gtin) == re.sub(
            r"\D", "", record.serial_number
        ):
            return True
        if record.mfg_date and record.exp_date and record.mfg_date > record.exp_date:
            return True
    return False


def extract_with_vision_llm(image_bytes: bytes, mime: str = "image/jpeg") -> list[MedicineRecord]:
    base_url, api_key, configured_model = _get_llm_settings()
    if not base_url:
        raise ValueError("LLM_BASE_URL is not set in environment.")
    if not api_key:
        raise ValueError("LLM_API_KEY is not set in environment.")

    client = _create_client()
    model = _resolve_model(base_url, api_key, configured_model)
    user_text = (
        "Extract all medicine traceability records from this image. "
        "Read every label, symbol, barcode, and rotated text. Return JSON only."
    )

    payload = _call_vision_llm(client, model, image_bytes, mime, user_text)
    records = _payload_to_records(payload)

    if _needs_retry(records):
        retry_payload = _call_vision_llm(client, model, image_bytes, mime, RETRY_PROMPT)
        retry_records = _payload_to_records(retry_payload)
        if len(retry_records) >= len(records):
            records = retry_records

    return records
