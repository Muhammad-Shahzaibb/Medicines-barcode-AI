from __future__ import annotations

import re

from src.models import MedicineRecord
from src.normalizer import normalize_date, normalize_gtin, normalize_label

try:
    import easyocr

    HAS_EASYOCR = True
    _reader: easyocr.Reader | None = None
except ImportError:
    HAS_EASYOCR = False
    _reader = None

HUMAN_LABEL_PATTERNS: list[tuple[str, str]] = [
    ("gtin", r"\b(?:GTIN|GTN)\b\s*[:\.]?\s*(\d[\d\s]{7,20})"),
    ("sn", r"\b(?:SN|SNO|Serial(?:\s*No\.?)?)\b\s*[:\.]?\s*([A-Z0-9]{6,20})"),
    ("batch", r"\b(?:BNO|B\.NO|B\.N|BN|Batch(?:\s*No\.?)?)\b\s*[:\.]?\s*([A-Z0-9\-/]+)"),
    ("lot", r"\b(?:LOT|Lote|Lot(?:\s*No\.?)?)\b\s*[:\.]?\s*([A-Z0-9\-/]+)"),
    ("mfg", r"\b(?:MFG|MFD|MD|Manuf(?:acturing)?(?:\s*Date)?|MFG\.DATE)\b\s*[:\.]?\s*(\d{8})"),
    ("mfg", r"\b(?:MFG|MFD|MD)\b\s*[:\.]?\s*(\d{1,2}\s+\d{4})"),
    ("mfg", r"\b(?:MFG|MFD|MD)\b\s*[:\.]?\s*([A-Z]{3}/\d{4})"),
    ("mfg", r"\bP:\s*(\d{1,2}/\d{1,2}/\d{4})"),
    ("exp", r"\b(?:EXP|CAD|Expir(?:y|ation)(?:\s*Date)?|EXP\.DATE)\b\s*[:\.]?\s*(\d{8})"),
    ("exp", r"\b(?:EXP|CAD)\b\s*[:\.]?\s*(\d{1,2}\s+\d{4})"),
    ("exp", r"\b(?:EXP|CAD)\b\s*[:\.]?\s*([A-Z]{3}/\d{4})"),
    ("exp", r"\b(?:EXP|CAD)\b\s*[:\.]?\s*(\d{1,2}-\d{4})"),
    ("mfg", r"\b(\d{4}-\d{2}-\d{2})\b(?=.*(?:factory|mfg|mfd|manuf))"),
]


def _get_reader() -> "easyocr.Reader":
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en", "ar"], gpu=False, verbose=False)
    return _reader


def _extract_gs1_from_text(text: str) -> MedicineRecord | None:
    from src.gs1_parser import parse_gs1

    compact = re.sub(r"\s+", "", text)
    patterns = [
        r"\(01\)\d{14}.*?(?=\s|$)",
        r"\(01\)\d{8,14}(?:\(1[017]\)\d{6})?(?:\(10\)[^(\s]+)?(?:\(21\)[^(\s]+)?",
        r"01\d{14}(?:17\d{6})?(?:10[^21]+)?(?:21.+)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact)
        if match:
            record = parse_gs1(match.group(0))
            if record.gtin or record.batch_no or record.exp_date:
                record.extraction_method = "ocr"
                record.raw_ocr = text
                return record
    return None


def _assign_field(record: MedicineRecord, label: str, value: str) -> None:
    value = value.strip(" -")
    if label == "gtin":
        record.gtin = normalize_gtin(value)
        record.source_fields["gtin"] = "ocr"
    elif label == "batch":
        record.batch_no = value
        record.source_fields["batch_no"] = "ocr"
    elif label == "lot":
        record.lot = value
        record.source_fields["lot"] = "ocr"
    elif label == "mfg":
        record.mfg_date = normalize_date(value.split()[0] if " " in value else value)
        record.source_fields["mfg_date"] = "ocr"
    elif label == "exp":
        record.exp_date = normalize_date(value.split()[0] if " " in value else value)
        record.source_fields["exp_date"] = "ocr"
    elif label == "sn":
        record.serial_number = value
        record.source_fields["serial_number"] = "ocr"


def _parse_human_labels(text: str) -> MedicineRecord:
    record = MedicineRecord(extraction_method="ocr", confidence=0.55, raw_ocr=text)
    compact = re.sub(r"\s+", " ", text)

    for label, pattern in HUMAN_LABEL_PATTERNS:
        found = re.search(pattern, compact, re.I)
        if found:
            _assign_field(record, label, found.group(1).strip())

    for line in text.splitlines():
        match = re.match(r"^([A-Za-z.#\s]+?)[:]\s*(.+)$", line.strip())
        if not match:
            continue
        label = normalize_label(match.group(1))
        _assign_field(record, label, match.group(2).strip())

    if record.gtin or record.batch_no or record.lot or record.mfg_date or record.exp_date:
        record.confidence = min(0.85, 0.45 + 0.08 * record.filled_count())
    return record


def _merge_ocr_parts(*parts: MedicineRecord | None) -> MedicineRecord | None:
    records = [r for r in parts if r and r.filled_count() > 0]
    if not records:
        return None

    merged = MedicineRecord(extraction_method="ocr", raw_ocr=records[0].raw_ocr)
    for record in records:
        merged.gtin = merged.gtin or record.gtin
        merged.batch_no = merged.batch_no or record.batch_no
        merged.lot = merged.lot or record.lot
        merged.mfg_date = merged.mfg_date or record.mfg_date
        merged.exp_date = merged.exp_date or record.exp_date
        merged.serial_number = merged.serial_number or record.serial_number
        merged.source_fields.update(record.source_fields)
        merged.raw_ocr = merged.raw_ocr or record.raw_ocr

    merged.confidence = min(0.85, 0.45 + 0.08 * merged.filled_count())
    return merged


def _parse_key_value_lines(text: str) -> MedicineRecord:
    gs1_record = _extract_gs1_from_text(text)
    human_record = _parse_human_labels(text)
    merged = _merge_ocr_parts(gs1_record, human_record)
    return merged or MedicineRecord(extraction_method="ocr", raw_ocr=text)


def extract_from_ocr(image_bgr) -> list[MedicineRecord]:
    if not HAS_EASYOCR:
        return []

    reader = _get_reader()
    results = reader.readtext(image_bgr)
    if not results:
        return []

    lines: dict[int, list[str]] = {}
    for bbox, text, _score in results:
        y = int((bbox[0][1] + bbox[2][1]) / 2)
        bucket = (y // 40) * 40
        lines.setdefault(bucket, []).append((bbox[0][0], text))

    region_texts: list[str] = []
    for bucket in sorted(lines):
        row = " ".join(text for _, text in sorted(lines[bucket], key=lambda x: x[0]))
        region_texts.append(row)

    full_text = "\n".join(region_texts)
    all_text = " ".join(text for _, text, _ in results)

    record = _merge_ocr_parts(
        _parse_key_value_lines(full_text),
        _parse_human_labels(all_text),
    )
    return [record] if record and record.filled_count() > 0 else []
