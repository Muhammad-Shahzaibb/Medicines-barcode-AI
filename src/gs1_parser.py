from __future__ import annotations

import re

from src.models import MedicineRecord
from src.normalizer import normalize_date, normalize_gtin

GS_SEPARATOR = "\x1d"

FIXED_LENGTH_AIS: dict[str, int] = {
    "00": 18,
    "01": 14,
    "02": 14,
    "11": 6,
    "12": 6,
    "13": 6,
    "15": 6,
    "16": 6,
    "17": 6,
    "20": 2,
}

VARIABLE_LENGTH_AIS = {"10", "21", "22", "240", "241", "242", "250", "251", "253", "254", "255"}

ALL_AIS = set(FIXED_LENGTH_AIS) | VARIABLE_LENGTH_AIS


def _ai_length(ai: str) -> int:
    return len(ai)


def _parse_parenthesized(raw: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"\((\d{2,4})\)", raw))
    pairs: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        ai = match.group(1)
        value_start = match.end()
        value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        pairs.append((ai, raw[value_start:value_end]))
    return pairs


def _read_variable_field(data: str, start: int, following_ais: tuple[str, ...]) -> tuple[str, int]:
    if start >= len(data):
        return "", start

    for j in range(start, len(data)):
        if data[j] == GS_SEPARATOR:
            return data[start:j], j + 1
        for ai in following_ais:
            if data.startswith(ai, j):
                return data[start:j], j

    return data[start:], len(data)


def _parse_concatenated(raw: str) -> list[tuple[str, str]]:
    data = raw.replace(GS_SEPARATOR, "")
    if not data.startswith("01"):
        return []

    pos = 2
    pairs: list[tuple[str, str]] = []
    pairs.append(("01", data[pos : pos + 14]))
    pos += 14

    while pos < len(data):
        ai = None
        for ai_len in (4, 3, 2):
            candidate = data[pos : pos + ai_len]
            if candidate in ALL_AIS:
                ai = candidate
                break
        if not ai:
            break

        pos += _ai_length(ai)

        if ai in FIXED_LENGTH_AIS:
            length = FIXED_LENGTH_AIS[ai]
            pairs.append((ai, data[pos : pos + length]))
            pos += length
        elif ai == "10":
            value, pos = _read_variable_field(data, pos, ("21", "22"))
            pairs.append((ai, value))
        elif ai == "21":
            pairs.append((ai, data[pos:]))
            break
        else:
            value, pos = _read_variable_field(data, pos, tuple(sorted(ALL_AIS, key=len, reverse=True)))
            pairs.append((ai, value))

    return pairs


def _tokenize(raw: str) -> list[tuple[str, str]]:
    text = raw.strip()
    if re.search(r"\(\d{2,4}\)", text):
        return _parse_parenthesized(text)
    if GS_SEPARATOR in text:
        chunks = [chunk for chunk in text.split(GS_SEPARATOR) if chunk]
        pairs: list[tuple[str, str]] = []
        for chunk in chunks:
            pairs.extend(_parse_concatenated(chunk) if chunk.startswith("01") else [])
        if pairs:
            return pairs
    if text.startswith("01"):
        return _parse_concatenated(text)
    return []


def parse_gs1(raw: str) -> MedicineRecord:
    record = MedicineRecord(extraction_method="gs1_barcode", confidence=0.95, raw_barcode=raw)

    for ai, value in _tokenize(raw):
        value = value.strip()
        if ai == "01":
            record.gtin = normalize_gtin(value)
            record.source_fields["gtin"] = "barcode"
        elif ai == "10":
            record.batch_no = value
            record.lot = value
            record.source_fields["batch_no"] = "barcode"
        elif ai == "11":
            record.mfg_date = normalize_date(value)
            record.source_fields["mfg_date"] = "barcode"
        elif ai == "17":
            record.exp_date = normalize_date(value)
            record.source_fields["exp_date"] = "barcode"
        elif ai == "21":
            record.serial_number = value
            record.source_fields["serial_number"] = "barcode"

    if record.gtin:
        record.confidence = min(0.99, 0.7 + 0.05 * record.filled_count())
    return record


def is_gs1_payload(text: str) -> bool:
    if GS_SEPARATOR in text:
        return True
    if re.search(r"\(\d{2,4}\)", text):
        return True
    return bool(re.match(r"^01\d{14}", text))
