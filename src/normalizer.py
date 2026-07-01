from __future__ import annotations

import re
from datetime import datetime

MONTH_MAP = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
    "ene": "01",
    "feb": "02",
    "mar": "03",
    "abr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "ago": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dic": "12",
}

FIELD_SYNONYMS: dict[str, list[str]] = {
    "gtin": ["gtin", "gtn", "ean", "udi"],
    "batch": ["batch", "bno", "b.no", "b.n", "bn", "batch no", "batch no.", "batch number"],
    "lot": ["lot", "lote", "lot no", "lot no.", "lot number"],
    "mfg": [
        "mfg",
        "mfd",
        "md",
        "manuf",
        "manufacturing",
        "mfg date",
        "mfg.date",
        "mfg data",
        "manufacturing date",
        "manufacturing data",
        "p",
        "prod",
        "production",
    ],
    "exp": ["exp", "expiry", "expire", "expiration", "exp date", "exp.date", "cad", "caducidad"],
    "sn": ["sn", "sno", "sr", "sr#", "serial", "serial no", "serial number"],
}


def normalize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9#.\s]", "", label.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    for canonical, synonyms in FIELD_SYNONYMS.items():
        if cleaned in synonyms or cleaned.rstrip(".:") in synonyms:
            return canonical
    return cleaned


def normalize_gtin(value: str | None) -> str | None:
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) < 8:
        return None
    if len(digits) == 13:
        digits = "0" + digits
    return digits if len(digits) in (8, 12, 13, 14) else digits


def normalize_date(value: str | None) -> str | None:
    if not value:
        return None

    text = value.strip()
    text = re.sub(r"\s+", " ", text)

    if re.fullmatch(r"\d{6}", text):
        yy, mm, dd = int(text[:2]), int(text[2:4]), int(text[4:6])
        year = 2000 + yy if yy < 80 else 1900 + yy
        if dd == 0:
            return f"{year:04d}-{mm:02d}-01"
        return f"{year:04d}-{mm:02d}-{dd:02d}"

    if re.fullmatch(r"\d{8}", text):
        yyyy, mm, dd = int(text[:4]), int(text[4:6]), int(text[6:8])
        return f"{yyyy:04d}-{mm:02d}-{dd:02d}"

    month_year = re.match(r"^([A-Za-z]{3,9})[/\-\s]+(\d{4})$", text, re.I)
    if month_year:
        month_key = month_year.group(1).lower()[:3]
        month = MONTH_MAP.get(month_key)
        if month:
            return f"{month_year.group(2)}-{month}-01"

    mm_yyyy = re.match(r"^(\d{1,2})\s+(\d{4})$", text)
    if mm_yyyy:
        return f"{int(mm_yyyy.group(2)):04d}-{int(mm_yyyy.group(1)):02d}-01"

    mm_yyyy_dash = re.match(r"^(\d{1,2})\s*-\s*(\d{4})$", text)
    if mm_yyyy_dash:
        return f"{int(mm_yyyy_dash.group(2)):04d}-{int(mm_yyyy_dash.group(1)):02d}-01"

    mm_slash_yyyy = re.match(r"^(\d{1,2})/(\d{4})$", text)
    if mm_slash_yyyy:
        return f"{int(mm_slash_yyyy.group(2)):04d}-{int(mm_slash_yyyy.group(1)):02d}-01"

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    y_m_d = re.match(r"^(\d{4})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{1,2})$", text)
    if y_m_d:
        return f"{int(y_m_d.group(1)):04d}-{int(y_m_d.group(2)):02d}-{int(y_m_d.group(3)):02d}"

    yyyy_mm = re.match(r"^(\d{4})-(\d{1,2})$", text)
    if yyyy_mm:
        return f"{int(yyyy_mm.group(1)):04d}-{int(yyyy_mm.group(2)):02d}-01"

    return text
