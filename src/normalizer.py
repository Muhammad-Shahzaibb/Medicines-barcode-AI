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
    "abr": "04",
    "ago": "08",
    "dic": "12",
}

FIELD_SYNONYMS: dict[str, list[str]] = {
    "gtin": ["gtin", "gtn", "ean", "udi"],
    "batch": ["batch", "bno", "b.no", "b.n", "bn", "b/n", "batch no", "batch no.", "batch number", "batch code"],
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
        "pro",
        "p",
        "prod",
        "production",
    ],
    "exp": ["exp", "expiry", "expire", "expiration", "exp date", "exp.date", "expiry date", "cad", "caducidad"],
    "sn": ["sn", "sno", "sr", "sr#", "serial", "serial no", "serial number"],
}


def normalize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9#./\s]", "", label.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    for canonical, synonyms in FIELD_SYNONYMS.items():
        if cleaned in synonyms or cleaned.rstrip(".:") in synonyms:
            return canonical
    return cleaned


def _gtin_check_digit_ok(digits: str) -> bool:
    if not digits.isdigit() or len(digits) not in (8, 12, 13, 14):
        return False
    body, check = digits[:-1], int(digits[-1])
    total = 0
    for i, ch in enumerate(reversed(body)):
        total += int(ch) * (3 if i % 2 == 0 else 1)
    return (10 - (total % 10)) % 10 == check


def normalize_gtin(value: str | None, *, require_check: bool | None = None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if text.startswith("+"):
        return None
    if re.search(r"[A-Za-z]", re.sub(r"\(\d{2,4}\)", "", text)):
        return None

    ai_match = re.search(r"\(01\)\s*(\d{8,14})", text)
    from_ai = False
    if ai_match:
        from_ai = True
        digits = ai_match.group(1)
    else:
        digits = re.sub(r"\D", "", text)
        if digits.startswith("01") and len(digits) >= 16:
            from_ai = True
            digits = digits[2:16]

    if len(digits) == 13:
        digits = "0" + digits
    if len(digits) not in (8, 12, 13, 14):
        return None
    if require_check is None:
        require_check = not from_ai
    if require_check and not _gtin_check_digit_ok(digits):
        return None
    return digits


def _format_date(year: int, month: int, day: int) -> str | None:
    try:
        parsed = datetime(year, month, day)
    except ValueError:
        return None
    if not (1990 <= parsed.year <= 2045):
        return None
    return parsed.strftime("%Y-%m-%d")


def normalize_date(value: str | None) -> str | None:
    if not value:
        return None

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(".", "-")

    if re.fullmatch(r"\d{6}", text):
        yy, mm, dd = int(text[:2]), int(text[2:4]), int(text[4:6])
        year = 2000 + yy if yy < 80 else 1900 + yy
        if dd == 0:
            dd = 1
        return _format_date(year, mm, dd)

    if re.fullmatch(r"\d{8}", text):
        yyyy, mm, dd = int(text[:4]), int(text[4:6]), int(text[6:8])
        if dd == 0:
            dd = 1
        return _format_date(yyyy, mm, dd)

    month_year = re.match(r"^([A-Za-z]{3,9})[/\-\s]+(\d{4})$", text, re.I)
    if month_year:
        month_key = month_year.group(1).lower()[:3]
        month = MONTH_MAP.get(month_key)
        if month:
            return _format_date(int(month_year.group(2)), int(month), 1)

    mm_yyyy = re.match(r"^(\d{1,2})\s+(\d{4})$", text)
    if mm_yyyy:
        return _format_date(int(mm_yyyy.group(2)), int(mm_yyyy.group(1)), 1)

    mm_yyyy_dash = re.match(r"^(\d{1,2})\s*-\s*(\d{4})$", text)
    if mm_yyyy_dash:
        return _format_date(int(mm_yyyy_dash.group(2)), int(mm_yyyy_dash.group(1)), 1)

    mm_slash_yyyy = re.match(r"^(\d{1,2})/(\d{4})$", text)
    if mm_slash_yyyy:
        return _format_date(int(mm_slash_yyyy.group(2)), int(mm_slash_yyyy.group(1)), 1)

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%Y.%m.%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return _format_date(parsed.year, parsed.month, parsed.day)
        except ValueError:
            continue

    y_m_d = re.match(r"^(\d{4})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{1,2})$", text)
    if y_m_d:
        return _format_date(int(y_m_d.group(1)), int(y_m_d.group(2)), int(y_m_d.group(3)))

    yyyy_mm = re.match(r"^(\d{4})-(\d{1,2})$", text)
    if yyyy_mm:
        return _format_date(int(yyyy_mm.group(1)), int(yyyy_mm.group(2)), 1)

    return None
