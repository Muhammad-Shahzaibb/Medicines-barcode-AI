"""Turn raw OCR text of a medicine / medical-device label into structured fields.

Handles the ways these labels encode data:
  * GS1 human-readable strings:  (01)GTIN (11)MFG YYMMDD (17)EXP YYMMDD (10)LOT (21)SERIAL
  * ISO 15223 symbols that OCR usually drops (factory icon = MFG date,
    hourglass = EXP date, "LOT" box, "SN" box)  -> resolved by label, else by position
  * Many date formats: YYYY-MM-DD, YYYYMMDD, DD/MM/YYYY, MM/YYYY, YYYY-MM, 03 MAR 2023 ...
  * Several products in one image (one record per lot / GS1 set)
"""
from __future__ import annotations

import calendar
import html
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .schemas import Medicine

METHOD_PREFIX = "paddleocr_vl"

# --------------------------------------------------------------------------- text cleaning

_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"})
_OCR_DATEISH = re.compile(r"(?<![A-Za-z0-9])[\dOolI|]{1,4}(?:[ \t]*[-/.][ \t]*[\dOolI|]{1,4}){1,2}(?![A-Za-z0-9])")


def clean_text(raw: str) -> str:
    t = html.unescape(raw or "")
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"<br\s*/?>|</tr>", "\n", t, flags=re.I)
    t = re.sub(r"</t[dh]>", " ", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\\\(|\\\)|\\\[|\\\]|\\text\{|\\mathrm\{|\\quad|\$", " ", t)
    t = t.replace("|", " ").replace(" ", " ")
    t = re.sub(r"[*#`]+", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    # O->0 / l->1 confusions, only inside tokens that look like dates and contain real digits
    def fix(m: re.Match) -> str:
        s = m.group(0)
        if sum(c.isdigit() for c in s) >= 2:
            return s.translate(_DIGIT_FIX)
        return s
    t = _OCR_DATEISH.sub(fix, t)
    return "\n".join(line.strip() for line in t.split("\n"))


# --------------------------------------------------------------------------- dates

_MON = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

_P_YMD = re.compile(r"(?<!\d)(\d{4})[ \t]*[-/.][ \t]*(\d{1,2})[ \t]*[-/.][ \t]*(\d{1,2})(?!\d)")
_P_DMY = re.compile(r"(?<!\d)(\d{1,2})[ \t]*[-/.][ \t]*(\d{1,2})[ \t]*[-/.][ \t]*(\d{4})(?!\d)")
_P_YMD8 = re.compile(r"(?<!\d)((?:19|20)\d{2})(\d{2})(\d{2})(?!\d)")
_P_MON = re.compile(r"(?<![A-Za-z0-9])(?:(\d{1,2})[ \t]*[-/.]?[ \t]*)?\b(" + _MON + r")\b\.?[ \t]*[-/.,]?[ \t]*(\d{4}|\d{2})(?!\d)", re.I)
_P_YMON = re.compile(r"(?<!\d)(\d{4})[ \t]*[-/.]?[ \t]*\b(" + _MON + r")\b", re.I)
_P_MY = re.compile(r"(?<!\d)(\d{1,2})[ \t]*[-/.][ \t]*(\d{4})(?!\d)")
_P_YM = re.compile(r"(?<!\d)(\d{4})[ \t]*[-/.][ \t]*(\d{1,2})(?!\d)")
_P_SHORT = re.compile(r"(?<!\d)(\d{2})[ \t]*[-/.][ \t]*(\d{2})(?!\d)")  # MM/YY (keyword gated)

YEAR_MIN, YEAR_MAX = 2000, 2060


@dataclass
class DateHit:
    start: int
    end: int
    iso: str
    prec: str = "day"          # "day" | "month"
    ambiguous: bool = False
    short: bool = False
    label: Optional[str] = None  # "mfg" | "exp" | None
    label_src: Optional[str] = None  # "label" | "positional"


def _mk(y: int, m: int, d: Optional[int]) -> Optional[tuple[str, str]]:
    if not (YEAR_MIN <= y <= YEAR_MAX and 1 <= m <= 12):
        return None
    if d is None or d == 0:
        return f"{y:04d}-{m:02d}-01", "month"
    if not (1 <= d <= calendar.monthrange(y, m)[1]):
        return None
    return f"{y:04d}-{m:02d}-{d:02d}", "day"


def _free(occ: list[tuple[int, int]], s: int, e: int) -> bool:
    return not any(s < oe and e > os for os, oe in occ)


def parse_dates(text: str) -> list[DateHit]:
    hits: list[DateHit] = []
    occ: list[tuple[int, int]] = []

    def add(m: re.Match, built: Optional[tuple[str, str]], amb=False, short=False):
        if built and _free(occ, m.start(), m.end()):
            occ.append((m.start(), m.end()))
            hits.append(DateHit(m.start(), m.end(), built[0], built[1], amb, short))

    for m in _P_YMD.finditer(text):
        add(m, _mk(int(m[1]), int(m[2]), int(m[3])))
    for m in _P_DMY.finditer(text):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        if a > 12:
            add(m, _mk(y, b, a))
        elif b > 12:
            add(m, _mk(y, a, b))
        else:
            add(m, _mk(y, b, a), amb=(a != b))
    for m in _P_YMD8.finditer(text):
        add(m, _mk(int(m[1]), int(m[2]), int(m[3])))
    for m in _P_MON.finditer(text):
        yr = int(m[3])
        yr = yr + 2000 if yr < 100 else yr
        add(m, _mk(yr, _MONTHS[m[2][:3].lower()], int(m[1]) if m[1] else None))
    for m in _P_YMON.finditer(text):
        add(m, _mk(int(m[1]), _MONTHS[m[2][:3].lower()], None))
    for m in _P_MY.finditer(text):
        add(m, _mk(int(m[2]), int(m[1]), None))
    for m in _P_YM.finditer(text):
        add(m, _mk(int(m[1]), int(m[2]), None))
    for m in _P_SHORT.finditer(text):
        a, b = int(m[1]), int(m[2])
        if 1 <= a <= 12 and 20 <= b <= 60:
            add(m, _mk(2000 + b, a, None), amb=False, short=True)
    hits.sort(key=lambda h: h.start)
    return hits


# --------------------------------------------------------------------------- keywords

_MFG_KW = re.compile(
    r"\bmfg\b|\bmfd\b|\bmnf\b|\bmfr?\.?\s*date|manufactur\w*\s*date|date\s*of\s*manufactur\w*|"
    r"production\s*date|prod\.?\s*date|\bdom\b|fabrication|fabricaci[oó]n|herstellungsdatum|"
    r"生产日期|生產日期|制造日期|🏭", re.I)
_EXP_KW = re.compile(
    r"\bexp\b|\bexpiry\b|\bexpiration\b|\bexpires?\b|\buse\s*by\b|\bbest\s*before\b|\bbbe?\b|"
    r"\bvalid\s*(?:until|till|thru)\b|\bnot\s*after\b|p[ée]remption|caducidad|verwendbar|"
    r"haltbar\w*|有效期\w*|失效日期|保质期\w*|⌛|⏳|⧖|⧗|\bverfall\w*", re.I)


@dataclass
class _Kw:
    start: int
    end: int
    kind: str
    used: bool = False


def _find_keywords(text: str) -> list[_Kw]:
    kws = [_Kw(m.start(), m.end(), "mfg") for m in _MFG_KW.finditer(text)]
    kws += [_Kw(m.start(), m.end(), "exp") for m in _EXP_KW.finditer(text)]
    # an "exp date" phrase may overlap "exp" + "date"; de-overlap
    kws.sort(key=lambda k: (k.start, -(k.end - k.start)))
    out: list[_Kw] = []
    for k in kws:
        if out and k.start < out[-1].end:
            continue
        out.append(k)
    return out


def _assign_labels(text: str, hits: list[DateHit], kws: list[_Kw]) -> None:
    """Label dates from nearby keywords. FIFO handles 'MFG EXP' header rows above value rows."""
    events = sorted([(k.start, 0, k) for k in kws] + [(h.start, 1, h) for h in hits],
                    key=lambda e: (e[0], e[1]))
    queue: list[_Kw] = []
    for _, typ, obj in events:
        if typ == 0:
            queue.append(obj)
            continue
        h: DateHit = obj
        queue = [k for k in queue
                 if h.start - k.end <= 90 and text[k.end:h.start].count("\n") <= 2]
        chosen: Optional[_Kw] = None
        if queue:
            nearest = queue[-1]
            gap = text[nearest.end:h.start]
            if len(gap) <= 25 and "\n" not in gap:
                chosen = nearest
            elif len(queue) > 1:
                chosen = queue[0]
            elif len(gap) <= 45:
                chosen = nearest
        if chosen:
            queue.remove(chosen)
            chosen.used = True
            h.label, h.label_src = chosen.kind, "label"


# --------------------------------------------------------------------------- GS1

def gtin_valid(code: str) -> bool:
    if not code.isdigit() or len(code) not in (8, 12, 13, 14):
        return False
    digits = [int(c) for c in code]
    check = digits.pop()
    total = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(digits)))
    return (10 - total % 10) % 10 == check


@dataclass
class GS1Item:
    ai: str
    value: str
    start: int
    end: int


_GS1_OPEN = re.compile(r"[\(\[\{]\s*(01|10|11|13|15|17|21)\s*[\)\]\}]\s*")
_GS1_DATE = re.compile(r"(\d(?:\s?\d){5})")
_GS1_GTIN = re.compile(r"(\d(?:\s?\d){13})")
_GS1_ALNUM = re.compile(r"([A-Za-z0-9\-./]{1,20})")


def extract_gs1(text: str) -> list[GS1Item]:
    items: list[GS1Item] = []
    for m in _GS1_OPEN.finditer(text):
        ai = m.group(1)
        pat = {"01": _GS1_GTIN, "11": _GS1_DATE, "13": _GS1_DATE, "15": _GS1_DATE, "17": _GS1_DATE}.get(ai, _GS1_ALNUM)
        v = pat.match(text, m.end())
        if not v:
            continue
        val = v.group(1)
        if ai in ("01", "11", "13", "15", "17"):
            val = re.sub(r"\s", "", val)
        else:
            val = val.strip(".-/").upper()
        if val:
            items.append(GS1Item(ai, val, m.start(), v.end()))
    return items


_RAW_GS1 = re.compile(r"(?<!\d)01(\d{14})(?:11(\d{6}))?(?:17(\d{6}))?(?:10([A-Za-z0-9]{1,20}))?(?![A-Za-z0-9])")


def extract_raw_gs1(text: str, taken: list[tuple[int, int]]) -> list[GS1Item]:
    """Un-bracketed element strings like 010303829030658211727123110502224 7 (rare in OCR)."""
    out: list[GS1Item] = []
    for m in _RAW_GS1.finditer(text):
        if len(m.group(0)) < 24 or not gtin_valid(m.group(1)) or not _free(taken, m.start(), m.end()):
            continue
        out.append(GS1Item("01", m.group(1), m.start(), m.start() + 16))
        if m.group(2):
            out.append(GS1Item("11", m.group(2), m.start(), m.end()))
        if m.group(3):
            out.append(GS1Item("17", m.group(3), m.start(), m.end()))
        if m.group(4):
            out.append(GS1Item("10", m.group(4).upper(), m.start(), m.end()))
    return out


def gs1_date(yymmdd: str) -> Optional[str]:
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    built = _mk(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:]))
    return built[0] if built else None


# --------------------------------------------------------------------------- printed ids

_TOK = r"([A-Za-z0-9][A-Za-z0-9\-/\.]{1,30})"
_SEP = r"[ \t]*(?:no\.?|number|nr\.?|n°|#)?[ \t]*[:.\-#]*\s*"
_LOT_RE = re.compile(r"\b(?:l[o0]t(?![a-z])|lote\b|lotto\b|chargen?[- ]?(?:nr)?\b)" + _SEP + _TOK, re.I)
_LOT_ZH = re.compile(r"批号|批號")
_BATCH_RE = re.compile(r"\b(?:batch(?:[ \t]*(?:no|number|nr|code))?|b\.?[ \t]?no|b/n|bn)\b\.?[ \t]*[:.\-#]*\s*" + _TOK, re.I)
_SERIAL_RE = re.compile(r"\b(?:serial(?:[ \t]*(?:no|number|nr))?|s/n|sn|ser\.?[ \t]*no)\b\.?[ \t]*[:.\-#]*\s*" + _TOK, re.I)
_GTIN_RE = re.compile(r"\b(?:gtin(?:-?\d{1,2})?|ean(?:-?13)?|upc)\b[ \t]*[:.\-#]*[ \t]*(\d[\d ]{6,17}\d)", re.I)
_BARE14 = re.compile(r"(?<!\d)(\d{14})(?!\d)")


@dataclass
class Tok:
    value: str
    start: int
    end: int


def _clean_tok(v: str) -> str:
    return v.strip(".-/,;:").upper()


def _tok_ok(v: str, min_len: int = 3) -> bool:
    if len(v) < min_len or not any(c.isdigit() for c in v):
        return False
    # a bare date is not an identifier
    hits = parse_dates(v)
    if hits and hits[0].end - hits[0].start >= len(v) - 1:
        return False
    return True


def _find_tokens(rx: re.Pattern, text: str, min_len: int = 3) -> list[Tok]:
    out = []
    for m in rx.finditer(text):
        v = _clean_tok(m.group(m.lastindex))
        if _tok_ok(v, min_len):
            out.append(Tok(v, m.start(m.lastindex), m.end(m.lastindex)))
    return out


def _norm(v: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", v.upper()).replace("O", "0")


def _similar(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    if a == b:
        return True
    if min(len(a), len(b)) < 6 or abs(len(a) - len(b)) > 1:
        return False
    # Levenshtein <= 1
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


# --------------------------------------------------------------------------- records

@dataclass
class _Anchor:
    pos: list[int] = field(default_factory=list)
    gs1_lot: Optional[str] = None
    lot: Optional[str] = None
    batch: Optional[str] = None
    serial: Optional[str] = None
    gtin: Optional[str] = None
    gs1_mfg: Optional[str] = None
    gs1_exp: Optional[str] = None
    gs1_keys: set = field(default_factory=set)
    methods: dict = field(default_factory=dict)   # field -> "gs1" | "label" | "positional"
    dates: list[DateHit] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    ambiguous: bool = False

    def dist(self, p: int) -> int:
        return min((abs(p - q) for q in self.pos), default=10**9)


def _nearest(anchors: list[_Anchor], p: int) -> Optional[_Anchor]:
    return min(anchors, key=lambda a: a.dist(p)) if anchors else None


def _build_gs1_anchors(items: list[GS1Item]) -> list[_Anchor]:
    anchors: list[_Anchor] = []
    cur: Optional[_Anchor] = None
    for it in sorted(items, key=lambda i: i.start):
        if it.ai not in ("01", "10", "11", "13", "15", "17", "21"):
            continue
        key = {"01": "gtin", "10": "gs1_lot", "21": "serial", "11": "gs1_mfg", "17": "gs1_exp",
               "15": "gs1_exp", "13": None}[it.ai]
        if key is None:
            continue
        val = gs1_date(it.value) if key in ("gs1_mfg", "gs1_exp") else it.value
        if val is None:
            continue
        # a 15 (best-before) must not override a real 17
        if it.ai == "15" and cur and cur.gs1_exp:
            continue
        existing = getattr(cur, key) if cur else None
        is_date = key in ("gs1_mfg", "gs1_exp")
        differs = existing is not None and (existing != val if is_date else not _similar(str(existing), str(val)))
        if cur is None or differs:
            cur = _Anchor()
            anchors.append(cur)
        if getattr(cur, key) is None:
            setattr(cur, key, val)
            cur.methods[{"gs1_lot": "lot", "gs1_mfg": "mfg", "gs1_exp": "exp"}.get(key, key)] = "gs1"
        cur.pos.append(it.start)
    return anchors


def _date_quality(method: Optional[str]) -> float:
    return {"gs1": 1.0, "label": 0.9, "positional": 0.55, "positional_single": 0.4}.get(method or "", 0.0)


def _resolve_dates(a: _Anchor, today: date) -> None:
    mfg, exp = a.gs1_mfg, a.gs1_exp
    mm, em = ("gs1" if mfg else None), ("gs1" if exp else None)
    ds = a.dates
    # labelled
    lab_exp = sorted((d for d in ds if d.label == "exp"), key=lambda d: d.iso)
    lab_mfg = sorted((d for d in ds if d.label == "mfg"), key=lambda d: d.iso)
    if exp is None and lab_exp:
        exp, em = lab_exp[-1].iso, "label"
        a.ambiguous |= lab_exp[-1].ambiguous
    if mfg is None and lab_mfg:
        mfg, mm = lab_mfg[0].iso, "label"
        a.ambiguous |= lab_mfg[0].ambiguous

    # unlabelled -> positional heuristics
    used = {exp, mfg}
    seen: set[str] = set()
    U: list[DateHit] = []
    for d in ds:
        if d.label is None and not d.short and d.iso not in used and d.iso not in seen:
            seen.add(d.iso)
            U.append(d)
    if exp is None and mfg is None:
        if len(U) >= 2:
            # best consecutive (text-order) ascending pair; tolerates stray dates
            best = None
            for x, y in zip(U, U[1:]):
                if x.iso < y.iso:
                    gap = y.start - x.end
                    if best is None or gap < best[0]:
                        best = (gap, x, y)
            if best:
                _, x, y = best
            else:
                x, y = min(U, key=lambda d: d.iso), max(U, key=lambda d: d.iso)
            mfg, exp, mm, em = x.iso, y.iso, "positional", "positional"
            a.ambiguous |= x.ambiguous or y.ambiguous
        elif len(U) == 1:
            d = U[0]
            if d.iso >= today.isoformat():
                exp, em = d.iso, "positional_single"
            else:
                mfg, mm = d.iso, "positional_single"
            a.ambiguous |= d.ambiguous
    elif exp is None and mfg is not None:
        later = [d for d in U if d.iso > mfg]
        if later:
            exp, em = max(later, key=lambda d: d.iso).iso, "positional"
    elif mfg is None and exp is not None:
        earlier = [d for d in U if d.iso < exp]
        if earlier:
            mfg, mm = min(earlier, key=lambda d: d.iso).iso, "positional"

    if mfg and exp and mfg > exp:
        if "positional" in (mm, em) or "positional_single" in (mm, em):
            mfg, exp, mm, em = exp, mfg, em, mm
        else:
            a.notes.append("mfg_date is after exp_date; check the label")
    a.gs1_mfg, a.gs1_exp = mfg, exp
    if mm:
        a.methods["mfg"] = mm
    if em:
        a.methods["exp"] = em


def _confidence(a: _Anchor) -> float:
    ident_q = 0.0
    for k in ("lot", "batch"):
        if getattr(a, k):
            ident_q = max(ident_q, {"gs1": 1.0, "label": 0.95}.get(a.methods.get(k, ""), 0.5))
    exp_q = _date_quality(a.methods.get("exp")) if a.gs1_exp else 0.0
    mfg_q = _date_quality(a.methods.get("mfg")) if a.gs1_mfg else 0.0
    c = 0.15 + 0.30 * ident_q + 0.30 * exp_q + 0.15 * mfg_q
    if a.gtin:
        c += 0.10 * (1.0 if gtin_valid(a.gtin) else 0.4)
    if a.serial:
        c += 0.05
    if a.ambiguous:
        c -= 0.03
    return round(max(0.0, min(0.99, c)), 2)


@dataclass
class ParseResult:
    medicines: list[Medicine]
    steps: list[str]


def parse_label_text(raw_text: str, today: Optional[date] = None) -> ParseResult:
    today = today or date.today()
    text = clean_text(raw_text)
    steps: list[str] = []
    if not text.strip():
        return ParseResult([], ["parser: OCR returned no text"])

    # 1. GS1 element strings ------------------------------------------------
    items = extract_gs1(text)
    taken = [(i.start, i.end) for i in items]
    items += extract_raw_gs1(text, taken)
    anchors = _build_gs1_anchors(items)
    if anchors:
        steps.append(f"parser: GS1 data found ({len(anchors)} set(s))")

    masked = list(text)

    def mask(s: int, e: int) -> None:
        for i in range(s, min(e, len(masked))):
            masked[i] = " "

    for it in items:
        mask(it.start, it.end)

    # 2. printed identifiers --------------------------------------------------
    lots = _find_tokens(_LOT_RE, text)
    for m in _LOT_ZH.finditer(text):
        t = re.match(r"[ \t]*[:：]?[ \t]*([A-Za-z0-9\-/\.]{3,30})", text[m.end():])
        if t and _tok_ok(_clean_tok(t.group(1))):
            lots.append(Tok(_clean_tok(t.group(1)), m.end() + t.start(1), m.end() + t.end(1)))
    batches = _find_tokens(_BATCH_RE, text)
    serials = _find_tokens(_SERIAL_RE, text)

    # drop a "lot" that is really part of a GS1 string we already consumed
    for t in lots + batches + serials:
        mask(t.start, t.end)

    for t in lots:
        match = next((a for a in anchors if a.gs1_lot and _similar(a.gs1_lot, t.value)), None)
        if match:
            match.lot, match.methods["lot"] = match.gs1_lot, "gs1"
            match.pos.append(t.start)
            continue
        same = next((a for a in anchors if a.lot and _similar(a.lot, t.value)), None)
        if same:
            same.pos.append(t.start)
            continue
        # reuse a GS1 anchor that has no lot yet, if close
        free = [a for a in anchors if a.lot is None and a.gs1_lot is None and a.batch is None]
        host = min(free, key=lambda a: a.dist(t.start)) if free and anchors and len(anchors) == 1 else None
        if host is None:
            host = _Anchor()
            anchors.append(host)
        host.lot, host.methods["lot"] = t.value, "label"
        host.pos.append(t.start)

    for t in batches:
        match = next((a for a in anchors if a.gs1_lot and _similar(a.gs1_lot, t.value)), None)
        if match and not match.lot:
            match.batch, match.methods["batch"] = match.gs1_lot, "gs1"
            match.pos.append(t.start)
            continue
        host = _nearest([a for a in anchors if a.batch is None or _similar(a.batch, t.value)], t.start)
        if host is None:
            host = _Anchor()
            anchors.append(host)
        host.batch, host.methods["batch"] = host.batch or t.value, "label"
        host.pos.append(t.start)

    # GS1 (10) with no printed counterpart -> lot
    for a in anchors:
        if a.gs1_lot and not a.lot and not a.batch:
            a.lot, a.methods["lot"] = a.gs1_lot, "gs1"

    for t in serials:
        host = _nearest([a for a in anchors if a.serial is None or a.serial == t.value], t.start)
        if host is None:
            host = _Anchor()
            anchors.append(host)
        host.serial, host.methods["serial"] = t.value, "label"
        host.pos.append(t.start)

    # GTINs printed outside GS1 form
    mtext = "".join(masked)
    printed_gtins: list[Tok] = []
    for m in _GTIN_RE.finditer(mtext):
        code = re.sub(r"\s", "", m.group(1))
        if gtin_valid(code) or len(code) == 14:
            printed_gtins.append(Tok(code, m.start(1), m.end(1)))
    for m in _BARE14.finditer(mtext):
        if gtin_valid(m.group(1)) and not any(t.value == m.group(1) for t in printed_gtins):
            printed_gtins.append(Tok(m.group(1), m.start(), m.end()))
    for t in printed_gtins:
        if any(a.gtin == t.value for a in anchors):
            continue
        host = _nearest([a for a in anchors if a.gtin is None], t.start)
        if host is None:
            host = _Anchor()
            anchors.append(host)
        host.gtin, host.methods["gtin"] = t.value, "label"
        host.pos.append(t.start)
        mask(t.start, t.end)

    # 3. printed dates ---------------------------------------------------------
    mtext = "".join(masked)
    hits = parse_dates(mtext)
    kws = _find_keywords(mtext)
    _assign_labels(mtext, hits, kws)
    hits = [h for h in hits if not h.short or h.label]  # MM/YY only when a keyword precedes
    if hits and not anchors:
        anchors.append(_Anchor())
    for h in hits:
        host = _nearest(anchors, h.start)
        if host is not None:
            host.dates.append(h)

    if not anchors:
        return ParseResult([], steps + ["parser: no lot / GTIN / date patterns found in OCR text"])

    meds: list[Medicine] = []
    for a in anchors:
        _resolve_dates(a, today)
        if not any([a.lot, a.batch, a.gtin, a.serial, a.gs1_mfg, a.gs1_exp]):
            continue
        used = sorted({"label" if v == "label" else "gs1" if v == "gs1" else "positional"
                       for v in a.methods.values()})
        meds.append(Medicine(
            gtin=a.gtin, batch_no=a.batch, lot=a.lot,
            mfg_date=a.gs1_mfg, exp_date=a.gs1_exp, serial_number=a.serial,
            extraction_method=f"{METHOD_PREFIX}+" + "+".join(used),
            confidence=_confidence(a),
        ))
        for n in a.notes:
            steps.append(f"parser: warning - {n}")
        if "positional" in a.methods.values() or "positional_single" in a.methods.values():
            steps.append("parser: dates without text labels were assigned by position (icons: factory=MFG, hourglass=EXP)")
    steps.append(f"parser: extracted {len(meds)} medicine(s)")
    return ParseResult(meds, steps)
