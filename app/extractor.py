"""Turn raw OCR text of a medicine / medical-device label into structured fields.

Handles the ways these labels encode data:
  * GS1 human-readable strings:  (01)GTIN (11)MFG (17)EXP (10)LOT (21)SERIAL   (YYMMDD or YYYYMM dates)
  * ISO 15223 symbols that OCR usually drops (factory icon = MFG date, hourglass = EXP date,
    "LOT" box, "SN" box) -> resolved by keyword, by caption direction, else by position
  * Many date formats: YYYY-MM-DD, YYYYMMDD, DDMMYYYY, DD/MM/YYYY, MM/YYYY, "03 2027", YYYYMM,
    YYMMDD, YYYY MM DD, 03 MAR 2023, MAR2028 ...
  * Lot numbers that themselves look like dates (20250108) or YYMMDD (241028)
  * Column layouts ("REF LOT" header row, then a row of values)
  * Several products in one image (one record per lot / GS1 set) and several serial numbers
    sharing one lot (one record per serial)
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
# "2509/M2D607609/2025" -> lot glued to "09/2025": put a space in front of MM/YYYY that follows an alnum char
_GLUED_DATE = re.compile(r"(?<=[A-Za-z0-9])(?=\d{2}[/.\-](?:19|20)\d{2})")


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

    def fix(m: re.Match) -> str:
        s = m.group(0)
        if sum(c.isdigit() for c in s) >= 2:
            return s.translate(_DIGIT_FIX)
        return s

    t = _OCR_DATEISH.sub(fix, t)
    t = _GLUED_DATE.sub(" ", t)
    return "\n".join(line.strip() for line in t.split("\n"))


# --------------------------------------------------------------------------- dates

_MON = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

_P_YMD = re.compile(r"(?<!\d)(\d{4})[ \t]*[-/.][ \t]*(\d{1,2})[ \t]*[-/.][ \t]*(\d{1,2})(?!\d)")
_P_DMY = re.compile(r"(?<!\d)(\d{1,2})[ \t]*[-/.][ \t]*(\d{1,2})[ \t]*[-/.][ \t]*(\d{4})(?!\d)")
_P_YMD8 = re.compile(r"(?<!\d)((?:19|20)\d{2})(\d{2})(\d{2})(?!\d)")
_P_DMY8 = re.compile(r"(?<!\d)(0[1-9]|[12]\d|3[01])(0[1-9]|1[0-2])(20\d{2})(?!\d)")
_P_MON = re.compile(r"(?<![A-Za-z0-9])(?:(\d{1,2})[ \t]*[-/.]?[ \t]*)?(?<![A-Za-z])(" + _MON + r")(?![A-Za-z])\.?[ \t]*[-/.,]?[ \t]*(\d{4}|\d{2})(?!\d)", re.I)
_P_YMON = re.compile(r"(?<!\d)(\d{4})[ \t]*[-/.]?[ \t]*(?<![A-Za-z])(" + _MON + r")(?![A-Za-z])", re.I)
_P_MY = re.compile(r"(?<![\d/])(\d{1,2})[ \t]*[-/.][ \t]*(\d{4})(?![\d/])")
_P_YM = re.compile(r"(?<![\d/])(\d{4})[ \t]*[-/.][ \t]*(\d{1,2})(?![\d/])")
# keyword-gated patterns (only kept when an EXP/MFG keyword precedes them)
_P_SHORT = re.compile(r"(?<!\d)(\d{2})[ \t]*[-/.][ \t]*(\d{2})(?!\d)")          # MM/YY
_P_MY_SP = re.compile(r"(?<!\d)(\d{2})[ \t]+(20\d{2})(?!\d)")                     # "03 2027"
_P_YMD_SP = re.compile(r"(?<!\d)(20\d{2})[ \t]+(\d{2})[ \t]+(\d{2})(?!\d)")       # "2024 09 26"
# compact 6-digit tokens: YYMMDD or YYYYMM (used when nothing better is available)
_P_COMPACT6 = re.compile(r"(?<![\d\-/.])(\d{6})(?![\d\-/.]|\d)")

YEAR_MIN, YEAR_MAX = 2000, 2060


@dataclass
class DateHit:
    start: int
    end: int
    iso: str
    prec: str = "day"            # "day" | "month"
    ambiguous: bool = False
    short: bool = False          # keep only if a keyword labels it
    compact: bool = False        # bare 6-digit token (YYMMDD / YYYYMM)
    label: Optional[str] = None  # "mfg" | "exp" | None


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


def compact6_date(tok: str) -> Optional[tuple[str, str]]:
    """'251205' -> YYMMDD (2025-12-05);  '202410' -> YYYYMM (2024-10-01)."""
    if len(tok) != 6 or not tok.isdigit():
        return None
    yy, mm, dd = int(tok[:2]), int(tok[2:4]), int(tok[4:])
    if 15 <= yy <= 45 and 1 <= mm <= 12 and 1 <= dd <= 31:
        r = _mk(2000 + yy, mm, dd)
        if r:
            return r
    y4, m2 = int(tok[:4]), int(tok[4:])
    if 2015 <= y4 <= 2045 and 1 <= m2 <= 12:
        return _mk(y4, m2, None)
    return None


def parse_dates(text: str, include_compact: bool = False) -> list[DateHit]:
    hits: list[DateHit] = []
    occ: list[tuple[int, int]] = []

    def add(m: re.Match, built: Optional[tuple[str, str]], amb=False, short=False, compact=False):
        if built and _free(occ, m.start(), m.end()):
            occ.append((m.start(), m.end()))
            hits.append(DateHit(m.start(), m.end(), built[0], built[1], amb, short, compact))

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
    for m in _P_DMY8.finditer(text):
        add(m, _mk(int(m[3]), int(m[2]), int(m[1])))
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
    for m in _P_YMD_SP.finditer(text):
        add(m, _mk(int(m[1]), int(m[2]), int(m[3])), short=True)
    for m in _P_MY_SP.finditer(text):
        add(m, _mk(int(m[2]), int(m[1]), None), short=True)
    for m in _P_SHORT.finditer(text):
        a, b = int(m[1]), int(m[2])
        if 1 <= a <= 12 and 20 <= b <= 60:
            add(m, _mk(2000 + b, a, None), short=True)
    if include_compact:
        for m in _P_COMPACT6.finditer(text):
            add(m, compact6_date(m.group(1)), compact=True)
    hits.sort(key=lambda h: h.start)
    return hits


# --------------------------------------------------------------------------- keywords

_MFG_KW = re.compile(
    r"\bmfg\b|\bmfd\b|\bmnf\b|\bmfr?\.?\s*date|manufactur\w*\s*date|date\s*of\s*manufactur\w*|"
    r"production\s*date|prod\.?\s*date|\bdom\b|fabrication|fabricaci[oó]n|herstellungsdatum|"
    r"生产日期|生產日期|制造日期|🏭", re.I)
_EXP_KW = re.compile(
    r"\bex[pr]\b|\bexpiry\b|\bexpiration\b|\bexpires?\b|\buse\s*by\b|\bbest\s*before\b|\bbbe?\b|"
    r"\bvalid\s*(?:until|till|thru)\b|\bnot\s*after\b|p[ée]remption|caducidad|verwendbar|"
    r"haltbar\w*|有效期\w*|失效日期|保质期\w*|⌛|⏳|⧖|⧗|\bverfall\w*", re.I)


@dataclass
class _Kw:
    start: int
    end: int
    kind: str


def _find_keywords(text: str) -> list[_Kw]:
    kws = [_Kw(m.start(), m.end(), "mfg") for m in _MFG_KW.finditer(text)]
    kws += [_Kw(m.start(), m.end(), "exp") for m in _EXP_KW.finditer(text)]
    kws.sort(key=lambda k: (k.start, -(k.end - k.start)))
    out: list[_Kw] = []
    for k in kws:
        if out and k.start < out[-1].end:
            continue
        out.append(k)
    return out


def _label_pass(text: str, hits: list[DateHit], kws: list[_Kw], backward: bool) -> dict[int, str]:
    """Pair keywords with dates. forward: keyword precedes its date; backward: caption follows it.
    FIFO handles 'MFG EXP' header rows that sit above a row of values."""
    events = sorted([(k.start, 0, k) for k in kws] + [(h.start, 1, h) for h in hits], key=lambda e: (e[0], e[1]))
    if backward:
        events.reverse()
    res: dict[int, str] = {}
    queue: list[_Kw] = []
    for _, typ, obj in events:
        if typ == 0:
            queue.append(obj)
            continue
        h: DateHit = obj

        def gap(k: _Kw) -> str:
            return text[h.end:k.start] if backward else text[k.end:h.start]

        queue = [k for k in queue if len(gap(k)) <= 90 and gap(k).count("\n") <= 2]
        chosen: Optional[_Kw] = None
        if queue:
            nearest = queue[-1]
            g = gap(nearest)
            if len(g) <= 25 and "\n" not in g:
                chosen = nearest
            elif len(queue) > 1:
                chosen = queue[0]
            elif len(g) <= 45:
                chosen = nearest
        if chosen:
            queue.remove(chosen)
            res[id(h)] = chosen.kind
    return res


def _label_score(hits: list[DateHit], res: dict[int, str]) -> float:
    mf = [h.iso for h in hits if res.get(id(h)) == "mfg"]
    ex = [h.iso for h in hits if res.get(id(h)) == "exp"]
    score = float(len(mf) + len(ex))
    if mf and ex:
        score += 1.0 if min(mf) < max(ex) else -2.0
    return score


def _assign_labels(text: str, hits: list[DateHit], kws: list[_Kw]) -> None:
    fwd = _label_pass(text, hits, kws, backward=False)
    bwd = _label_pass(text, hits, kws, backward=True)
    best = bwd if _label_score(hits, bwd) > _label_score(hits, fwd) else fwd
    for h in hits:
        h.label = best.get(id(h))


# --------------------------------------------------------------------------- GS1

def gtin_valid(code: str) -> bool:
    if not code.isdigit() or len(code) not in (8, 12, 13, 14):
        return False
    digits = [int(c) for c in code]
    check = digits.pop()
    total = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(digits)))
    return (10 - total % 10) % 10 == check


def _gtin_eq(a: str, b: str) -> bool:
    return a.lstrip("0") == b.lstrip("0")


def _gtin14(code: str) -> str:
    return code.zfill(14) if len(code) in (12, 13) else code


@dataclass
class GS1Item:
    ai: str
    value: str
    start: int
    end: int


_GS1_OPEN = re.compile(r"[\(\[\{]\s*(01|10|11|13|15|17|21)\s*[\)\]\}]\s*")
_GS1_DATE = re.compile(r"(\d(?:\s?\d){5})")
_GS1_GTIN = re.compile(r"(\d(?:\s?\d){13})")
_GS1_ALNUM = re.compile(r"([A-Za-z0-9\-./]{1,24})")


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


def gs1_date(v: str) -> Optional[str]:
    """YYMMDD (standard; DD=00 -> day 01). Some labels print YYYYMM instead -> month precision."""
    if len(v) != 6 or not v.isdigit():
        return None
    built = _mk(2000 + int(v[:2]), int(v[2:4]), int(v[4:]))
    if built:
        return built[0]
    y4, m2 = int(v[:4]), int(v[4:])
    if 2010 <= y4 <= YEAR_MAX and 1 <= m2 <= 12:
        built = _mk(y4, m2, None)
        return built[0] if built else None
    return None


# --------------------------------------------------------------------------- printed identifiers

# label kinds: lot / batch / batchcap (caption "Batch Code") / serial / ref (claims its value)
_LABELS = re.compile(
    r"(?P<lot>\b(?:l[o0]t|[il1]ot|lote|lotto|chargen?[- ]?(?:nr)?)(?![A-Za-z]))|"
    r"(?P<batchcap>\bbatch[ \t]*code\b)|"
    r"(?P<batch>\b(?:batch(?:[ \t]*(?:no|number|nr|#))?|b[./]?[ \t]?n(?:o)?)(?![A-Za-z]))|"
    r"(?P<serial>\b(?:serial(?:[ \t]*(?:no|number|nr|num))?|s/n|sn|ser\.?[ \t]*no)(?![A-Za-z]))|"
    r"(?P<ref>\b(?:ref|cat(?:alog(?:ue)?)?(?:[ \t]*(?:no|number|nr))?|item(?:[ \t]*no)?|model|art(?:icle)?\.?[ \t]*(?:no|nr))(?![A-Za-z]))|"
    r"(?P<refx>\b(?:size|qty|quantity|pcs|code|pn|p/n)(?![A-Za-z]))",
    re.I)
_LABEL_LOT_ZH = re.compile(r"批号|批號")
_TOKEN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9][A-Za-z0-9\-/\.]{2,30})")
_INLINE_GAP = re.compile(r"^[ \t\n]*(?:no\.?|number|nr\.?|n°|#)?[ \t\n]*[:.\-#=]*[ \t\n]*$", re.I)


@dataclass
class Tok:
    value: str
    start: int
    end: int
    kind: str = ""


def _clean_tok(v: str) -> str:
    return v.strip(".-/,;:").upper()


def _is_sep_date(v: str) -> bool:
    if not re.search(r"[-/.]", v):
        return False
    hits = parse_dates(v)
    return bool(hits) and hits[0].end - hits[0].start >= len(v) - 1


def _tok_ok(v: str, min_len: int = 3) -> bool:
    if len(v) < min_len or not any(c.isdigit() for c in v):
        return False
    if re.fullmatch(r"\d+\.\d+", v):          # 6.5, 8.8 (dimensions)
        return False
    return not _is_sep_date(v)


def _norm(v: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", v.upper()).replace("O", "0")


def _similar(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    if a == b:
        return True
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    # truncated read, e.g. "KM" vs "KM2511639"
    if len(s) >= 2 and (l.startswith(s) or l.endswith(s)) and (len(s) >= 4 or s.isalpha()):
        return True
    if len(l) - len(s) <= 2 and len(s) >= 8:
        i = 0
        while i < len(s) and s[-1 - i] == l[-1 - i]:
            i += 1
        if i >= 7:                    # same 7+ char tail: "X02511639" vs "KM2511639"
            return True
    if len(s) < 6 or len(l) - len(s) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


def _find_ids(text: str) -> tuple[dict[str, list[Tok]], list[tuple[int, int]]]:
    """Pair LOT / BATCH / SN / REF labels with value tokens.

    Phase 1  a lone label takes the value right after it (same line, or the next line).
    Phase 2  labels printed side by side ("REF LOT <icon>" header row) take the values of the
             following row in order; labels that found nothing inline take the next free value.
    """
    labels = []
    for m in _LABELS.finditer(text):
        labels.append((m.lastgroup, m.start(), m.end()))
    for m in _LABEL_LOT_ZH.finditer(text):
        labels.append(("lot", m.start(), m.end()))
    labels.sort(key=lambda x: x[1])

    toks: list[Tok] = []
    for m in _TOKEN.finditer(text):
        v = _clean_tok(m.group(1))
        if any(c.isdigit() for c in v) and not any(s < m.end(1) and m.start(1) < e for _, s, e in labels):
            toks.append(Tok(v, m.start(1), m.start(1) + len(m.group(1).rstrip(".-/,;:")), ""))

    FIFO_KINDS = {"lot", "batch", "serial", "ref"}
    claimed: dict[int, str] = {}
    done: set[int] = set()

    # label groups: labels separated only by a little non-value text
    groups: list[list[int]] = []
    for li, (kind, ls, le) in enumerate(labels):
        if groups:
            pk, ps, pe = labels[groups[-1][-1]]
            between = text[pe:ls]
            if len(between) <= 30 and between.count("\n") <= 1 and not any(t.start >= pe and t.end <= ls for t in toks):
                groups[-1].append(li)
                continue
        groups.append([li])

    def first_after(pos: int, skip_claimed=True) -> Optional[int]:
        for ti, t in enumerate(toks):
            if t.start >= pos and not (skip_claimed and ti in claimed):
                return ti
        return None

    # phase 1: lone labels, inline value
    for g in groups:
        part = [li for li in g if labels[li][0] in FIFO_KINDS or labels[li][0] in ("refx", "batchcap")]
        if len(g) != 1:
            continue
        li = g[0]
        kind, ls, le = labels[li]
        ti = first_after(le)
        if ti is not None and _INLINE_GAP.match(text[le:toks[ti].start]) and _tok_ok(toks[ti].value, 3):
            claimed[ti] = kind
            done.add(li)

    # phase 2: header rows and labels without an inline value
    for g in groups:
        todo = [li for li in g if li not in done and labels[li][0] in FIFO_KINDS]
        if not todo:
            continue
        last_end = labels[g[-1]][2]
        for li in todo:
            kind, ls, le = labels[li]
            anchor = max(last_end if len(g) > 1 else le, 0)
            ti = None
            for k, t in enumerate(toks):
                if k in claimed or t.start < anchor:
                    continue
                gap = text[anchor:t.start]
                if len(gap) > 90 or gap.count("\n") > 3:
                    break
                if _tok_ok(t.value, 4):
                    ti = k
                    break
            if ti is not None:
                claimed[ti] = kind
                done.add(li)
                last_end = toks[ti].end
                if len(g) == 1:
                    break

    out: dict[str, list[Tok]] = {"lot": [], "batch": [], "serial": []}
    spans: list[tuple[int, int]] = []
    for ti, kind in claimed.items():
        t = toks[ti]
        spans.append((t.start, t.end))
        k = {"batchcap": "batch"}.get(kind, kind)
        if k in out:
            out[k].append(Tok(t.value, t.start, t.end, k))
    return out, spans


_GTIN_LABELED = re.compile(r"\b(?:gtin(?:-?\d{1,2})?|ean(?:-?13)?|upc)\b[ \t]*[:.\-#]*[ \t]*(\d[\d ]{6,17}\d)", re.I)
_BARE_GTIN = re.compile(r"(?<!\d)((?:\d[ ]?){12,13}\d)(?!\d)")


# --------------------------------------------------------------------------- records

@dataclass
class _Anchor:
    pos: list[int] = field(default_factory=list)
    gs1_lot: Optional[str] = None
    lot: Optional[str] = None
    batch: Optional[str] = None
    serials: list[str] = field(default_factory=list)
    gtin: Optional[str] = None
    gs1_mfg: Optional[str] = None
    gs1_exp: Optional[str] = None
    methods: dict = field(default_factory=dict)   # field -> gs1 | label | positional | inferred
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
        key = {"01": "gtin", "10": "gs1_lot", "21": "serial", "11": "gs1_mfg", "17": "gs1_exp", "15": "gs1_exp"}.get(it.ai)
        if key is None:
            continue
        if key == "serial":
            if cur is None:
                cur = _Anchor()
                anchors.append(cur)
            if it.value not in cur.serials:
                cur.serials.append(it.value)
            cur.pos.append(it.start)
            continue
        is_date = key in ("gs1_mfg", "gs1_exp")
        val = gs1_date(it.value) if is_date else it.value
        if val is None:
            continue
        if it.ai == "15" and cur and cur.gs1_exp:
            continue
        existing = getattr(cur, key) if cur else None
        if existing is None or cur is None:
            differs = False
        elif key == "gtin":
            differs = not _gtin_eq(existing, val)
        elif is_date:
            differs = existing != val
        else:
            differs = not _similar(existing, val)
        if cur is None or differs:
            cur = _Anchor()
            anchors.append(cur)
        cur_val = getattr(cur, key)
        if cur_val is None:
            setattr(cur, key, val)
            cur.methods[{"gs1_lot": "lot", "gs1_mfg": "mfg", "gs1_exp": "exp"}.get(key, key)] = "gs1"
        elif key == "gs1_lot" and len(val) > len(cur_val) and _similar(cur_val, val):
            cur.gs1_lot = val           # earlier read was truncated
        cur.pos.append(it.start)
    return anchors


def _date_quality(method: Optional[str]) -> float:
    return {"gs1": 1.0, "label": 0.9, "positional": 0.55, "positional_single": 0.4}.get(method or "", 0.0)


def _months_between(a: str, b: str) -> int:
    return (int(b[:4]) - int(a[:4])) * 12 + int(b[5:7]) - int(a[5:7])


def _hamming1(a: str, b: str) -> bool:
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1


def _resolve_dates(a: _Anchor, today: date) -> None:
    mfg, exp = a.gs1_mfg, a.gs1_exp
    mm, em = ("gs1" if mfg else None), ("gs1" if exp else None)
    ds = a.dates

    # sanity of GS1 dates: mfg must precede exp
    if mfg and exp and mfg >= exp:
        a.notes.append("GS1 mfg date is not before exp date - ignored")
        mfg, mm = None, None

    lab_exp = sorted((d for d in ds if d.label == "exp"), key=lambda d: d.iso)
    lab_mfg = sorted((d for d in ds if d.label == "mfg"), key=lambda d: d.iso)
    printed_day = [d for d in ds if not d.short or d.label]

    # a one-digit OCR slip in the tiny GS1 text: prefer the matching printed date
    plain = [d for d in ds if not d.short and not d.compact and d.prec == "day"]
    if exp and not any(d.iso == exp for d in ds if not d.short):
        alt = next((d for d in plain if d.label in (None, "exp") and _hamming1(d.iso, exp)), None)
        if alt:
            a.notes.append(f"GS1 exp {exp} differs from printed {alt.iso}; used printed")
            exp, em = alt.iso, "label" if alt.label else "positional"
    if mfg and not any(d.iso == mfg for d in ds if not d.short):
        alt = next((d for d in plain if d.label in (None, "mfg") and _hamming1(d.iso, mfg)), None)
        if alt:
            a.notes.append(f"GS1 mfg {mfg} differs from printed {alt.iso}; used printed")
            mfg, mm = alt.iso, "label" if alt.label else "positional"
    if exp is None and lab_exp:
        exp, em = lab_exp[-1].iso, "label"
        a.ambiguous |= lab_exp[-1].ambiguous
    if mfg is None and lab_mfg:
        mfg, mm = lab_mfg[0].iso, "label"
        a.ambiguous |= lab_mfg[0].ambiguous

    used = {exp, mfg}
    seen: set[str] = set()
    U: list[DateHit] = []
    for d in ds:
        if d.label is None and not d.short and not d.compact and d.iso not in used and d.iso not in seen:
            seen.add(d.iso)
            U.append(d)

    if exp is None and mfg is None:
        if len(U) >= 2:
            best = None
            for x, y in zip(U, U[1:]):
                if x.iso < y.iso and 3 <= _months_between(x.iso, y.iso) <= 150:
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
            # a lone date on a label is overwhelmingly the expiry
            exp, em = U[0].iso, "positional_single"
            a.ambiguous |= U[0].ambiguous
    elif exp is None and mfg is not None:
        later = [d for d in U if d.iso > mfg]
        if later:
            exp, em = max(later, key=lambda d: d.iso).iso, "positional"
    elif mfg is None and exp is not None:
        earlier = [d for d in U if d.iso < exp]
        if earlier:
            mfg, mm = min(earlier, key=lambda d: d.iso).iso, "positional"

    # bare 6-digit YYMMDD / YYYYMM tokens (icon-only labels, lot printed in the same format)
    if mfg is None or exp is None:
        C: list[DateHit] = []
        cs: set[str] = set()
        for d in ds:
            if d.compact and d.label is None and d.iso not in cs:
                cs.add(d.iso)
                C.append(d)
        if exp is None and mfg is None:
            best = None
            for x, y in zip(C, C[1:]):
                if x.iso < y.iso and 3 <= _months_between(x.iso, y.iso) <= 150:
                    gap = y.start - x.end
                    if best is None or gap < best[0]:
                        best = (gap, x, y)
            if best:
                mfg, exp, mm, em = best[1].iso, best[2].iso, "positional", "positional"
        elif mfg is None and exp is not None:
            cand = [d for d in C if 0 <= _months_between(d.iso, exp) <= 150 and d.iso != exp]
            if cand:
                mfg, mm = max(cand, key=lambda d: d.iso).iso, "positional"
        elif exp is None and mfg is not None:
            cand = [d for d in C if 3 <= _months_between(mfg, d.iso) <= 150]
            if cand:
                exp, em = max(cand, key=lambda d: d.iso).iso, "positional"

    if mfg and exp and mfg > exp:
        if mm in ("positional", "positional_single") or em in ("positional", "positional_single"):
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
        c += 0.10 * (1.0 if gtin_valid(a.gtin.lstrip("0").zfill(14)) or gtin_valid(a.gtin) else 0.4)
    if a.serials:
        c += 0.05
    if a.ambiguous:
        c -= 0.03
    return round(max(0.0, min(0.99, c)), 2)


def _compatible(a: _Anchor, b: _Anchor) -> bool:
    shared = False
    for f in ("gtin", "lot", "batch", "gs1_exp", "gs1_mfg"):
        x, y = getattr(a, f), getattr(b, f)
        if x and y:
            if f == "gtin":
                same = _gtin_eq(x, y)
            elif f in ("lot", "batch"):
                same = _similar(x, y)
            else:
                same = x == y
            if not same:
                return False
            shared = True
    return shared


def _merge_into(a: _Anchor, b: _Anchor) -> None:
    for f in ("gtin", "batch", "gs1_exp", "gs1_mfg", "gs1_lot"):
        if getattr(a, f) is None and getattr(b, f) is not None:
            setattr(a, f, getattr(b, f))
    if b.lot and (a.lot is None or (len(b.lot) > len(a.lot) and _similar(a.lot, b.lot))):
        a.lot = b.lot
    for k, v in b.methods.items():
        a.methods.setdefault(k, v)
    a.pos += b.pos
    a.dates += b.dates
    for s in b.serials:
        if s not in a.serials:
            a.serials.append(s)
    a.notes += b.notes
    a.ambiguous |= b.ambiguous


def _merge_anchors(anchors: list[_Anchor]) -> list[_Anchor]:
    changed = True
    while changed:
        changed = False
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                if _compatible(anchors[i], anchors[j]):
                    _merge_into(anchors[i], anchors[j])
                    del anchors[j]
                    changed = True
                    break
            if changed:
                break
    return anchors


def _fallback_lot(text: str, a: _Anchor, claimed: list[tuple[int, int]]) -> None:
    """OCR dropped the 'LOT' label: take a lone alphanumeric token printed just above the dates."""
    if a.lot or a.batch or not a.dates:
        return
    first = min(d.start for d in a.dates if d.label or not d.short)
    head = text[:first]
    lines = head.split("\n")[-4:]
    offset = len(head) - sum(len(l) + 1 for l in lines) + 1
    pos = max(offset, 0)
    line_spans = []
    for l in lines:
        line_spans.append((l, pos))
        pos += len(l) + 1
    for l, lpos in reversed(line_spans):
        s = l.strip()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-/]{4,17}", s) and any(c.isdigit() for c in s):
            if _is_sep_date(s) or gtin_valid(s) or re.fullmatch(r"\d{12,}", s):
                continue
            if any(lpos < e and lpos + len(l) > st for st, e in claimed):
                continue
            if compact6_date(s) and len(s) == 6:
                continue
            a.lot, a.methods["lot"] = s.upper(), "inferred"
            a.notes.append("lot inferred from an unlabelled value above the dates")
            return


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
    ids, id_spans = _find_ids("".join(masked))
    for s, e in id_spans:
        mask(s, e)

    printed_gtins_text = "".join(masked)
    gtin_vals = [re.sub(r"\s", "", m.group(1)) for m in _GTIN_LABELED.finditer(printed_gtins_text)]

    def reject_lot(v: str) -> bool:
        # digits copied from a barcode / serial are not a lot number
        dig = re.sub(r"\D", "", v)
        return len(dig) >= 9 and any(dig[:5] == g.lstrip("0")[:5] or dig[:5] == g[:5] for g in gtin_vals)

    lots = [t for t in ids["lot"] if not reject_lot(t.value)]
    batches = [t for t in ids["batch"] if not (t.value.isdigit() and len(t.value) <= 4)]
    serials = ids["serial"]

    for t in lots:
        matches = [a for a in anchors if a.gs1_lot and _similar(a.gs1_lot, t.value)]
        if matches:
            for m_ in matches:
                if m_.gs1_lot and len(m_.gs1_lot) > len(t.value) and _norm(m_.gs1_lot).startswith(_norm(t.value)):
                    m_.lot = m_.gs1_lot
                else:
                    m_.lot = t.value
                m_.methods["lot"] = "label"
                m_.pos.append(t.start)
            continue
        same = next((a for a in anchors if a.lot and _similar(a.lot, t.value)), None)
        if same:
            same.pos.append(t.start)
            continue
        free = [a for a in anchors if a.lot is None and a.gs1_lot is None and a.batch is None]
        host = _nearest(free, t.start)
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

    for a in anchors:
        if a.gs1_lot and not a.lot and not a.batch:
            a.lot, a.methods["lot"] = a.gs1_lot, "gs1"

    for t in serials:
        host = _nearest(anchors, t.start)
        if host is None:
            host = _Anchor()
            anchors.append(host)
        if t.value not in host.serials:
            host.serials.append(t.value)
        host.pos.append(t.start)

    # GTINs printed outside GS1 form (labelled, or an EAN-13 / GTIN-14 with a valid check digit)
    mtext = "".join(masked)
    found: list[Tok] = []
    for m in _GTIN_LABELED.finditer(mtext):
        code = re.sub(r"\s", "", m.group(1))
        if gtin_valid(code) or len(code) == 14:
            found.append(Tok(_gtin14(code), m.start(1), m.end(1)))
    for m in _BARE_GTIN.finditer(mtext):
        code = re.sub(r"\s", "", m.group(1))
        if len(code) in (13, 14) and gtin_valid(code) and not any(_gtin_eq(t.value, code) for t in found):
            found.append(Tok(_gtin14(code), m.start(), m.end()))
    for t in found:
        if any(a.gtin and _gtin_eq(a.gtin, t.value) for a in anchors):
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
    hits = parse_dates(mtext, include_compact=True)
    kws = _find_keywords(mtext)
    _assign_labels(mtext, hits, kws)
    hits = [h for h in hits if h.label or not h.short]   # gated patterns need a keyword
    if hits and not anchors:
        anchors.append(_Anchor())
    for h in hits:
        host = _nearest(anchors, h.start)
        if host is not None:
            host.dates.append(h)

    if not anchors:
        return ParseResult([], steps + ["parser: no lot / GTIN / date patterns found in OCR text"])

    anchors = _merge_anchors(anchors)

    # one distinct lot in the whole picture -> it also belongs to anchors that lack one (e.g. carton GS1)
    distinct = {_norm(a.lot) for a in anchors if a.lot}
    if len(distinct) == 1:
        lot = next(a.lot for a in anchors if a.lot)
        for a in anchors:
            if not a.lot and not a.batch:
                a.lot, a.methods["lot"] = lot, "inferred"

    meds: list[Medicine] = []
    for a in anchors:
        _resolve_dates(a, today)
        _fallback_lot(mtext, a, id_spans)
        if not any([a.lot, a.batch, a.gtin, a.serials, a.gs1_mfg, a.gs1_exp]):
            continue
        used = sorted({"label" if v == "label" else "gs1" if v == "gs1" else "positional"
                       for v in a.methods.values()})
        base = dict(
            gtin=a.gtin, batch_no=a.batch, lot=a.lot, mfg_date=a.gs1_mfg, exp_date=a.gs1_exp,
            extraction_method=f"{METHOD_PREFIX}+" + "+".join(used), confidence=_confidence(a))
        for s in (a.serials or [None]):
            meds.append(Medicine(serial_number=s, **base))
        for n in a.notes:
            steps.append(f"parser: note - {n}")
        if any(v in ("positional", "positional_single", "inferred") for v in a.methods.values()):
            steps.append("parser: some fields had no text label and were assigned by position "
                         "(icons: factory=MFG, hourglass=EXP)")
    steps.append(f"parser: extracted {len(meds)} medicine(s)")
    return ParseResult(meds, steps)
