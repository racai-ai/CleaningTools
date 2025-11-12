#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import unicodedata
from pathlib import Path

# -------- CLI --------

def parse_args():
    ap = argparse.ArgumentParser(description="Clean PDF-derived texts, remove bibliography, scrub names in first lines (multi-model NER + ALL-CAPS heuristics + cue-based removal).")
    ap.add_argument("in_dir", type=Path, help="Input folder with .txt/.md")
    ap.add_argument("out_dir", type=Path, help="Output folder (mirror tree)")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--ext", default=".txt,.md", help="Comma-separated extensions to include (e.g., .txt,.md)")

    # Core toggles
    ap.add_argument("--no-biblio", dest="remove_biblio", action="store_false",
                    help="Do NOT remove bibliography (default: remove if detected)")
    ap.add_argument("--keep-tabs", action="store_true",
                    help="Keep literal tab characters instead of converting to spaces")
    ap.add_argument("--unhyphen", action="store_true",
                    help="Also merge hyphenated line breaks (safer off by default)")
    ap.add_argument("--debug", action="store_true", help="Print detection debug info")

    # NER options (multi-model)
    ap.add_argument("--ner", action="store_true", help="Enable NER-based scrubbing of personal names.")
    ap.add_argument("--ner-model", default=None, help="(Deprecated) Single spaCy model name.")
    ap.add_argument("--ner-models", default=None,
                    help="Comma-separated spaCy models to ensemble (e.g., 'ro_core_news_lg,en_core_web_sm'). "
                         "If omitted, tries ro_core_news_lg, ro_core_news_sm, en_core_web_sm.")
    ap.add_argument("--ner-lines", type=int, default=25, help="Number of first lines to scan (default 25)")
    ap.add_argument("--ner-mask", default=None, help="If set, replace PERSON entities with this token (e.g., '[REDACTED]')")
    ap.add_argument("--ner-drop-name-lines", action="store_true",
                    help="Drop lines if they are dominated by names (based on NER coverage).")

    # ALL-CAPS heuristic options
    ap.add_argument("--cap-detect", action="store_true",
                    help="Enable ALL-CAPS name heuristic detection.")
    ap.add_argument("--cap-drop-name-lines", action="store_true",
                    help="Drop lines dominated by ALL-CAPS name patterns.")
    ap.add_argument("--cap-mask", default=None,
                    help="If set, mask ALL-CAPS name tokens with this token (default: reuse --ner-mask if provided).")
    ap.add_argument("--cap-min-tokens", type=int, default=2,
                    help="Minimum ALL-CAPS name tokens to consider a name line (default 2).")
    ap.add_argument("--cap-min-prop", type=float, default=0.5,
                    help="Minimum proportion of ALL-CAPS name tokens to all word tokens to treat as name-dominated (default 0.5).")

    # Cue-based identity removal (NEW)
    ap.add_argument("--id-cues", action="store_true", default=True,
                    help="Enable cue-based removal around lines like 'Coordonator:' / 'Candidat:' (default on).")
    ap.add_argument("--id-cues-lines", type=int, default=120,
                    help="Examine the first N lines for identity cues (default 120).")
    ap.add_argument("--id-mask", default=None,
                    help="If set, mask names after cues instead of dropping their lines (rarely needed).")

    return ap.parse_args()

# -------- Unicode / Normalization --------

ZWS = "".join([
    "\u00ad",  # soft hyphen
    "\u200b", "\u200c", "\u200d", "\u2060",  # zero width chars
    "\ufeff",  # BOM
])

TRANSLIT_MAP = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'", "‹": "'", "›": "'",
    "—": "—", "–": "-", "-": "-", "‒": "-", "−": "-",  # keep em dash, normalize others
    "•": "- ", "·": "- ", "▪": "- ", "◦": "- ",
}

CONTROL_RE = re.compile(
    "[" +
    "".join(chr(c) for c in range(0x00, 0x20) if c not in (0x09, 0x0A, 0x0D)) +
    "\u007f" +  # DEL
    "]"
)

MULTISPACE_RE = re.compile(r"[ \t]{3,}")  # collapse 3+ spaces inside a line
SPACIFY_TABS_RE = re.compile(r"\t")

# Romanian diacritics fixes (cedilla -> comma; combining marks -> precomposed)
COMBINING_CEDILLA = "\u0327"
COMBINING_COMMA_BELOW = "\u0326"
ROM_DIACRITICS_FIXES = [
    (re.compile(r"s(?:%s|%s)" % (COMBINING_CEDILLA, COMBINING_COMMA_BELOW), flags=re.IGNORECASE), "ș"),
    (re.compile(r"t(?:%s|%s)" % (COMBINING_CEDILLA, COMBINING_COMMA_BELOW), flags=re.IGNORECASE), "ț"),
]
ROM_SWAP = str.maketrans({"ş": "ș", "Ş": "Ș", "ţ": "ț", "Ţ": "Ț"})

# Hyphenation undo: word-\ncontinue -> wordcontinue (heuristic)
UNHYPHEN_RE = re.compile(r"(?<=\w)-\n(?=[a-zăâîșțà-öø-ÿ])", re.IGNORECASE)

# Headers/footers/rules
PAGE_LINE_RE = re.compile(r"^\s*(page|pagina|pag\.)\s*\d+(\s*/\s*\d+)?\s*$", re.IGNORECASE)
JUST_DIGITS_RE = re.compile(r"^\s*\d{1,4}\s*$")
RULE_LINE_RE = re.compile(r"^\s*[-=—_]{3,}\s*$")

# -------- Bibliography detection --------

BIBLIO_HEAD_RE = re.compile(
    r"""(?ix)
    ^\s*
    [\f>\-–—\s]*                # optional form-feed, bullets, rules
    (?:
        bibliografie(?:\s+selectiv[ăa])? |
        bibliograph(?:y|ies)   |
        references?            |
        referin(?:te|țe)(?:\s+bibliografice)? |
        works \s+ cited        |
        lista \s+ bibliografic[ăa] |
        lucr[ăa]ri \s+ cit[aă]te    |
        webograf(?:ie|y)?      |
        sitograf(?:ie|y)?      |
        resurse (?:\s*[/\-]\s*web)?
    )
    \s*[:\-–—]?\s*$
    """,
    re.UNICODE,
)

REF_ITEM_LEAD_RE = re.compile(r"^\s*(?:\d+[\.\)]|[-–•])\s+")
YEAR_RE          = re.compile(r"\((?:19|20)\d{2}\)")
URL_RE           = re.compile(r"https?://|doi\.org/\S+", re.IGNORECASE)

def detect_bibliography_start(lines: list[str], *, debug=False) -> int:
    n = len(lines)
    last_head = -1
    for i, line in enumerate(lines):
        if BIBLIO_HEAD_RE.match(line.strip()):
            last_head = i
    if last_head >= 0:
        if debug:
            print(f"[BIB] Heading at line {last_head}")
        return last_head

    # Fallback: tail scanning
    start_scan = int(n * 0.6)
    window = 12
    for i in range(start_scan, max(start_scan, n - window + 1)):
        win = lines[i:i+window]
        ref_like = 0
        items = 0
        for w in win:
            s = w.strip()
            if not s:
                continue
            score = 0
            if REF_ITEM_LEAD_RE.match(s): score += 1; items += 1
            if YEAR_RE.search(s):         score += 1
            if URL_RE.search(s):          score += 1
            if re.search(r"[A-ZĂÂÎȘȚ][a-zăâîșț]+,\s*[A-Z]\.", s): score += 1
            if score >= 1: ref_like += 1
        if items >= 5 and ref_like >= 6:
            if debug:
                print(f"[BIB] Fallback window start at {i} (items={items}, ref_like={ref_like})")
            return i

    # Long-run heuristic
    run = 0
    best = -1
    for i in range(start_scan, n):
        s = lines[i].strip()
        if not s:
            run += 1
            continue
        looks_ref = (
            REF_ITEM_LEAD_RE.match(s) or
            YEAR_RE.search(s) or
            URL_RE.search(s) or
            re.search(r"[A-ZĂÂÎȘȚ][a-zăâîșț]+,\s*[A-Z]\.", s)
        )
        if looks_ref:
            run += 1
            if best == -1:
                best = i
        else:
            if run >= 8:
                if debug:
                    print(f"[BIB] Fallback long-run start at {best} (len={run})")
                return best
            run = 0
            best = -1
    if run >= 8 and best != -1:
        if debug:
            print(f"[BIB] Fallback long-run (EOF) start at {best} (len={run})")
        return best
    return -1

def remove_bibliography(text: str, *, debug=False) -> str:
    lines = text.split("\n")
    start = detect_bibliography_start(lines, debug=debug)
    if start >= 0:
        return "\n".join(lines[:start]).rstrip() + "\n"
    return text

# -------- Title/heading detection (to protect titles) --------

TEZA_HEAD_RE = re.compile(r"^\s*tez[ăa]\s+de\s+doctorat\s*$", re.IGNORECASE)
TITLE_CUE_RE = re.compile(r"^\s*(titlul|titlu[lui]?|title)\b", re.IGNORECASE)
UNIV_BLOCK_RE = re.compile(r"^\s*(universitatea|academia|facultatea|școala\s+doctorală|scoala\s+doctorala)\b", re.IGNORECASE)
UPPERISH_RE = re.compile(r"^[^a-zăâîșț]*[A-ZĂÂÎȘȚ][^a-zăâîșț]*$")

def line_looks_like_title(s: str) -> bool:
    st = s.strip()
    if not st:
        return False
    if TITLE_CUE_RE.match(st):
        return True
    if TEZA_HEAD_RE.match(st) or UNIV_BLOCK_RE.match(st):
        return True
    lowers = sum(ch.islower() for ch in st)
    uppers = sum(ch.isupper() for ch in st)
    if (uppers >= 3 and lowers <= 1) or (len(st) >= 100 and uppers >= lowers):
        return True
    if UPPERISH_RE.match(st) and len(st) >= 15:
        return True
    return False

# -------- NER loader (multi-model) --------

def try_load_spacy_multi(model_name_single: str | None, models_multi: str | None, debug=False):
    try:
        import spacy
    except Exception:
        if debug:
            print("[NER] spaCy not installed; skipping NER.")
        return []
    if models_multi:
        candidates = [m.strip() for m in models_multi.split(",") if m.strip()]
    elif model_name_single:
        candidates = [model_name_single.strip()]
    else:
        candidates = ["ro_core_news_lg", "ro_core_news_sm", "en_core_web_sm"]
    loaded = []
    for m in candidates:
        try:
            nlp = spacy.load(m)
            loaded.append(nlp)
            if debug:
                print(f"[NER] Loaded spaCy model: {m}")
        except Exception as e:
            if debug:
                print(f"[NER] Could not load {m}: {e}")
    return loaded

# -------- ALL-CAPS heuristic detection --------

ACADEMIC_TITLES_RE = re.compile(
    r"""(?ix)
    \b(
        PR\.?|PREOT|PREOTUL|PĂRINTE|PARINTE|
        PROF\.?|UNIV\.?|DR\.?|DRD\.?|CONF\.?|LECT\.?|ASIST\.?|ȘEF\s+LUCR\.?|SEF\s+LUCR\.?|
        PH\.?D\.?|MSC|ING\.?|FARM\.?|AVOC\.?
    )\b
    """
)

ALLCAPS_NAME_TOKEN_RE = re.compile(r"^[A-ZĂÂÎȘȚ][A-ZĂÂÎȘȚ\-]{1,}$")
TITLECASE_NAME_TOKEN_RE = re.compile(r"^[A-ZĂÂÎȘȚ][a-zăâîșț]+(?:-[A-ZĂÂÎȘȚ][a-zăâîșț]+)?$")

def is_name_like_token(tok: str) -> bool:
    return bool(ALLCAPS_NAME_TOKEN_RE.match(tok) or TITLECASE_NAME_TOKEN_RE.match(tok) or re.fullmatch(r"[A-ZĂÂÎȘȚ]\.?", tok))

def detect_caps_name_line(s: str, min_tokens: int, min_prop: float) -> tuple[bool, list[tuple[int,int]]]:
    spans = []
    if not s.strip():
        return False, spans
    tokens = []
    for m in re.finditer(r"\S+", s):
        tok = m.group(0); start, end = m.span()
        tokens.append((tok, start, end))
    if not tokens:
        return False, spans
    title_hits = 1 if ACADEMIC_TITLES_RE.search(s) else 0
    name_hits = 0
    for tok, start, end in tokens:
        if is_name_like_token(tok):
            name_hits += 1
            spans.append((start, end))
    word_tokens = sum(1 for tok, _, _ in tokens if re.search(r"[A-Za-zĂÂÎȘȚăâîșț]", tok))
    prop = (name_hits / max(1, word_tokens))
    is_name_dominated = (name_hits >= min_tokens and prop >= min_prop) or (title_hits and name_hits >= 1)
    return is_name_dominated, spans

# -------- NER multi-model spans & coverage --------

def person_spans_union_from_models(nlps: list, s: str):
    if not nlps:
        return []
    try:
        import spacy  # noqa: F401
    except Exception:
        return []
    spans = []
    s_tc = s.title()
    for nlp in nlps:
        doc = nlp(s)
        spans.extend((e.start_char, e.end_char) for e in doc.ents if e.label_ == "PERSON")
        if len(s_tc) == len(s):
            doc_tc = nlp(s_tc)
            spans.extend((e.start_char, e.end_char) for e in doc_tc.ents if e.label_ == "PERSON")
    if not spans:
        return []
    spans.sort()
    merged = []
    a, b = spans[0]
    for x, y in spans[1:]:
        if x <= b:
            b = max(b, y)
        else:
            merged.append((a, b))
            a, b = x, y
    merged.append((a, b))
    return merged

def ner_name_coverage_ratio(nlps: list, s: str) -> float:
    spans = person_spans_union_from_models(nlps, s)
    nonspace = len(re.sub(r"\s+", "", s))
    if nonspace == 0:
        return 0.0
    covered = sum(b - a for a, b in spans)
    return covered / nonspace

# -------- Utilities --------

def mask_spans(s: str, spans: list[tuple[int,int]], token: str) -> str:
    out = s
    for a, b in sorted(spans, key=lambda t: t[0], reverse=True):
        out = out[:a] + token + out[b:]
    return out

# -------- Cue-based ID removal (NEW & IMPORTANT) --------

ID_CUE_RE = re.compile(
    r"""(?ix)
    ^\s*
    (?:coordonator(?:\s*:\s*)?|
       conduc[aă]tor(?:\s+științific| \s+stiintific)?(?:\s*:\s*)?|
       candidat(?:\s*:\s*)?|
       doctorand(?:[ăa]?)(?:\s*:\s*)?|
       autor(?:\s*:\s*)?|
       prezentat[ăa].*autor\s*:\s*
    )\s*$
    """
)

def remove_identity_by_cues(text: str, *,
                            first_n_lines: int,
                            cap_min_tokens: int,
                            cap_min_prop: float,
                            mask_token: str | None = None,
                            debug: bool = False) -> str:
    """
    Find lines like 'Coordonator:' / 'Candidat:' in the first N lines (twice to cover a second title page),
    remove the cue line and the following 1–3 name-like lines (caps/titles), or mask them.
    """
    def process_once(lines: list[str]) -> list[str]:
        keep = [True] * len(lines)
        limit = min(first_n_lines, len(lines))
        i = 0
        while i < limit:
            s = lines[i].strip()
            if ID_CUE_RE.match(s):
                if debug:
                    print(f"[CUE] Found identity cue at line {i}: {lines[i]!r}")
                # drop the cue line
                keep[i] = False
                # consume next up to 3 lines if they look like names/titles
                j = i + 1
                consumed = 0
                while j < len(lines) and consumed < 3:
                    st = lines[j].strip()
                    if not st:
                        keep[j] = False
                        j += 1
                        consumed += 1
                        continue
                    # detect caps-name dominance
                    is_name_line, spans = detect_caps_name_line(st, cap_min_tokens, cap_min_prop)
                    if is_name_line or ACADEMIC_TITLES_RE.search(st):
                        if mask_token:
                            if spans:
                                masked = mask_spans(lines[j], spans, mask_token)
                                lines[j] = masked
                                # keep[j] stays True (masked)
                                if debug:
                                    print(f"[CUE] Masked after cue at {j}: {masked!r}")
                            else:
                                # if no spans, drop as fallback to be safe
                                keep[j] = False
                        else:
                            keep[j] = False
                        j += 1
                        consumed += 1
                        continue
                    break
                i = j
                continue
            i += 1
        return [l for l, k in zip(lines, keep) if k]

    # Run twice to catch a second front-matter block after a form-feed/page break
    lines = text.split("\n")
    lines = process_once(lines)
    lines = process_once(lines)  # second pass
    return "\n".join(lines)

# -------- Core cleaning passes --------

def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for k, v in TRANSLIT_MAP.items():
        text = text.replace(k, v)
    text = text.translate(ROM_SWAP)
    for rx, repl in ROM_DIACRITICS_FIXES:
        text = rx.sub(lambda m: ("Ș" if m.group(0)[0].isupper() else "ș") if repl == "ș"
                      else ("Ț" if m.group(0)[0].isupper() else "ț"), text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")  # NBSP
    text = text.translate({ord(c): None for c in ZWS})
    text = CONTROL_RE.sub("", text)
    return text

def fix_spacing(text: str, keep_tabs: bool) -> str:
    if not keep_tabs:
        text = SPACIFY_TABS_RE.sub("    ", text)
    fixed_lines = []
    for line in text.split("\n"):
        if line.strip() == "":
            fixed_lines.append("")
            continue
        m = re.match(r"^(\s*)(.*)$", line)
        indent, body = (m.group(1), m.group(2)) if m else ("", line)
        body = MULTISPACE_RE.sub("  ", body)  # collapse 3+ spaces to 2
        fixed_lines.append(indent + body.strip())
    return "\n".join(fixed_lines)

def unhyphenate(text: str) -> str:
    return UNHYPHEN_RE.sub("", text)

def strip_headers_footers(text: str) -> str:
    out = []
    for line in text.split("\n"):
        ls = line.strip()
        if not ls:
            out.append(line); continue
        if PAGE_LINE_RE.match(ls):
            continue
        if JUST_DIGITS_RE.match(ls):
            continue
        if RULE_LINE_RE.match(ls):
            continue
        out.append(line)
    return "\n".join(out)

def ner_caps_scrub_first_lines(text: str, *,
                               nlp_list: list,
                               max_lines: int,
                               ner_mask: str | None,
                               ner_drop: bool,
                               cap_detect: bool,
                               cap_mask: str | None,
                               cap_drop: bool,
                               cap_min_tokens: int,
                               cap_min_prop: float,
                               debug=False) -> str:
    lines = text.split("\n")
    limit = min(max_lines, len(lines))
    out = lines[:]  # copy

    for i in range(limit):
        s = lines[i]
        st = s.strip()
        if not st:
            continue

        # Skip if clearly a title/header
        if line_looks_like_title(st):
            if debug:
                print(f"[NAME] Skip title-like line {i}: {st!r}")
            continue

        # ----- NER layer (ensemble) -----
        if nlp_list:
            spans = person_spans_union_from_models(nlp_list, st)
            if ner_drop and spans:
                # name coverage ratio
                nonspace = len(re.sub(r"\s+", "", st))
                if nonspace and (sum(b - a for a, b in spans) / nonspace) >= 0.55:
                    if debug:
                        print(f"[NAME] Drop (NER-dominated) line {i}: {st!r}")
                    out[i] = ""
                    continue
            if ner_mask and spans:
                masked = mask_spans(st, spans, ner_mask)
                if masked != st:
                    if debug:
                        print(f"[NAME] Masked by NER (ensemble) line {i}: {st!r} -> {masked!r}")
                    out[i] = masked
                    st = masked  # continue with updated text

        # ----- ALL-CAPS heuristic layer -----
        if cap_detect:
            is_name_line, cap_spans = detect_caps_name_line(st, min_tokens=cap_min_tokens, min_prop=cap_min_prop)
            if is_name_line and cap_drop:
                if debug:
                    print(f"[NAME] Drop (CAPS-dominated) line {i}: {st!r}")
                out[i] = ""
                continue
            if cap_spans:
                token = cap_mask if cap_mask is not None else (ner_mask if ner_mask is not None else None)
                if token:
                    masked2 = mask_spans(out[i] if out[i] else st, cap_spans, token)
                    if masked2 != out[i]:
                        if debug:
                            print(f"[NAME] Masked by CAPS line {i}: {st!r} -> {masked2!r}")
                        out[i] = masked2

    return "\n".join(out)

def clean_text(raw: str, *,
               remove_biblio: bool,
               keep_tabs: bool, do_unhyphen: bool,
               use_ner: bool, nlp_list: list, ner_lines: int, ner_mask: str | None, ner_drop: bool,
               cap_detect: bool, cap_mask: str | None, cap_drop: bool, cap_min_tokens: int, cap_min_prop: float,
               use_id_cues: bool, id_cues_lines: int, id_mask: str | None,
               debug=False) -> str:
    t = normalize_unicode(raw)
    if do_unhyphen:
        t = unhyphenate(t)
    t = strip_headers_footers(t)

    # Cue-based identity removal FIRST (before NER/title heuristics), limited to front matter
    if use_id_cues and id_cues_lines > 0:
        t = remove_identity_by_cues(
            t,
            first_n_lines=id_cues_lines,
            cap_min_tokens=cap_min_tokens,
            cap_min_prop=cap_min_prop,
            mask_token=id_mask,
            debug=debug
        )

    # Name scrubbing on first lines (NER + CAPS)
    if (use_ner or cap_detect) and ner_lines > 0:
        t = ner_caps_scrub_first_lines(
            t,
            nlp_list=nlp_list,
            max_lines=ner_lines,
            ner_mask=ner_mask,
            ner_drop=ner_drop,
            cap_detect=cap_detect,
            cap_mask=cap_mask,
            cap_drop=cap_drop,
            cap_min_tokens=cap_min_tokens,
            cap_min_prop=cap_min_prop,
            debug=debug
        )

    if remove_biblio:
        t = remove_bibliography(t, debug=debug)

    t = fix_spacing(t, keep_tabs=keep_tabs)
    t = re.sub(r"\n{3,}", "\n\n", t).strip() + "\n"
    return t

# -------- I/O --------

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write_text(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def should_take(path: Path, exts) -> bool:
    return path.suffix.lower() in exts

def iter_files(root: Path, recursive: bool):
    if recursive:
        yield from root.rglob("*")
    else:
        yield from root.glob("*")

def main():
    args = parse_args()
    if not args.in_dir.is_dir():
        raise SystemExit(f"Input folder not found: {args.in_dir}")

    # Load spaCy models if NER is requested
    nlp_list = []
    if args.ner:
        nlp_list = try_load_spacy_multi(args.ner_model, args.ner_models, debug=args.debug)

    exts = {e.strip().lower() for e in args.ext.split(",") if e.strip()}
    total, cleaned = 0, 0

    for src in iter_files(args.in_dir, args.recursive):
        if not src.is_file() or not should_take(src, exts):
            continue
        total += 1
        rel = src.relative_to(args.in_dir)
        dst = args.out_dir / rel

        try:
            raw = read_text(src)
            out = clean_text(
                raw,
                remove_biblio=args.remove_biblio,
                keep_tabs=args.keep_tabs,
                do_unhyphen=args.unhyphen,
                use_ner=args.ner,
                nlp_list=nlp_list,
                ner_lines=args.ner_lines,
                ner_mask=args.ner_mask,
                ner_drop=args.ner_drop_name_lines,
                cap_detect=args.cap_detect,
                cap_mask=args.cap_mask,
                cap_drop=args.cap_drop_name_lines,
                cap_min_tokens=args.cap_min_tokens,
                cap_min_prop=args.cap_min_prop,
                use_id_cues=args.id_cues,
                id_cues_lines=args.id_cues_lines,
                id_mask=args.id_mask,
                debug=args.debug,
            )
            write_text(dst, out)
            cleaned += 1
            print(f"[OK] {rel} -> {dst}")
        except Exception as e:
            print(f"[ERR] {src}: {e}")

    print(f"\nDone. Cleaned {cleaned}/{total} files into: {args.out_dir}")

if __name__ == "__main__":
    main()

