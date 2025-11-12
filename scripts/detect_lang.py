#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan a folder of .md files, detect languages with lingua, export a CSV,
and optionally COPY (not move) files that are Romanian-only.

Install deps:
  pip install lingua-language-detector

Usage (non-recursive):
  python detect_lang.py /path/to/mds --out language_audit.csv

Recursive + custom thresholds/languages + copy Romanian-only:
  python detect_lang.py ./texts -r --min-conf 0.12 --top-n 2 --langs ro,en,fr,de \
    --copy-ro-only-to ./romanian_only
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from lingua import Language, LanguageDetectorBuilder

# ---- language code map (extend if needed) ----
LANG_CODE_MAP = {
    "en": Language.ENGLISH,
    "ro": Language.ROMANIAN,
    "fr": Language.FRENCH,
    "de": Language.GERMAN,
    "es": Language.SPANISH,
    "it": Language.ITALIAN,
    "pt": Language.PORTUGUESE,
    "ru": Language.RUSSIAN,
    "uk": Language.UKRAINIAN,
    "bg": Language.BULGARIAN,
    "hu": Language.HUNGARIAN,
    "pl": Language.POLISH,
    "nl": Language.DUTCH,
    "sv": Language.SWEDISH,
    "fi": Language.FINNISH,
    "da": Language.DANISH,
    "tr": Language.TURKISH,
    "el": Language.GREEK,
    "cs": Language.CZECH,
    "sk": Language.SLOVAK,
    "sr": Language.SERBIAN,
    "hr": Language.CROATIAN,
    "ja": Language.JAPANESE,
}

DEFAULT_LANGS = ["en", "ro", "fr", "de", "es", "hu", "el"]


def build_detector(lang_codes):
    langs = [LANG_CODE_MAP[c] for c in lang_codes if c in LANG_CODE_MAP]
    if not langs:
        raise SystemExit("No valid languages after parsing --langs.")
    return (
        LanguageDetectorBuilder
        .from_languages(*langs)
        .with_preloaded_language_models()
        .build()
    )


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def select_confidences(conf_values, min_conf: float, top_n: int | None):
    """
    Keep only languages with confidence >= min_conf, optionally limited to top_n.
    Returns [{"lang": "ROMANIAN", "value": 0.97}, ...]
    """
    kept = []
    for c in conf_values:
        val = float(c.value)
        if val < min_conf:
            continue
        kept.append({"lang": c.language.name, "value": val})
        if top_n is not None and len(kept) >= top_n:
            break
    return kept


def copy_preserve_tree(src: Path, src_root: Path, dst_root: Path):
    rel = src.relative_to(src_root)
    dst_path = dst_root / rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_path)
    return dst_path


def main():
    ap = argparse.ArgumentParser(description="Language audit for .md files using lingua (no max-drop).")
    ap.add_argument("folder", type=Path, help="Folder containing .md files")
    ap.add_argument("--out", default="language_audit.csv", help="Output CSV path")
    ap.add_argument("--recursive", "-r", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--langs", default=",".join(DEFAULT_LANGS),
                    help="Comma-separated language codes to consider (e.g., ro,en,fr,de)")
    ap.add_argument("--min-conf", type=float, default=0.10, help="Minimum confidence to keep a language (default 0.10)")
    ap.add_argument("--top-n", type=int, default=None, help="Keep at most N languages per file (default: keep all ≥ min-conf)")
    ap.add_argument("--max-chars", type=int, default=30000, help="Analyze at most this many characters per file (default 30000)")
    ap.add_argument("--copy-ro-only-to", type=Path, default=None,
                    help="If set, COPY files that are ROMANIAN-only (among languages ≥ --min-conf) to this folder, preserving the directory tree.")
    args = ap.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"Folder not found: {args.folder}")

    if args.copy_ro_only_to:
        args.copy_ro_only_to.mkdir(parents=True, exist_ok=True)

    lang_codes = [c.strip().lower() for c in args.langs.split(",") if c.strip()]
    detector = build_detector(lang_codes)

    # Collect .md files
    if args.recursive:
        files = sorted(args.folder.rglob("*.md"))
    else:
        files = sorted(args.folder.glob("*.md"))

    if not files:
        print("No .md files found.")
        return

    counts = {}  # top language counts
    copied = 0

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "chars", "top_lang", "top_conf",
                        "kept_languages_json", "min_conf", "top_n", "langs"]
        )
        writer.writeheader()

        for p in files:
            try:
                text = read_text(p)
                if args.max_chars and len(text) > args.max_chars:
                    text = text[:args.max_chars]

                confs = detector.compute_language_confidence_values(text or "")

                # For CSV/reporting, respect the provided --top-n
                kept_for_csv = select_confidences(confs, min_conf=args.min_conf, top_n=args.top_n)
                top_lang = kept_for_csv[0]["lang"] if kept_for_csv else ""
                top_conf = kept_for_csv[0]["value"] if kept_for_csv else 0.0
                counts[top_lang] = counts.get(top_lang, 0) + 1

                writer.writerow({
                    "path": str(p),
                    "chars": len(text),
                    "top_lang": top_lang,
                    "top_conf": f"{top_conf:.3f}",
                    "kept_languages_json": json.dumps(kept_for_csv, ensure_ascii=False),
                    "min_conf": args.min_conf,
                    "top_n": "" if args.top_n is None else args.top_n,
                    "langs": ",".join(lang_codes),
                })

                # For Romanian-only COPY decision, ignore --top-n to avoid accidental filtering
                kept_all = select_confidences(confs, min_conf=args.min_conf, top_n=None)
                is_ro_only = (len(kept_all) == 1 and kept_all[0]["lang"] == "ROMANIAN")

                if args.copy_ro_only_to and is_ro_only:
                    dst = copy_preserve_tree(p, args.folder, args.copy_ro_only_to)
                    copied += 1
                    print(f"[COPY] {p.name:40} -> {dst}")

                print(f"[OK]   {p.name:40} -> {top_lang or 'N/A'} ({top_conf:.3f})  chars={len(text)}")

            except Exception as e:
                print(f"[ERR]  {p}: {e}")

    print("\nTop language counts:")
    for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        label = k or "N/A"
        print(f"  {label:10} {v}")
    if args.copy_ro_only_to:
        print(f"\nCopied Romanian-only files: {copied} -> {args.copy_ro_only_to}")
    print(f"\nCSV saved to: {args.out}")


if __name__ == "__main__":
    main()

