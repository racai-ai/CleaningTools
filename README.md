# CleaningTools

Small set of utilities for processing thesis archives:

- `scripts/pdf2txt_markit.py` – wraps MarkItDown to batch convert PDFs to Markdown-ish text while logging rough word counts.
- `scripts/clean_text.py` – scrubs the raw text (Romanian diacritics fixes, header/footer removal, bibliography trimming, optional NER/caps masking for names) and mirrors the folder tree.
- `scripts/detect_lang.py` – runs lingua on `.md` files, writes a CSV with confidences, and can optionally copy everything that’s confidently Romanian into a separate folder.
