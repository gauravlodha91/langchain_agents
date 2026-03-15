"""
tesseract_converter.py
PDF → Markdown  |  Tesseract (text/tables) + Ollama LLaVA (diagrams/images)

Routing gate (per page):
  PyMuPDF native text > 100 chars  → Tesseract  (fast, accurate for real text)
  PyMuPDF native text ≤ 100 chars  → LLaVA      (vector diagrams + raster images)

This catches BOTH:
  - Raster images (photos, scanned charts)
  - Vector drawings (mind maps, flow diagrams, boxes + arrows)

Install:
    pip install pytesseract pymupdf opencv-python Pillow ollama
    ollama pull llava      ← one-time, ~4 GB
    ollama serve           ← keep running in a separate terminal
"""

import re
import base64
import logging
import time
from pathlib import Path

import fitz
import cv2
import numpy as np
import pytesseract
import ollama
from PIL import Image


# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _hms(seconds: float) -> str:
    """Format seconds as  Xm Ys  or  X.XXs  depending on magnitude."""
    if seconds >= 60:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.1f}s"
    return f"{seconds:.2f}s"


# ── Config ────────────────────────────────────────────────────────────────────

DPI          = 300      # Render DPI. 300 = balanced. 150 = faster.
LANG         = "eng"    # Tesseract language. "eng+hin" for multilingual.
PSM          = 3        # Page Segmentation Mode. 3 = fully automatic.
OEM          = 1        # OCR Engine. 1 = LSTM neural net (most accurate).
OLLAMA_MODEL = "llava"  # Vision model. "llava:13b" for higher accuracy.

# Pages whose PyMuPDF-extracted native text is shorter than this are treated
# as diagrams/images and routed to LLaVA instead of Tesseract.
# Raise this if diagram pages with some labels slip through to Tesseract.
# Lower this if short-text pages (title pages, TOC) are wrongly sent to LLaVA.
NATIVE_TEXT_THRESHOLD = 100   # characters

# ── Tesseract binary (Windows) ────────────────────────────────────────────────
# Run  `where tesseract`  in your terminal to get your exact path.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ── PDF loading ───────────────────────────────────────────────────────────────

def load_pdf(pdf_path: str):
    """
    Open the PDF with PyMuPDF and return the fitz.Document object.
    Kept separate so callers can iterate pages directly and access
    both the rendered image AND the native text in one pass.
    """
    return fitz.open(pdf_path)


# ── Page rendering ────────────────────────────────────────────────────────────

def render_page(page: fitz.Page) -> np.ndarray:
    """
    Rasterise a single fitz.Page to a BGR numpy array at the configured DPI.
    fitz internal DPI is 72 — scale = DPI / 72 gives the correct zoom factor.
    """
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ── Routing gate ──────────────────────────────────────────────────────────────

def get_native_text(page: fitz.Page) -> str:
    """
    Extract selectable/native text directly from the PDF page structure
    using PyMuPDF — no OCR involved, essentially instant.

    This works for:
      ✓ Digital PDFs with embedded text
      ✓ Vector diagrams where text labels are PDF text objects

    Returns empty string for:
      ✗ Scanned/raster-only pages (no text objects in PDF structure)
      ✗ Pages where text is part of an embedded image
    """
    return page.get_text("text").strip()


def should_use_llava(native_text: str) -> bool:
    """
    Routing decision: True → LLaVA,  False → Tesseract.

    Logic:
      If PyMuPDF can extract meaningful text (> NATIVE_TEXT_THRESHOLD chars),
      the page contains real readable text → Tesseract is fast and accurate.

      If native text is short or empty, the page is one of:
        - A scanned page (no text layer)
        - A vector diagram (boxes, arrows, mind maps)
        - A raster image (chart, photo)
      All of these are better handled by LLaVA's visual understanding.
    """
    return len(native_text) <= NATIVE_TEXT_THRESHOLD


# ── Preprocessing  (Tesseract only) ──────────────────────────────────────────

def preprocess(img: np.ndarray) -> np.ndarray:
    """
    grayscale → denoise → Otsu binarize.
    Applied only before Tesseract — LLaVA always receives the original
    full-colour image (colour matters for chart legends, bar colours, etc.)
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray  = cv2.fastNlMeansDenoising(gray, h=10)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


# ── Table detection ───────────────────────────────────────────────────────────

def has_table(img: np.ndarray) -> bool:
    """
    Detect ruled tables via morphological line detection.
    Looks for intersecting horizontal + vertical lines — both must be present.
    """
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w  = bw.shape
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1)))
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 20)))
    return h_lines.any() and v_lines.any()


# ── Tesseract extraction ──────────────────────────────────────────────────────

def extract_table(img: np.ndarray) -> str:
    """PSM 6 OCR → split columns on 2+ spaces → Markdown pipe table."""
    text  = pytesseract.image_to_string(
        Image.fromarray(img), lang=LANG, config=f"--psm 6 --oem {OEM}"
    )
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    rows     = [re.split(r"\s{2,}", l.strip()) for l in lines]
    max_cols = max(len(r) for r in rows)
    rows     = [r + [""] * (max_cols - len(r)) for r in rows]
    md  = "| " + " | ".join(rows[0]) + " |\n"
    md += "| " + " | ".join(["---"] * max_cols) + " |\n"
    for row in rows[1:]:
        md += "| " + " | ".join(row) + " |\n"
    return md


def extract_text(img: np.ndarray) -> str:
    """PSM 3 OCR → basic Markdown (headings + bullets)."""
    text = pytesseract.image_to_string(
        Image.fromarray(img), lang=LANG, config=f"--psm {PSM} --oem {OEM}"
    )
    md = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            md.append("")
        elif re.match(r"^[A-Z][A-Z\s]{4,50}$", line):
            md.append(f"## {line.title()}")
        elif re.match(r"^[•·▪\-\*]\s", line):
            md.append(f"- {line[2:].strip()}")
        else:
            md.append(line)
    return "\n".join(md)


# ── Ollama LLaVA ──────────────────────────────────────────────────────────────

def describe_with_llava(img: np.ndarray) -> str:
    """
    Send the full-colour page image to LLaVA via Ollama.
    Encodes as base64 PNG — Ollama's Python client accepts base64 strings
    in the `images` field of the message dict.

    Prompt is tuned for financial/technical documents:
      - Flow diagrams → describe nodes, arrows, relationships
      - Mind maps     → describe hierarchy and branch labels
      - Charts        → type, title, axes, key values
      - Mixed pages   → describe all sections
    """
    _, buf = cv2.imencode(".png", img)
    b64    = base64.b64encode(buf.tobytes()).decode("utf-8")

    prompt = (
        "You are analysing a page from a technical or financial document. "
        "Describe the content in clean, structured Markdown:\n"
        "- Flow diagram / mind map: describe each node, its connections, "
        "  and the overall hierarchy or flow direction.\n"
        "- Chart or graph: state type, title, axis labels, legend, "
        "  and key data points or trends.\n"
        "- Table without borders: extract as a Markdown pipe table.\n"
        "- Mixed content: cover each section clearly.\n"
        "Output Markdown only. No preamble or commentary."
    )

    try:
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [b64]}]
        )
        print("-" * 80)
        print()
        print("LLaVA response:", resp["message"]["content"].strip())  # Debug log for LLaVA output
        print()
        print("-" * 80)
        return resp["message"]["content"].strip()
    except Exception as e:
        log.warning(f"    Ollama error: {e}")
        return "> ⚠️ *Diagram/image — Ollama unavailable. Run `ollama serve` and retry.*"


# ── Page processor ────────────────────────────────────────────────────────────

def process_page(page: fitz.Page, page_num: int) -> str:
    """
    Full pipeline for one page:
      1. Extract native text via PyMuPDF (instant, no OCR).
      2. Render page to BGR image.
      3. Route:
           short native text → LLaVA  (diagram / image / scan)
           long  native text → Tesseract (text page, optionally table)
      4. Log time taken for each stage and the routing decision.

    Returns the Markdown string for this page.
    """
    page_start = time.perf_counter()
    log.info(f"── Page {page_num} ──────────────────────────")

    # Stage 1: native text extraction (PyMuPDF, ~instant)
    t0 = time.perf_counter()
    native_text = get_native_text(page)
    log.info(f"  [native text]  {len(native_text)} chars  ({_hms(time.perf_counter()-t0)})")

    # Stage 2: render to image
    t0 = time.perf_counter()
    img_bgr = render_page(page)
    log.info(f"  [render]       {img_bgr.shape[1]}×{img_bgr.shape[0]}px  ({_hms(time.perf_counter()-t0)})")

    parts = [f"## Page {page_num}\n"]

    # Stage 3: route
    if should_use_llava(native_text):
        log.info(f"  [route]        → LLaVA  (native text ≤ {NATIVE_TEXT_THRESHOLD} chars)")
        t0 = time.perf_counter()
        parts.append(describe_with_llava(img_bgr))
        log.info(f"  [llava]        done  ({_hms(time.perf_counter()-t0)})")

    else:
        preprocessed = preprocess(img_bgr)

        if has_table(preprocessed):
            log.info(f"  [route]        → Tesseract TABLE")
            t0 = time.perf_counter()
            parts.append(extract_table(preprocessed))
            log.info(f"  [tesseract]    table done  ({_hms(time.perf_counter()-t0)})")
        else:
            log.info(f"  [route]        → Tesseract TEXT")
            t0 = time.perf_counter()
            parts.append(extract_text(preprocessed))
            log.info(f"  [tesseract]    text done  ({_hms(time.perf_counter()-t0)})")

    elapsed = time.perf_counter() - page_start
    log.info(f"  [page total]   {_hms(elapsed)}")

    return "\n".join(parts)


# ── Main ─────────────────────────────────────────────────────────────────────

def convert(pdf_path: str, output_dir: str = "output") -> Path:
    """
    Convert a full PDF to a single Markdown file.
    Logs per-page and total timing for every stage.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    log.info(f"Starting conversion: {pdf_path}")
    log.info(f"Config: DPI={DPI}  lang={LANG}  model={OLLAMA_MODEL}  threshold={NATIVE_TEXT_THRESHOLD}")

    doc = load_pdf(pdf_path)
    log.info(f"Loaded PDF: {len(doc)} page(s)")

    md_pages = []
    for i, page in enumerate(doc, start=1):
        md_pages.append(process_page(page, i))

    doc.close()

    # Assemble and save
    t0       = time.perf_counter()
    markdown = "\n\n---\n\n".join(md_pages)
    out_file = out_dir / (Path(pdf_path).stem + ".md")
    out_file.write_text(markdown, encoding="utf-8")
    log.info(f"Saved → {out_file}  ({len(markdown):,} chars, write: {_hms(time.perf_counter()-t0)})")

    total = time.perf_counter() - total_start
    log.info(f"══ Total time: {_hms(total)} for {len(md_pages)} page(s) "
             f"(avg {_hms(total/max(len(md_pages),1))}/page) ══")

    return out_file


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="PDF → Markdown  (Tesseract + LLaVA)")
    p.add_argument("pdf",            help="Input PDF path")
    p.add_argument("--output", "-o", default="output", help="Output directory")
    args = p.parse_args()
    convert(args.pdf, args.output)