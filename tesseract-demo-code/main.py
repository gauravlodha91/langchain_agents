"""
tesseract_converter.py
PDF → Markdown using Tesseract OCR + PyMuPDF.
Handles: text, tables, images.

Install:
    pip install pytesseract pymupdf opencv-python Pillow
    (No Poppler needed — PyMuPDF works natively on Windows/Mac/Linux)
"""

import re
import logging
from pathlib import Path

import fitz                  # PyMuPDF — renders PDF pages, no Poppler needed
import cv2
import numpy as np
import pytesseract
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

DPI   = 300   # Render quality. 300 = good for finance docs. 150 = faster.
LANG  = "eng" # Tesseract language. "eng+hin" for multilingual.
PSM   = 3     # Page Segmentation Mode. 3 = auto (best for mixed pages).
OEM   = 1     # OCR Engine. 1 = LSTM neural net (most accurate).

# ── Point pytesseract to your local Tesseract install ────────────────────────
# Update this path to wherever tesseract.exe lives on YOUR machine.
# Run  `where tesseract`  in your terminal to find it.
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ── PDF → page images (PyMuPDF, no Poppler) ───────────────────────────────────

def pdf_to_images(pdf_path: str) -> list[np.ndarray]:
    """
    Render each PDF page to a numpy BGR image using PyMuPDF (fitz).
    fitz.Matrix scales the page to the target DPI (default 72 dpi internally).
    scale = DPI / 72 gives the correct zoom factor.
    Returns a list of numpy arrays, one per page.
    """
    doc   = fitz.open(pdf_path)
    scale = DPI / 72
    mat   = fitz.Matrix(scale, scale)   # zoom matrix for target DPI
    imgs  = []
    for page in doc:
        # get_pixmap renders the page; colorspace=fitz.csRGB gives an RGB pixmap
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        # Convert pixmap bytes → numpy array → BGR for OpenCV
        arr  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        imgs.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    doc.close()
    return imgs


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Clean the page image before OCR.
    grayscale → denoise → Otsu binarize.
    These three steps noticeably boost Tesseract accuracy on scanned docs.
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray  = cv2.fastNlMeansDenoising(gray, h=10)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


# ── Table detection & extraction ──────────────────────────────────────────────

def has_table(img: np.ndarray) -> bool:
    """
    Detect tables by finding intersecting horizontal + vertical ruled lines
    via morphological operations. Returns True if both line types are found.
    """
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w  = bw.shape
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1)))
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 20)))
    return h_lines.any() and v_lines.any()


def extract_table(img: np.ndarray) -> str:
    """
    OCR a table with PSM 6 (uniform block) and render as a Markdown pipe table.
    Splits columns on 2+ consecutive spaces — works well for ruled tables.
    """
    text  = pytesseract.image_to_string(
        Image.fromarray(img), lang=LANG, config=f"--psm 6 --oem {OEM}"
    )
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return ""

    rows     = [re.split(r"\s{2,}", l.strip()) for l in lines]
    max_cols = max(len(r) for r in rows)
    rows     = [r + [""] * (max_cols - len(r)) for r in rows]  # pad short rows

    md  = "| " + " | ".join(rows[0]) + " |\n"
    md += "| " + " | ".join(["---"] * max_cols) + " |\n"
    for row in rows[1:]:
        md += "| " + " | ".join(row) + " |\n"
    return md


# ── Page OCR ─────────────────────────────────────────────────────────────────

def ocr_page(img: np.ndarray, page_num: int) -> str:
    """
    OCR one page.
    Table detected → extract_table (pipe table Markdown).
    Otherwise      → image_to_string + text_to_markdown.
    """
    parts = [f"## Page {page_num}\n"]

    if has_table(img):
        log.info(f"  Page {page_num}: table detected")
        parts.append(extract_table(img))
    else:
        text = pytesseract.image_to_string(
            Image.fromarray(img), lang=LANG, config=f"--psm {PSM} --oem {OEM}"
        )
        parts.append(text_to_markdown(text))

    return "\n".join(parts)


def text_to_markdown(text: str) -> str:
    """
    Lightweight Tesseract output → Markdown conversion.
    ALL-CAPS short lines → ## Heading | bullet chars → - list item.
    """
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


# ── Main ─────────────────────────────────────────────────────────────────────

def convert(pdf_path: str, output_dir: str = "output") -> Path:
    """Convert a PDF to a Markdown file. Returns path to saved .md file."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Rasterising {pdf_path} at {DPI} DPI …")
    pages = pdf_to_images(pdf_path)          # ← PyMuPDF, no Poppler

    md_pages = []
    for i, img in enumerate(pages, start=1):
        log.info(f"  OCR page {i}/{len(pages)}")
        img = preprocess(img)
        md_pages.append(ocr_page(img, i))

    markdown = "\n\n---\n\n".join(md_pages)
    out_file  = out_dir / (Path(pdf_path).stem + ".md")
    out_file.write_text(markdown, encoding="utf-8")
    log.info(f"Saved → {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="PDF → Markdown via Tesseract + PyMuPDF")
    p.add_argument("pdf",            help="Input PDF path")
    p.add_argument("--output", "-o", default="output", help="Output directory")
    args = p.parse_args()
    convert(args.pdf, args.output)