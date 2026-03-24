"""
PDF OCR Pipeline — PyMuPDF (fitz) + Tesseract
==============================================

For each page:
  - If native text exists  → extract text directly via fitz
  - If scanned/image page  → run Tesseract OCR

Outputs:
  - output.md   : clean readable Markdown
  - output.json : structured data with text blocks + image metadata

Install:
    pip install pymupdf pytesseract pillow

Tesseract must be installed locally:
    Linux  : sudo apt install tesseract-ocr
    macOS  : brew install tesseract
    Windows: https://github.com/UB-Mannheim/tesseract/wiki



    This is Gaurav Lodha File
"""

import os
import io
import json
import argparse
from pathlib import Path
from typing import Optional

import fitz                      # pip install pymupdf
import pytesseract
from pytesseract import Output
from PIL import Image


# ──────────────────────────────────────────────
# 1. Tesseract path setup
# ──────────────────────────────────────────────

def setup_tesseract(path: Optional[str] = None) -> None:
    """
    Point pytesseract at the local Tesseract binary.
    Pass --tesseract-path explicitly, or set TESSERACT_CMD env var,
    or let it auto-detect from common install locations.
    """
    candidates = [
        path,
        os.environ.get("TESSERACT_CMD"),
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            pytesseract.pytesseract.tesseract_cmd = c
            print(f"  Tesseract: {c}")
            return
    print("  Tesseract: using system PATH")


# ──────────────────────────────────────────────
# 2. Page helpers
# ──────────────────────────────────────────────

def page_to_pil(page: fitz.Page, dpi: int = 300) -> Image.Image:
    """Render a PDF page to a PIL image at the given DPI."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def has_native_text(page: fitz.Page, min_chars: int = 50) -> bool:
    """
    Return True only if the page has meaningful embedded text.
    Threshold is 50 chars — low char counts usually mean
    the page is a scanned image with just a few stray characters.
    """
    return len(page.get_text("text").strip()) >= min_chars


# ──────────────────────────────────────────────
# 3. Native text extraction (fitz)
# ──────────────────────────────────────────────

def extract_native_text(page: fitz.Page) -> list[dict]:
    """
    Extract text blocks from the PDF's embedded text layer via fitz.
    Uses 'dict' mode which preserves reading order per block.
    """
    results = []
    # Use 'dict' (not 'rawdict') — more reliable line grouping
    data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

    for block in data.get("blocks", []):
        if block["type"] != 0:
            continue

        lines_text = []
        for line in block.get("lines", []):
            line_str = " ".join(
                span["text"].strip()
                for span in line.get("spans", [])
                if span.get("text", "").strip()
            )
            if line_str:
                lines_text.append(line_str)

        if lines_text:
            x0, y0, x1, y1 = block["bbox"]
            results.append({
                "type": "text",
                "source": "native",
                "text": "\n".join(lines_text),
                "bbox": {"x0": round(x0, 1), "y0": round(y0, 1),
                         "x1": round(x1, 1), "y1": round(y1, 1)},
            })
    return results


# ──────────────────────────────────────────────
# 4. OCR text extraction (Tesseract)
# ──────────────────────────────────────────────

def extract_ocr_text(
    page: fitz.Page,
    dpi: int = 300,
    lang: str = "eng",
    oem: int = 3,
    psm: int = 1,          # PSM 1 = auto with OSD — best for mixed-layout pages
) -> list[dict]:
    """
    Run Tesseract on the rendered page image.

    PSM modes that work well:
      1  = Auto page segmentation with OSD  ← best for most real docs
      3  = Fully automatic (no OSD)
      6  = Single uniform block of text

    Words are grouped into lines using Tesseract's own
    block_num / par_num / line_num — this preserves reading order
    correctly even for multi-column layouts.
    """
    pil_img = page_to_pil(page, dpi=dpi)
    scale = dpi / 72.0
    config = f"--oem {oem} --psm {psm}"

    data = pytesseract.image_to_data(
        pil_img, lang=lang, config=config, output_type=Output.DICT
    )

    # Group words into lines by (block_num, par_num, line_num)
    line_map: dict[tuple, dict] = {}
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue

        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lx = data["left"][i]
        ly = data["top"][i]
        lw = data["width"][i]
        lh = data["height"][i]

        if key not in line_map:
            line_map[key] = {
                "words": [], "confs": [],
                "x0": lx, "y0": ly, "x1": lx + lw, "y1": ly + lh
            }
        line_map[key]["words"].append(word)
        line_map[key]["confs"].append(conf)
        line_map[key]["x0"] = min(line_map[key]["x0"], lx)
        line_map[key]["y0"] = min(line_map[key]["y0"], ly)
        line_map[key]["x1"] = max(line_map[key]["x1"], lx + lw)
        line_map[key]["y1"] = max(line_map[key]["y1"], ly + lh)

    # Now group lines into paragraphs by (block_num, par_num)
    # This gives natural paragraph breaks instead of one line at a time
    para_map: dict[tuple, dict] = {}
    for (blk, par, ln), line in sorted(line_map.items()):
        key = (blk, par)
        line_text = " ".join(line["words"])
        avg_conf = sum(line["confs"]) / len(line["confs"])

        if key not in para_map:
            para_map[key] = {
                "lines": [], "confs": [],
                "x0": line["x0"], "y0": line["y0"],
                "x1": line["x1"], "y1": line["y1"]
            }
        para_map[key]["lines"].append(line_text)
        para_map[key]["confs"].append(avg_conf)
        para_map[key]["x0"] = min(para_map[key]["x0"], line["x0"])
        para_map[key]["y0"] = min(para_map[key]["y0"], line["y0"])
        para_map[key]["x1"] = max(para_map[key]["x1"], line["x1"])
        para_map[key]["y1"] = max(para_map[key]["y1"], line["y1"])

    results = []
    for para in para_map.values():
        x0 = round(para["x0"] / scale, 1)
        y0 = round(para["y0"] / scale, 1)
        x1 = round(para["x1"] / scale, 1)
        y1 = round(para["y1"] / scale, 1)
        avg_conf = round(sum(para["confs"]) / len(para["confs"]), 1)
        results.append({
            "type": "text",
            "source": "ocr",
            "text": "\n".join(para["lines"]),
            "confidence": avg_conf,
            "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        })

    # Sort top-to-bottom, left-to-right
    results.sort(key=lambda b: (b["bbox"]["y0"], b["bbox"]["x0"]))
    return results


# ──────────────────────────────────────────────
# 5. Image block extraction (fitz)
# ──────────────────────────────────────────────

def extract_image_blocks(page: fitz.Page) -> list[dict]:
    """
    Find all embedded image blocks on the page.
    Returns metadata only (bbox, size) — images are not saved to disk.
    """
    results = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 1:
            continue
        x0, y0, x1, y1 = block["bbox"]
        results.append({
            "type": "image",
            "source": "embedded",
            "bbox": {"x0": round(x0, 1), "y0": round(y0, 1),
                     "x1": round(x1, 1), "y1": round(y1, 1)},
            "width_pt":  round(x1 - x0, 1),
            "height_pt": round(y1 - y0, 1),
        })
    return results


# ──────────────────────────────────────────────
# 6. Process single page
# ──────────────────────────────────────────────

def process_page(
    page: fitz.Page,
    page_no: int,
    force_ocr: bool = False,
    dpi: int = 300,
    lang: str = "eng",
    oem: int = 3,
    psm: int = 1,
) -> dict:
    """
    Process one PDF page. Decides strategy, extracts text + image blocks.
    Returns a structured dict ready for Markdown or JSON output.
    """
    use_ocr = force_ocr or not has_native_text(page)
    strategy = "ocr" if use_ocr else "native"

    if use_ocr:
        text_blocks = extract_ocr_text(page, dpi=dpi, lang=lang, oem=oem, psm=psm)
    else:
        text_blocks = extract_native_text(page)

    image_blocks = extract_image_blocks(page)

    # Merge and sort all blocks top-to-bottom
    all_blocks = sorted(
        text_blocks + image_blocks,
        key=lambda b: (b["bbox"]["y0"], b["bbox"]["x0"])
    )

    plain_text = "\n\n".join(
        b["text"] for b in all_blocks if b["type"] == "text" and b["text"].strip()
    )

    rect = page.rect
    return {
        "page": page_no,
        "width_pt": round(rect.width, 1),
        "height_pt": round(rect.height, 1),
        "strategy": strategy,
        "image_count": len(image_blocks),
        "text_block_count": len(text_blocks),
        "plain_text": plain_text,
        "blocks": all_blocks,
    }


# ──────────────────────────────────────────────
# 7. Process full PDF
# ──────────────────────────────────────────────

def process_pdf(
    pdf_path: str,
    force_ocr: bool = False,
    dpi: int = 300,
    lang: str = "eng",
    oem: int = 3,
    psm: int = 1,
    password: str = "",
    tesseract_path: Optional[str] = None,
) -> list[dict]:
    """Open and process every page. Returns list of page result dicts."""
    setup_tesseract(tesseract_path)

    doc = fitz.open(pdf_path)
    if password:
        doc.authenticate(password)

    pages = []
    for i in range(doc.page_count):
        pg = doc[i]
        print(f"  Page {i+1}/{doc.page_count} ...", end=" ")
        result = process_page(pg, i + 1, force_ocr=force_ocr,
                              dpi=dpi, lang=lang, oem=oem, psm=psm)
        print(f"strategy={result['strategy']}  "
              f"text_blocks={result['text_block_count']}  "
              f"images={result['image_count']}")
        pages.append(result)

    doc.close()
    return pages


# ──────────────────────────────────────────────
# 8. Markdown output
# ──────────────────────────────────────────────

def to_markdown(pages: list[dict], pdf_path: str) -> str:
    """
    Clean Markdown output:
      - Text printed as plain paragraphs
      - Image blocks noted as inline placeholders with their position
    """
    lines = [f"# {Path(pdf_path).name}\n"]

    for page in pages:
        lines.append(f"## Page {page['page']}\n")
        lines.append(
            f"> strategy: `{page['strategy']}` | "
            f"images: {page['image_count']} | "
            f"text blocks: {page['text_block_count']}\n"
        )

        for block in page["blocks"]:
            if block["type"] == "text":
                lines.append(block["text"])
                lines.append("")
            elif block["type"] == "image":
                bb = block["bbox"]
                lines.append(
                    f"*[Image — {block['width_pt']} × {block['height_pt']} pt"
                    f" at ({bb['x0']}, {bb['y0']})]*"
                )
                lines.append("")

        lines.append("---\n")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# 9. JSON output
# ──────────────────────────────────────────────

def to_json(pages: list[dict], output_path: str) -> None:
    """
    Save full structured data to JSON.

    JSON structure per page:
      page             : page number (1-based)
      width_pt         : page width in PDF points
      height_pt        : page height in PDF points
      strategy         : "native" or "ocr"
      image_count      : number of embedded image blocks found
      text_block_count : number of text blocks extracted
      plain_text       : all page text joined — easy to read / feed to NLP
      blocks[]         : all content blocks sorted top-to-bottom
        text block:
          type         : "text"
          source       : "native" (fitz) | "ocr" (Tesseract)
          text         : extracted paragraph text
          confidence   : average OCR confidence 0-100 (ocr blocks only)
          bbox         : {x0, y0, x1, y1} in PDF points
        image block:
          type         : "image"
          source       : "embedded"
          bbox         : {x0, y0, x1, y1} in PDF points
          width_pt     : image width in PDF points
          height_pt    : image height in PDF points
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    print(f"  JSON → {output_path}")


# ──────────────────────────────────────────────
# 10. CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PDF OCR → Markdown + JSON  (PyMuPDF + Tesseract)"
    )
    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument("-o", "--output", help="Output .md path (default: <name>.md)")
    parser.add_argument("--json", help="Output .json path (default: <name>.json)")
    parser.add_argument("--force-ocr", action="store_true",
                        help="Always use Tesseract on every page")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Render DPI for OCR pages (default: 300)")
    parser.add_argument("--lang", default="eng",
                        help="Tesseract language(s), e.g. eng+hin (default: eng)")
    parser.add_argument("--oem", type=int, default=3,
                        help="Tesseract OEM 0-3 (default: 3)")
    parser.add_argument("--psm", type=int, default=1,
                        help="Tesseract PSM 0-13 (default: 1 = auto+OSD)")
    parser.add_argument("--password", default="",
                        help="PDF password if encrypted")
    parser.add_argument("--tesseract-path", default=None,
                        help="Local path to tesseract binary")
    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        print(f"[error] File not found: {args.pdf}")
        return

    stem     = Path(args.pdf).stem
    out_md   = args.output or f"{stem}.md"
    out_json = args.json   or f"{stem}.json"

    print(f"\nInput  : {args.pdf}")
    print(f"Output : {out_md}  +  {out_json}\n")

    pages = process_pdf(
        pdf_path=args.pdf,
        force_ocr=args.force_ocr,
        dpi=args.dpi,
        lang=args.lang,
        oem=args.oem,
        psm=args.psm,
        password=args.password,
        tesseract_path=args.tesseract_path,
    )

    md = to_markdown(pages, args.pdf)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  MD   → {out_md}")

    to_json(pages, out_json)
    print(f"\nDone — {len(pages)} page(s) processed.\n")


if __name__ == "__main__":
    main()
