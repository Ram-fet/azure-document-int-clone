from typing import Optional
from fastapi import FastAPI
from fastapi import UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import logging
from pyzbar.pyzbar import decode as decode_barcodes  # for barcode detection
import os, tempfile
from uuid import uuid4
from io import StringIO
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from html import escape as html_escape, unescape as html_unescape


# pdfminer imports
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
# import translation parts
from translate_agent import TranslateRequest, translate_core

# ---------------- Logging setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("ocr_server.log"), logging.StreamHandler()],
)
logger = logging.getLogger("ocr_server")
logger.info("🚀 OCR server starting...")

# ---------------- FastAPI app ----------------
app = FastAPI(title="OCR + Translate API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
os.makedirs("generated", exist_ok=True)
app.mount("/generated", StaticFiles(directory="generated", html=True), name="generated")

# ---------------- OCR Endpoint ----------------
@app.post("/ocr")
async def run_ocr(
    file: UploadFile = File(...),
    page: int = Form(1),
    analysisRange: str = Form("current"),
    pageRange: str = Form("all"),
    barcodes: bool = Form(False),
    language: bool = Form(False),
    highRes: bool = Form(False),
    styleFont: bool = Form(False),
    formulas: bool = Form(False),
    layout: str = Form("paragraph"),   # word / line / paragraph
    searchText: Optional[str] = Form(None),   # text search
):
    logger.info("📥 Received OCR request")

    pdf_bytes = await file.read()
    images = convert_from_bytes(pdf_bytes, dpi=300 if highRes else 200)

    if page < 1 or page > len(images):
        return {"error": "Invalid page number"}

    image = images[page - 1].convert("RGB")
    width, height = image.size
    img_cv = np.array(image)

    lang = "eng+fra+deu" if language else "eng"
    results = []
    psm_map = {"word": 11, "line": 6, "paragraph": 3}
    custom_config = f"--psm {psm_map.get(layout, 3)}"
    data = pytesseract.image_to_data(image, lang=lang, config=custom_config, output_type=Output.DICT)

    if layout == "word":
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            x0, y0 = data["left"][i], data["top"][i]
            x1, y1 = x0 + data["width"][i], y0 + data["height"][i]
            results.append({"text": word, "box": [x0, y0, x1, y1]})
    elif layout == "line":
        line_groups = {}
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            line_no = data["line_num"][i]
            if line_no not in line_groups:
                line_groups[line_no] = {
                    "text": [],
                    "x0": data["left"][i],
                    "y0": data["top"][i],
                    "x1": data["left"][i] + data["width"][i],
                    "y1": data["top"][i] + data["height"][i],
                }
            line_groups[line_no]["text"].append(word)
            line_groups[line_no]["x0"] = min(line_groups[line_no]["x0"], data["left"][i])
            line_groups[line_no]["y0"] = min(line_groups[line_no]["y0"], data["top"][i])
            line_groups[line_no]["x1"] = max(line_groups[line_no]["x1"], data["left"][i] + data["width"][i])
            line_groups[line_no]["y1"] = max(line_groups[line_no]["y1"], data["top"][i] + data["height"][i])
        results = [
            {"text": " ".join(line["text"]), "box": [line["x0"], line["y0"], line["x1"], line["y1"]]}
            for line in line_groups.values()
        ]
    else:  # paragraph
        para_groups = {}
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            block_no = data["block_num"][i]
            if block_no not in para_groups:
                para_groups[block_no] = {
                    "text": [],
                    "x0": data["left"][i],
                    "y0": data["top"][i],
                    "x1": data["left"][i] + data["width"][i],
                    "y1": data["top"][i] + data["height"][i],
                }
            para_groups[block_no]["text"].append(word)
            para_groups[block_no]["x0"] = min(para_groups[block_no]["x0"], data["left"][i])
            para_groups[block_no]["y0"] = min(para_groups[block_no]["y0"], data["top"][i])
            para_groups[block_no]["x1"] = max(para_groups[block_no]["x1"], data["left"][i] + data["width"][i])
            para_groups[block_no]["y1"] = max(para_groups[block_no]["y1"], data["top"][i] + data["height"][i])
        results = [
            {"text": " ".join(block["text"]), "box": [block["x0"], block["y0"], block["x1"], block["y1"]]}
            for block in para_groups.values()
        ]

    barcode_results = []
    if barcodes:
        detected = decode_barcodes(img_cv)
        for b in detected:
            x, y, w, h = b.rect
            barcode_results.append({"data": b.data.decode("utf-8"), "box": [x, y, x + w, y + h]})

    search_matches = []
    if searchText:
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            if searchText.lower() in word.lower():
                x0, y0 = data["left"][i], data["top"][i]
                x1, y1 = x0 + data["width"][i], y0 + data["height"][i]
                search_matches.append({"text": word, "box": [x0, y0, x1, y1], "highlight": True})

    return {
        "ocr_results": results,
        "search_matches": search_matches if searchText else None,
        "barcode_results": barcode_results if barcodes else None,
        "page_size": [width, height],
        "page": page,
        "extracted_text": "\n\n".join([r["text"] for r in results]),
    }

# ---------------- Translate Endpoint ----------------
@app.post("/translate")
async def translate(req: TranslateRequest):
    return translate_core(req)


# ---------------- PDF → HTML ----------------# ---- imports (add if missing) ----
import re, os, tempfile
from uuid import uuid4
from fastapi import Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import fitz  # PyMuPDF

# one-time mount (keep if you already have it)
os.makedirs("generated", exist_ok=True)
app.mount("/generated", StaticFiles(directory="generated", html=True), name="generated")



# --- spacing/escaping helpers ---
import re
from html import escape as html_escape, unescape as html_unescape

_TAG_RE = re.compile(r"<[^>]+>")               # remove HTML tags

def _plain(s: str) -> str:
    return _TAG_RE.sub("", s)

def _needs_space(prev: str, curr: str) -> bool:
    """Add a space if previous char and next char are alnum (PDF often omits spaces)."""
    if not prev or not curr:
        return False
    return prev[-1].isalnum() and curr[0].isalnum()

def _merge_lines_html(lines_html):
    """
    Merge line-level HTML strings while:
      - de-hyphenating across lines
      - inserting spaces where needed
    """
    out = []
    for h in lines_html:
        if not out:
            out.append(h)
            continue
        prev_h = out[-1]
        prev_plain = _plain(prev_h).rstrip()
        curr_plain = _plain(h).lstrip()

        # de-hyphenate: '...-' + 'word' -> '...word'
        if prev_plain.endswith("-") and curr_plain[:1].islower():
            out[-1] = re.sub(r"-\s*$", "", prev_h)
            out.append(h.lstrip())
        else:
            if _needs_space(prev_plain[-1:], curr_plain[:1]):
                out.append(" " + h.lstrip())
            else:
                out.append(h)
    return "".join(out)

# -------- helpers --------


def _merge_lines(lines):
    buf = []
    for t in lines:
        if buf and buf[-1].endswith("-") and t and t[0].islower():
            buf[-1] = buf[-1][:-1] + t   # de-hyphenate
        else:
            buf.append(t)
    return " ".join(s.strip() for s in buf if s.strip())

def _page_median_font(blocks):
    sizes = []
    for b in blocks:
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                if sp.get("size"):
                    sizes.append(sp["size"])
    if not sizes:
        return 10.0
    sizes.sort()
    mid = len(sizes) // 2
    return sizes[mid] if len(sizes) % 2 else (sizes[mid-1] + sizes[mid]) / 2

_bullet_re = re.compile(r"^\s*(?:[\u2022\u25E6\u2023\u2043\u2219\-–—·•◦]|[0-9]+[.)])\s+")

def _span_flags(span):
    flags = span.get("flags", 0)
    font = (span.get("font") or "").lower()
    is_bold = bool(flags & 16) or "bold" in font
    is_italic = bool(flags & 2) or "italic" in font or "oblique" in font
    return is_bold, is_italic

def _block_to_html(block, median_size):
    if "lines" not in block:
        return None

    line_htmls = []
    max_span_size = 0.0
    list_like = True

    for ln in block["lines"]:
        pieces = []
        for sp in ln.get("spans", []):
            t = sp.get("text", "")
            if not t:
                continue
            s = sp.get("size") or 0.0
            max_span_size = max(max_span_size, s)
            bold, italic = _span_flags(sp)
            et = html_escape(t)

            frag = (
                f"<strong><em>{et}</em></strong>" if bold and italic else
                f"<strong>{et}</strong>" if bold else
                f"<em>{et}</em>" if italic else
                et
            )

            # insert a space between spans if PDF didn't include one
            if pieces:
                prev_plain = _plain(pieces[-1])
                curr_plain = _plain(frag)
                if _needs_space(prev_plain[-1:], curr_plain[:1]):
                    pieces.append(" ")
            pieces.append(frag)

        joined_html = "".join(pieces).strip()
        if not joined_html:
            continue

        # list-like? (check plain text)
        if not _bullet_re.search(_plain(joined_html)):
            list_like = False
        line_htmls.append(joined_html)

    if not line_htmls:
        return None

    # heading heuristic from largest span size
    if max_span_size >= median_size * 1.7:
        tag = "h1"
    elif max_span_size >= median_size * 1.35:
        tag = "h2"
    elif max_span_size >= median_size * 1.2:
        tag = "h3"
    else:
        tag = None

    if tag:
        # headings as plain text (no inline strong/em); join with spaces
        plain_lines = [_bullet_re.sub("", _plain(h)).strip() for h in line_htmls]
        heading_text = " ".join(p for p in plain_lines if p)
        return f"<{tag}>{html_escape(heading_text)}</{tag}>"

    # simple list detection
    plain_lines = [_bullet_re.sub("", _plain(h)).strip() for h in line_htmls]
    if list_like and len([p for p in plain_lines if p]) >= 2:
        items = "".join(f"<li>{html_escape(p)}</li>" for p in plain_lines if p)
        return f"<ul>{items}</ul>"

    # paragraph: merge *HTML* lines (preserve inline tags) – do NOT escape again
    para_html = _merge_lines_html(line_htmls)
    return f'<div class="para"><p>{para_html}</p></div>'

def build_semantic_html(pdf_bytes, title="Document"):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for page in doc:
        p = page.get_text("dict")
        blocks = [b for b in p.get("blocks", []) if "lines" in b]
        # top-to-bottom, left-to-right ordering helps robustness
        blocks.sort(key=lambda b: (round(b["bbox"][1], 1), b["bbox"][0]))
        median = _page_median_font(blocks)
        for b in blocks:
            html = _block_to_html(b, median)
            if html:
                parts.append(html)
    doc.close()

    body = "\n".join(parts)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{html_escape(title)} — Semantic HTML</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  :root {{ color-scheme: light dark }}
  body {{
    margin: 24px;
    background: #0f1115;
    color: #e6e6e6;
    font: 16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;
  }}
  h1, h2, h3 {{ margin: 1rem 0 .5rem; line-height: 1.2 }}
  h1 {{ font-size: 1.9rem }}
  h2 {{ font-size: 1.6rem }}
  h3 {{ font-size: 1.35rem }}
  .para p {{ margin: .65rem 0; }}
  ul {{ margin: .6rem 1.2rem; }}
</style>
</head>
<body>
  <div style="opacity:.75;margin-bottom:12px">
    Generated from: <strong>{html_escape(title)}</strong>
  </div>
  {body}
</body>
</html>"""

# -------- endpoint --------
@app.post("/pdf-to-html")
async def pdf_to_html_semantic(
    request: Request,
    file: UploadFile = File(...),
    page_range: str = Form("all"),  # reserved; not used in this basic version
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    html = build_semantic_html(pdf_bytes, title=file.filename or "Document")

    out_id = uuid4().hex
    out_dir = os.path.join("generated", out_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    url = request.url_for("generated", path=f"{out_id}/index.html")
    return JSONResponse({"url": str(url)})