import tempfile
from pathlib import Path

TEXT_EXT = {".txt", ".md", ".csv", ".json", ".html", ".htm", ".xml", ".tsv", ""}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}


def read_text_file(path):
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16"), "utf-16"
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1 (fallback, utf-8 failed)"


def pdf_pages_to_images(path, max_pages=20):
    import pdfplumber
    out_dir = Path(tempfile.mkdtemp(prefix="docagent_pdf_"))
    images = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages]):
            img_path = out_dir / f"page_{i + 1:03d}.png"
            page.to_image(resolution=200).save(str(img_path))
            images.append(str(img_path))
    return images


def load(path):
    p = Path(path)
    ext = p.suffix.lower()
    doc = {
        "path": str(p),
        "name": p.name,
        "format": ext.lstrip(".") or "plain",
        "bytes": p.stat().st_size,
        "kind": "text",
        "text": "",
        "images": [],
        "pages": None,
        "notes": [],
    }
    if ext in IMAGE_EXT:
        doc["kind"] = "image"
        doc["images"] = [str(p)]
        doc["pages"] = 1
        return doc
    if ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(p) as pdf:
            doc["pages"] = len(pdf.pages)
            doc["text"] = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
        if len(doc["text"].strip()) < 20:
            doc["kind"] = "image"
            doc["images"] = pdf_pages_to_images(p)
            doc["notes"].append("PDF has no text layer, treated as scanned pages")
        return doc
    if ext == ".docx":
        import docx
        d = docx.Document(str(p))
        doc["text"] = "\n".join(par.text for par in d.paragraphs)
        return doc
    text, enc = read_text_file(p)
    doc["text"] = text
    doc["encoding"] = enc
    if "fallback" in enc:
        doc["notes"].append("file was not valid UTF-8, decoded as latin-1")
    return doc
