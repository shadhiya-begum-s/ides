#ingest.py 
import uuid
from io import BytesIO
import os
import PyPDF2
import docx
from bs4 import BeautifulSoup

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    Image = None
    OCR_AVAILABLE = False

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploaded_docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_uploaded_file_bytes(name: str, b: bytes) -> str:
    """Save uploaded file bytes to disk and return path."""
    path = os.path.join(UPLOAD_DIR, name)
    with open(path, "wb") as f:
        f.write(b)
    return path

def extract_text_from_pdf_bytes(b: bytes) -> str:
    """Extract text from PDF bytes. Uses OCR if no text found."""
    bio = BytesIO(b)
    reader = PyPDF2.PdfReader(bio)
    parts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text and OCR_AVAILABLE:
            try:
                from pdf2image import convert_from_bytes
                images = convert_from_bytes(b, first_page=i+1, last_page=i+1)
                if images:
                    text = pytesseract.image_to_string(images[0])
            except Exception:
                text = ""
        parts.append(f"[Page {i+1}]\n{text.strip()}")
    return "\n\n".join(parts)

def extract_text_from_docx_bytes(b: bytes) -> str:
    """Extract text from DOCX bytes."""
    bio = BytesIO(b)
    doc = docx.Document(bio)
    parts = []
    for i, para in enumerate(doc.paragraphs, start=1):
        if para.text and para.text.strip():
            parts.append(f"[Paragraph {i}]\n{para.text.strip()}")
    return "\n\n".join(parts)

def extract_text_from_html_bytes(b: bytes) -> str:
    """Extract text from HTML bytes using BeautifulSoup."""
    try:
        soup = BeautifulSoup(b, "html.parser")
        return soup.get_text(separator="\n")
    except Exception:
        return b.decode("utf-8", errors="ignore")

def ingest_file(uploaded_file, from_disk: bool = False):
    """
    Ingest a file either from uploaded Streamlit file or from disk.
    
    Parameters:
    - uploaded_file: file object (Streamlit upload) or file path string if from_disk=True
    - from_disk: True if uploaded_file is a path on disk
    """
    if from_disk:
        name = os.path.basename(uploaded_file)
        with open(uploaded_file, "rb") as f:
            raw = f.read()
    else:
        raw = uploaded_file.read()
        name = uploaded_file.name

    # Save uploaded file to disk
    _ = save_uploaded_file_bytes(name, raw)

    # Extract text based on file type
    if name.lower().endswith(".pdf"):
        text = extract_text_from_pdf_bytes(raw)
    elif name.lower().endswith(".docx"):
        text = extract_text_from_docx_bytes(raw)
    elif name.lower().endswith(".html") or name.lower().endswith(".htm"):
        text = extract_text_from_html_bytes(raw)
    else:
        try:
            text = raw.decode("utf-8")
        except Exception:
            text = raw.decode("latin-1", errors="ignore")

    return {"id": uuid.uuid4().hex, "name": name, "text": text}