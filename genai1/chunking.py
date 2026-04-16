#chunking.py 
from typing import List
import re
import math

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[dict]:
    """
    Split text into chunks with automatic page numbers.
    - If document has real page markers [Page n], use them.
    - If not, assign sequential fake page numbers based on chunk_size.
    """
    if not text:
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Detect pages from PDF markers
    pages = []
    page_pattern = re.compile(r"^\[Page\s+(\d+)\]", re.MULTILINE)
    matches = list(page_pattern.finditer(text))
    if matches:
        parts = page_pattern.split(text)
        i = 1
        while i < len(parts):
            page_num = int(parts[i])
            page_text = parts[i + 1].strip()
            pages.append((page_num, page_text))
            i += 2
    else:
        # If no page markers, treat entire text as one "page"
        pages = [(None, text)]

    chunks = []
    chunk_idx = 0

    for page_num, page_text in pages:
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        current = ""
        for para in paragraphs:
            if not current:
                current = para
                continue
            if len(current) + 2 + len(para) > chunk_size:
                # Assign page number: use real if available, else calculate fake page
                fake_page = page_num if page_num is not None else chunk_idx // max(1, (chunk_size // 200)) + 1
                chunks.append({
                    "text": current.strip(),
                    "page": fake_page,
                    "chunk_index": chunk_idx
                })
                chunk_idx += 1
                overlap_text = current[-overlap:] if overlap < len(current) else current
                current = (overlap_text + " " + para).strip()
            else:
                current = (current + "\n\n" + para).strip()
        if current:
            fake_page = page_num if page_num is not None else chunk_idx // max(1, (chunk_size // 200)) + 1
            chunks.append({
                "text": current.strip(),
                "page": fake_page,
                "chunk_index": chunk_idx
            })
            chunk_idx += 1

    # Further split very large chunks
    final_chunks = []
    for ch in chunks:
        txt = ch["text"]
        if len(txt) <= chunk_size:
            final_chunks.append(ch)
        else:
            start = 0
            L = len(txt)
            while start < L:
                end = start + chunk_size
                final_chunks.append({
                    "text": txt[start:end].strip(),
                    "page": ch["page"],
                    "chunk_index": ch["chunk_index"]
                })
                start = end - overlap
                if start < 0:
                    start = 0
                if start >= L:
                    break

    return final_chunks
