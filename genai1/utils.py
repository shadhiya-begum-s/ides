#utils.py 
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

def safe(text):
    """Sanitize text for PDF output."""
    if not text:
        return ""
    return str(text).replace("&", "&").replace("<", "<").replace(">", ">")

def export_chat_to_pdf_bytes(chat_history):
    """
    Export chat history to PDF and return as bytes.
    Shows references per answer, or 'none (generated — not in documents)' if answer not from docs.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    for i, msg in enumerate(chat_history, start=1):
        # Question
        story.append(Paragraph(f"<b>Q{i}:</b> {safe(msg.get('question'))}", styles["Heading3"]))
        story.append(Spacer(1, 0.1 * inch))

        # Answer
        story.append(Paragraph(f"<b>A{i}:</b> {safe(msg.get('answer'))}", styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))

        # References
        refs = msg.get("refs", [])
        if refs:
            story.append(Paragraph("<i>References:</i>", styles["Italic"]))
            for r in refs:
                story.append(Paragraph(f"- {safe(r)}", styles["Normal"]))
            story.append(Spacer(1, 0.2 * inch))
        else:
            # Explicitly mark generated answers
            story.append(Paragraph("- References: none (generated — not in documents)", styles["Normal"]))
            story.append(Spacer(1, 0.2 * inch))

        story.append(Spacer(1, 0.2 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()

def format_reference(meta):
    """
    Format reference for a document chunk.
    Always prefer page number; fallback to paragraph index only if page is missing.
    """
    source = meta.get("source", "unknown")
    page = meta.get("page")
    chunk = meta.get("chunk_index", "?")
    if page is not None:
        return f"{source}: Page {page}"
    return f"{source}: Paragraph {chunk}"