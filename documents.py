"""Document processing module — extract text from photos, PDFs, DOCX files."""

import io
import os
import tempfile
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""


async def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        text_parts = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return ""


async def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from an image using OCR (pytesseract) or OpenAI Vision."""
    # Try pytesseract first
    try:
        import pytesseract
        from PIL import Image
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image, lang='rus+eng')
        if text.strip():
            return text.strip()
    except Exception as e:
        logger.warning(f"Tesseract OCR failed: {e}")

    # Fallback: return empty (Vision API will be used in bot.py for image understanding)
    return ""


async def extract_text_from_image_openai(file_bytes: bytes, openai_client) -> str:
    """Extract text from image using OpenAI Vision API."""
    import base64
    try:
        b64 = base64.b64encode(file_bytes).decode('utf-8')
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Извлеки ВЕСЬ текст с этого изображения/документа. Верни только текст, без комментариев."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}}
                ]
            }],
            max_tokens=4096
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI Vision extraction error: {e}")
        return ""


async def process_document(file_bytes: bytes, file_name: str, openai_client=None) -> str:
    """Process a document and extract text based on file type."""
    ext = os.path.splitext(file_name)[1].lower() if file_name else ""

    if ext == ".pdf":
        text = await extract_text_from_pdf(file_bytes)
        if not text.strip() and openai_client:
            # PDF might be scanned — try OCR via Vision on first page
            logger.info("PDF text empty, trying Vision API")
        return text

    elif ext in (".docx", ".doc"):
        return await extract_text_from_docx(file_bytes)

    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        # Try tesseract first, then OpenAI Vision
        text = await extract_text_from_image(file_bytes)
        if not text.strip() and openai_client:
            text = await extract_text_from_image_openai(file_bytes, openai_client)
        return text

    else:
        # Try as image
        if openai_client:
            return await extract_text_from_image_openai(file_bytes, openai_client)
        return ""
