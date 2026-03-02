import pytesseract
from pdf2image import convert_from_bytes

def extract_text_with_ocr(file_bytes):
    images = convert_from_bytes(file_bytes)
    text = ""

    for img in images:
        page_text = pytesseract.image_to_string(img)
        text += page_text + "\n"

    return text