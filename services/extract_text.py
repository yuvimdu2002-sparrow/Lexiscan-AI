from file_detector import is_digital_pdf
from ocr_service import extract_text_with_ocr
from text_cleaner import clean_text

def process_pdf(file_path):
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    if is_digital_pdf(file_bytes):
        raw_text = file_bytes.decode(errors="ignore")
    else:
        raw_text = extract_text_with_ocr(file_bytes)

    cleaned = clean_text(raw_text)

    return cleaned