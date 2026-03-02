from fastapi import APIRouter, UploadFile, File
from app.services.file_detector import is_digital_pdf
from app.services.ocr_service import extract_text_with_ocr
from app.services.preprocessing_service import clean_text

router = APIRouter()

@router.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):

    file_bytes = await file.read()

    if is_digital_pdf(file_bytes):
        raw_text = file_bytes.decode(errors="ignore")
    else:
        raw_text = extract_text_with_ocr(file_bytes)

    cleaned_text = clean_text(raw_text)

    return {
        "text_length": len(cleaned_text),
        "preview": cleaned_text[:500]
    }