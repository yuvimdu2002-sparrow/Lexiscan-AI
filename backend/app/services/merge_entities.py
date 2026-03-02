from app.services.chunking_service import chunk_text
from app.services.ner_service import run_ner
from app.services.merge_entities import merge_entities

@router.post("/extract")
async def extract_contract(file: UploadFile = File(...)):

    file_bytes = await file.read()

    if is_digital_pdf(file_bytes):
        raw_text = file_bytes.decode(errors="ignore")
    else:
        raw_text = extract_text_with_ocr(file_bytes)

    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)

    all_predictions = []

    for chunk in chunks:
        preds = run_ner(chunk)
        all_predictions.extend(preds)

    merged = merge_entities(all_predictions)

    return {
        "entities": merged
    }