import pdfplumber
import io

def is_digital_pdf(file_bytes):
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return len(text.strip()) > 50
    except:
        return False