import pdfplumber
import pytesseract

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t
            else:
                img = page.to_image().original
                text += pytesseract.image_to_string(img)
    return text
