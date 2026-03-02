import re

def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9$.,:;/\-() ]", "", text)
    return text.strip()