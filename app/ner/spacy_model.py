import spacy
import re

# Load model (custom or default)
try:
    nlp = spacy.load("model/spacy_model")
except:
    nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    doc = nlp(text)

    result = {
        "DATE": [],
        "AMOUNT": [],
        "PARTY": [],
        "CLAUSE": []
    }

    # ✅ spaCy extraction
    for ent in doc.ents:

        if ent.label_ == "DATE":
            result["DATE"].append(ent.text)

        elif ent.label_ == "MONEY":
            result["AMOUNT"].append(ent.text)

        elif ent.label_ in ["PERSON", "ORG"]:
            result["PARTY"].append(ent.text)

    # ✅ Regex for AMOUNT (extra support)
    amounts = re.findall(r'(₹\s?\d+(?:,\d+)*(?:\.\d+)?|\$\s?\d+(?:,\d+)*(?:\.\d+)?)', text)
    result["AMOUNT"].extend(amounts)

    # ✅ Regex for PARTY (basic legal pattern)
    parties = re.findall(r'(?:Between|By|Party)\s*:?\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)', text)
    result["PARTY"].extend(parties)

    # ✅ Regex for CLAUSE (IMPORTANT)
    clauses = re.findall(r'(Clause\s*\d+[^\.]*)', text)
    result["CLAUSE"].extend(clauses)

    # ✅ Remove duplicates
    for key in result:
        result[key] = list(set(result[key]))

    return result