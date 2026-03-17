import re


# ---------- COMMON ----------
def unique(items):
    return list(dict.fromkeys([i.strip() for i in items if i.strip()]))


# ---------- DATES (UNCHANGED) ----------
def clean_dates(dates):
    result = []
    for d in dates:
        result += re.findall(r"\b\d{4}-\d{2}-\d{2}\b", d)
    return unique(result)


# ---------- AMOUNT (UNCHANGED) ----------
def clean_amounts(amounts, text):
    result = []

    # only real money (must contain comma)
    pattern = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"

    for amt in amounts:
        result += re.findall(pattern, amt)

    result += re.findall(pattern, text)

    return unique(result)


# ---------- PARTIES (FIXED) ----------
def clean_parties(parties, text):
    result = []

    # ✅ STEP 1: Extract from FULL TEXT (fallback)
    text_matches = re.findall(
        r'\b([A-Z][A-Za-z& ]+(?:LLC|Ltd|Inc|Corporation|Company|LLP))\b',
        text
    )

    # Merge both
    all_parties = parties + text_matches

    for p in all_parties:
        p = p.strip()
        p_lower = p.lower()

        if not p:
            continue

        # ❌ remove noise
        if any(word in p_lower for word in [
            "agreement", "contract", "type", "date",
            "information", "clause", "signatory",
            "authorised", "client", "provider"
        ]):
            continue

        # ❌ remove sentence-like strings
        if len(p.split()) > 6:
            continue

        # ✅ must contain company keyword
        if not re.search(r"(llc|ltd|inc|corporation|company|llp)", p_lower):
            continue

        result.append(p)

    return unique(result)
    
# ---------- CLAUSES (FIXED) ----------
def extract_clauses(text):
    clauses = []

    text = re.sub(r'\s+', ' ', text)

    parts = re.split(r'\.\s|\n', text)

    for p in parts:
        p = p.strip()

        if len(p) < 50:
            continue

        # ❌ remove weak sentences
        if any(x in p.lower() for x in [
            "prepared with",
            "provider, a corporation",
            "this agreement is prepared"
        ]):
            continue

        if re.search(r"(llc|ltd|inc|corporation|company|llp)", p.lower()):
            continue

        clauses.append(p)

    return unique(clauses[:3])

# ---------- FINAL ----------

def postprocess(entities, text):
    return {
        "dates": clean_dates(entities.get("DATE", [])),
        "amount": clean_amounts(entities.get("MONEY", []), text),
        "parties": clean_parties(entities.get("ORG", []), text),  # <-- FIXED
        "clauses": extract_clauses(text),
    }