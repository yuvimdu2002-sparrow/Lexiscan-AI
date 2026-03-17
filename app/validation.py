import re
def validate(data):
    return {
        "DATE": [d for d in data.get("DATE", []) if re.match(r"\d{4}-\d{2}-\d{2}", d)],
        "AMOUNT": [a for a in data.get("AMOUNT", []) if "₹" in a or "$" in a],
        "PARTY": data.get("PARTY", []),
        "CLAUSE": data.get("CLAUSE", [])
    }
