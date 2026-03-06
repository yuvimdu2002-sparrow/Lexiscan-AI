"""
LexiScan Auto REST API — Production Microservice
Endpoint: POST /extract
Input:  PDF file or raw text
Output: Structured JSON with extracted entities
"""

from flask import Flask, request, jsonify
import os
import re
import time
import json

app = Flask(__name__)

# ── Entity extraction engine ─────────────────
class LexiScanExtractor:
    PATTERNS = {
        "DATE": [
            r'\b(?:January|February|March|April|May|June|July|August|September|'
            r'October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        ],
        "AMOUNT": [
            r'\$[\d,]+(?:\.\d{2})?',
            r'USD\s*[\d,]+(?:\.\d{2})?',
        ],
        "PARTY": [
            r'[A-Z][A-Za-z\s]+(?:LLC|Inc\.|Corp\.|Ltd\.|LLP|Associates|'
            r'Partners|Holdings|Capital|Ventures|Services|Solutions|Group)\.?',
        ],
        "TERMINATION_CLAUSE": [
            r'(?:Either party|The Agreement) (?:may|shall) terminat\w*[^.]*\.',
            r'[^.]*(?:written notice|days notice)[^.]*\.',
        ]
    }

    def extract(self, text):
        entities = {}
        for label, patterns in self.PATTERNS.items():
            matches = []
            for p in patterns:
                for m in re.finditer(p, text, re.IGNORECASE):
                    matches.append({"text": m.group(), "start": m.start(), "end": m.end()})
            if matches:
                entities[label] = matches
        return entities


extractor = LexiScanExtractor()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "LexiScan Auto NER API",
                    "version": "1.0.0"})


@app.route("/extract", methods=["POST"])
def extract_entities():
    """
    Main endpoint. Accepts:
      - application/json: {"text": "...contract text..."}
      - multipart/form-data: file=<pdf or txt>
    Returns structured JSON with extracted entities.
    """
    t0 = time.time()

    # Handle input
    if request.is_json:
        data = request.get_json()
        text = data.get("text", "")
        doc_id = data.get("document_id", "api_doc")
    elif "file" in request.files:
        file = request.files["file"]
        try:
            text = file.read().decode("utf-8", errors="ignore")
        except Exception as e:
            return jsonify({"error": f"Could not read file: {str(e)}"}), 400
        doc_id = file.filename
    else:
        return jsonify({"error": "Provide JSON body with 'text' or upload a file"}), 400

    if not text.strip():
        return jsonify({"error": "Empty document text"}), 400

    # Extract entities
    entities = extractor.extract(text)
    proc_time = round((time.time() - t0) * 1000, 2)

    response = {
        "document_id": doc_id,
        "processing_time_ms": proc_time,
        "word_count": len(text.split()),
        "entities": entities,
        "structured_output": {
            "parties": [e["text"] for e in entities.get("PARTY", [])],
            "dates": [e["text"] for e in entities.get("DATE", [])],
            "amounts": [e["text"] for e in entities.get("AMOUNT", [])],
            "termination_clauses": [e["text"] for e in entities.get("TERMINATION_CLAUSE", [])]
        }
    }
    return jsonify(response)


@app.route("/batch", methods=["POST"])
def batch_extract():
    """Process multiple documents at once."""
    data = request.get_json()
    documents = data.get("documents", [])
    results = []
    for doc in documents[:20]:  # limit to 20
        entities = extractor.extract(doc.get("text", ""))
        results.append({"id": doc.get("id"), "entities": entities})
    return jsonify({"results": results, "count": len(results)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
