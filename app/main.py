from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import shutil
import os
import json

from app.ocr import extract_text
from app.preprocess import clean_text
from app.ner.spacy_model import extract_entities
from app.postprocess import postprocess

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------- UI ----------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LexiScan AI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                margin: 0;
                font-family: Arial;
                background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
                color: white;
                text-align: center;
            }
            h1 {
                font-size: 38px;
                color: #00f5d4;
                margin-top: 30px;
            }
            input {
                padding: 10px;
                margin: 10px;
                border-radius: 10px;
            }
            button {
                padding: 12px 20px;
                border: none;
                background: linear-gradient(90deg, #00f5d4, #00bbf9);
                border-radius: 10px;
                font-weight: bold;
                cursor: pointer;
                margin: 5px;
            }
            .result {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                margin-top: 20px;
            }
            .card {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                margin: 10px;
                border-radius: 15px;
                width: 300px;
            }
            .hidden {
                display: none;
            }
            p {
                white-space: pre-line;
                text-align: center;
            }
        </style>
    </head>
    <body>
    <h1>⚖️ LexiScan AI</h1>
    <p>Upload Legal Document (PDF/Image)</p>
    <input type="file" id="fileInput">
    <br>
    <button onclick="uploadFile()">🔍 Extract</button>
    <button onclick="downloadJSON()">⬇️ Download JSON</button>
    <div id="loading" class="hidden">⏳ Processing...</div>
    <div id="result" class="result hidden">
        <div class="card">
            <h3>📅 Dates</h3>
            <p id="date"></p>
            <h3>💰 Amount</h3>
            <p id="amount"></p>
        </div>
        <div class="card">
            <h3>👥 Parties</h3>
            <p id="party"></p>
            <h3>📜 Clauses</h3>
            <p id="clause"></p>
        </div>
    </div>
    <script>
    async function uploadFile() {
        const file = document.getElementById("fileInput").files[0];
        if (!file) {
            alert("Please upload a file");
            return;
        }
        document.getElementById("loading").classList.remove("hidden");
        const formData = new FormData();
        formData.append("file", file);
        try {
            const res = await fetch("/extract", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            document.getElementById("loading").classList.add("hidden");
            document.getElementById("result").classList.remove("hidden");
            function formatList(items) {
                if (!items || items.length === 0) return "No data";
                return items.map(i => "• " + i).join("\\n");
            }
            // ✅ CORRECT KEYS
            document.getElementById("date").innerText = formatList(data.dates);
            document.getElementById("amount").innerText = formatList(data.amount);
            document.getElementById("party").innerText = formatList(data.parties);
            document.getElementById("clause").innerText = formatList(data.clauses);
        } catch (err) {
            document.getElementById("loading").innerText = "❌ Error";
            console.error(err);
        }
    }
    function downloadJSON() {
        window.open("/download-json", "_blank");
    }
    </script>
    </body>
    </html>
    """


# ---------- API ----------
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # pipeline
    raw_text = extract_text(file_path)
    text = clean_text(raw_text)

    entities = extract_entities(text)
    clean_data = postprocess(entities, text)

    # ✅ fallback (important for parties)
    if not clean_data["parties"]:
        import re
        fallback = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}", text)
        clean_data["parties"] = list(set(fallback))[:5]

    # ✅ fallback clauses
    if not clean_data["clauses"]:
        clean_data["clauses"] = ["No clauses detected"]

    # save json
    output_path = os.path.join(UPLOAD_DIR, "result.json")
    with open(output_path, "w") as f:
        json.dump(clean_data, f, indent=4)

    return clean_data


# ---------- DOWNLOAD ----------
@app.get("/download-json")
def download_json():
    file_path = os.path.join(UPLOAD_DIR, "result.json")

    if not os.path.exists(file_path):
        return {"error": "No file available. Please extract first."}

    return FileResponse(
        path=file_path,
        filename="lexiscan_result.json",
        media_type="application/json"
    )