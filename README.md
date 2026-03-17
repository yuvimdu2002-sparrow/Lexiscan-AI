🧾 Legal Contract NER System

A complete Legal Named Entity Recognition (NER) system that extracts key information from contracts such as Parties, Dates, Monetary Values, and Clauses using OCR + NLP.

---

🚀 Features

- 📄 Upload PDF or Image
- 🔍 Extract:
  - Parties (Organizations / Persons)
  - Dates
  - Monetary values
  - Clauses
- 🧠 Uses:
  - OCR (for scanned documents)
  - spaCy / BERT (for NER)
- 🧹 Post-processing for clean structured output
- 🌐 FastAPI-based backend
- 🎨 Simple UI for testing

---

🏗️ Project Structure

```text

legal-ner-system/
│
├── app/
│   ├── main.py              # FastAPI app
│   ├── ocr.py               # OCR extraction
│   ├── preprocess.py        # Text cleaning
│   ├── postprocess.py       # Entity cleaning & filtering
│   │
│   ├── ner/
│   │   ├── spacy_model.py   # spaCy NER
│   │   ├── bert_model.py    # BERT NER (optional)
│
├── uploads/                 # Uploaded files
├── requirements.txt
├── Dockerfile (optional)
└── README.md
```
---

⚙️ Installation

1. Clone repository

git clone https://github.com/yuvimdu2002-sparrow/Lexiscan-AI

2. Create virtual environment

python -m venv venv
venv\Scripts\activate   # Windows

3. Install dependencies

pip install -r requirements.txt

4. Download spaCy model

python -m spacy download en_core_web_sm

---

▶️ Run the Application

uvicorn app.main:app --reload

Open browser:

http://127.0.0.1:8000

---

📤 Input

- PDF files
- Image files (JPG, PNG)

---

📥 Output (Example)

{
  "parties": ["ABC Pvt Ltd", "John Doe"],
  "dates": ["01 January 2024"],
  "money": ["Rs. 50,000"],
  "clauses": [
    "Payment shall be made within 30 days",
    "Agreement may be terminated with notice"
  ]
}

---

🧠 Processing Pipeline

1. OCR
   
   - Extract text from PDF/Image

2. Preprocessing
   
   - Remove noise
   - Normalize text

3. NER Extraction
   
   - spaCy / BERT detects entities

4. Post-processing
   
   - Remove duplicates
   - Clean incorrect entities
   - Separate:
     - Parties vs Clauses
     - Money vs Dates


---

🧪 Testing

Use provided sample PDFs:

- Service Agreement
- Employment Agreement
- Lease Agreement

Test with:

- Clean PDFs
- Scanned documents (OCR)
- Long contracts

---

🐳 Docker (Optional)

docker build -t legal-ner .
docker run -p 8000:8000 legal-ner

---

🔮 Future Improvements

- ✅ Better clause segmentation (NLP-based)
- ✅ Custom trained legal NER model
- ✅ UI improvements (React / Streamlit)
- ✅ Multi-language support
- ✅ Export results (JSON / CSV)

---

👨‍💻 Author

Yuvaraj A

---

⭐ Support

If you like this project:

- ⭐ Star the repo
- 🍴 Fork it
- 🛠 Improve it

---

📌 Summary

This project helps automate legal document analysis by extracting structured data from unstructured contracts using OCR + NLP, making it useful for:

- Legal tech
- Document automation
- Contract analysis systems

---
