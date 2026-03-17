"""
generate_dataset.py
────────────────────
Generates a realistic legal contract NER dataset for LexiScan Auto.

Produces:
  - 20 PDF contracts (contracts/contract_001.pdf … contract_020.pdf)
  - annotations.json  — ground truth entity spans for every contract
  - train.json        — SpaCy training format
  - train_bert.json   — BERT BIO-tagged format
  - dataset_summary.pdf — human-readable summary report

Entity types: DATE | PARTY | AMOUNT | TERMINATION_CLAUSE

Usage:
  python generate_dataset.py
"""

import json
import random
import re
import os
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ── Output dirs ───────────────────────────────────────────────────────────────
OUT_DIR       = Path("output")
CONTRACT_DIR  = OUT_DIR / "contracts"
CONTRACT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ── Data pools ────────────────────────────────────────────────────────────────

LAW_FIRMS = [
    "Morrison & Foerster LLP", "Skadden Arps Slate Meagher & Flom LLP",
    "Latham & Watkins LLP", "Sullivan & Cromwell LLP", "Davis Polk & Wardwell LLP",
]

COMPANIES = [
    "Apex Capital Holdings LLC", "BlueStar Financial Group Inc.",
    "Crestwood Advisory Partners Ltd.", "Dynamo Ventures Corp.",
    "EagleRock Asset Management LLC", "Frontier Investment Partners Inc.",
    "GlobalBridge Consulting Group Ltd.", "Harborview Private Equity LLC",
    "Ironclad Technologies Corporation", "JetStream Financial Services Ltd.",
    "Keystone Capital Management Inc.", "Lakefront Holdings Group LLC",
    "Meridian Investment Advisors Corp.", "NorthStar Legal Enterprises Inc.",
    "OceanWave Partners LLC", "PinnacleCrest Capital Ltd.",
    "Quantum Analytics Group Inc.", "Riverside Law Associates LLC",
    "SilverOak Financial Consultants Ltd.", "Titan Corporate Services Corp.",
]

CONTRACT_TYPES = [
    "SERVICES AGREEMENT", "CONSULTING AGREEMENT", "NON-DISCLOSURE AGREEMENT",
    "MASTER SERVICES AGREEMENT", "PROFESSIONAL SERVICES AGREEMENT",
    "SOFTWARE LICENSE AGREEMENT", "FINANCIAL ADVISORY AGREEMENT",
    "RETAINER AGREEMENT", "ENGAGEMENT LETTER", "PARTNERSHIP AGREEMENT",
]

GOVERNING_LAWS = [
    "the State of New York", "the State of Delaware", "the State of California",
    "the State of Texas", "England and Wales",
]

TERMINATION_PHRASES = [
    "Either party may terminate this Agreement upon thirty (30) days prior written notice to the other party.",
    "Client may cancel this Agreement immediately upon written notice if Provider materially breaches any obligation herein.",
    "This Agreement shall automatically expire upon the Expiration Date unless renewed in writing by both parties.",
    "Provider may terminate with immediate effect in the event of Client's default, insolvency, or failure to pay.",
    "Either party may dissolve this Agreement without cause upon sixty (60) days written notice.",
    "Upon a material breach, the non-breaching party may wind down and terminate this Agreement within fifteen (15) days of written notice.",
    "This Agreement may be cancelled by either party upon ninety (90) days advance written notice.",
    "In the event of a change of control of either party, the other party may terminate this Agreement within thirty (30) days.",
]

def rand_date(start_year=2022, end_year=2025):
    start = date(start_year, 1, 1)
    end   = date(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def fmt_date(d: date) -> str:
    return d.strftime("%Y-%m-%d")

def fmt_date_natural(d: date) -> str:
    return d.strftime("%B %d, %Y")

def rand_amount(low=5000, high=500000, step=500):
    return random.randrange(low, high, step)

def fmt_amount(n: int) -> str:
    return f"${n:,.2f}"

# ── Contract text builder ─────────────────────────────────────────────────────

def build_contract(idx: int) -> dict:
    """Return a dict with all fields + the annotated entity list."""
    client   = random.choice(COMPANIES)
    provider = random.choice([c for c in COMPANIES if c != client])
    law_firm = random.choice(LAW_FIRMS)
    ctype    = random.choice(CONTRACT_TYPES)
    gov_law  = random.choice(GOVERNING_LAWS)

    eff_date    = rand_date(2022, 2024)
    exp_date    = eff_date + timedelta(days=random.choice([365, 730, 365*3]))
    sign_date   = eff_date - timedelta(days=random.randint(1, 14))

    monthly_fee = rand_amount(5000, 50000, 500)
    onboard_fee = rand_amount(1000, 10000, 250)
    late_fee    = rand_amount(500, 3000, 100)
    total_val   = monthly_fee * 12

    term_clause = random.choice(TERMINATION_PHRASES)

    # ── Build full contract text ──────────────────────────────────────────
    sections = []

    intro = (
        f"This {ctype} (\"Agreement\") is entered into as of {fmt_date(eff_date)} "
        f"(\"Effective Date\") by and between {client} (\"Client\"), a limited liability "
        f"company duly organised under applicable law, and {provider} (\"Provider\"), "
        f"a corporation duly organised under applicable law. This Agreement is prepared "
        f"with the assistance of {law_firm}."
    )
    sections.append(("RECITALS", intro))

    term_section = (
        f"The term of this Agreement commences on {fmt_date(eff_date)} and shall "
        f"continue in full force and effect until {fmt_date(exp_date)} (\"Expiration Date\"), "
        f"unless sooner terminated in accordance with the provisions hereof."
    )
    sections.append(("1. TERM", term_section))

    fee_section = (
        f"In consideration for the services rendered hereunder, Client shall pay Provider "
        f"a monthly retainer fee of {fmt_amount(monthly_fee)} USD, due on the first business "
        f"day of each calendar month. A one-time onboarding fee of {fmt_amount(onboard_fee)} "
        f"is payable upon execution of this Agreement. The total estimated contract value "
        f"is {fmt_amount(total_val)} USD for the initial term. Invoices not paid within "
        f"thirty (30) days shall accrue a late payment fee of {fmt_amount(late_fee)} per month."
    )
    sections.append(("2. FEES AND PAYMENT", fee_section))

    services_section = (
        f"Provider agrees to deliver professional services as mutually agreed upon "
        f"in writing from time to time by the parties. Provider shall assign qualified "
        f"personnel with the relevant expertise to perform all services. Any additional "
        f"services outside the scope hereof shall be separately scoped and priced."
    )
    sections.append(("3. SCOPE OF SERVICES", services_section))

    term_clause_section = (
        f"4.1 Termination for Convenience. {term_clause}\n\n"
        f"4.2 Termination for Cause. Either party may terminate this Agreement immediately "
        f"upon written notice if the other party materially breaches this Agreement and "
        f"fails to cure such breach within ten (10) days after receiving written notice "
        f"thereof. Upon expiration or termination, all outstanding payment obligations "
        f"shall become immediately due and payable."
    )
    sections.append(("4. TERMINATION", term_clause_section))

    conf_section = (
        f"Each party agrees to hold all Confidential Information of the other party in "
        f"strict confidence and not to disclose such information to any third party without "
        f"prior written consent. This obligation of confidentiality shall survive termination "
        f"or expiration of this Agreement for a period of five (5) years."
    )
    sections.append(("5. CONFIDENTIALITY", conf_section))

    gov_section = (
        f"This Agreement shall be governed by and construed in accordance with the laws of "
        f"{gov_law}, without regard to its conflict of law provisions. Any dispute arising "
        f"hereunder shall be submitted to binding arbitration in accordance with the rules "
        f"of the American Arbitration Association."
    )
    sections.append(("6. GOVERNING LAW", gov_section))

    sig_section = (
        f"IN WITNESS WHEREOF, the parties have executed this Agreement as of "
        f"{fmt_date_natural(sign_date)}."
    )
    sections.append(("SIGNATURE", sig_section))

    # ── Build full plain text for annotation ─────────────────────────────
    full_text = f"{ctype}\n\n"
    for title, body in sections:
        full_text += f"{title}\n\n{body}\n\n"

    # ── Collect entities with char offsets ────────────────────────────────
    entities = []

    def find_and_add(needle, label):
        start = full_text.find(needle)
        if start != -1:
            entities.append({
                "text": needle, "label": label,
                "start": start, "end": start + len(needle)
            })

    # Dates
    find_and_add(fmt_date(eff_date),          "DATE")
    find_and_add(fmt_date(exp_date),          "DATE")
    find_and_add(fmt_date_natural(sign_date), "DATE")

    # Parties
    find_and_add(client,   "PARTY")
    find_and_add(provider, "PARTY")

    # Amounts
    find_and_add(fmt_amount(monthly_fee), "AMOUNT")
    find_and_add(fmt_amount(onboard_fee), "AMOUNT")
    find_and_add(fmt_amount(late_fee),    "AMOUNT")
    find_and_add(fmt_amount(total_val),   "AMOUNT")

    # Termination clause sentence
    find_and_add(term_clause, "TERMINATION_CLAUSE")

    entities.sort(key=lambda e: e["start"])

    return {
        "id":        idx,
        "filename":  f"contract_{idx:03d}.pdf",
        "text":      full_text,
        "sections":  sections,
        "metadata": {
            "contract_type": ctype,
            "client":        client,
            "provider":      provider,
            "law_firm":      law_firm,
            "eff_date":      fmt_date(eff_date),
            "exp_date":      fmt_date(exp_date),
            "sign_date":     fmt_date_natural(sign_date),
            "monthly_fee":   fmt_amount(monthly_fee),
            "onboard_fee":   fmt_amount(onboard_fee),
            "late_fee":      fmt_amount(late_fee),
            "total_value":   fmt_amount(total_val),
            "governing_law": gov_law,
        },
        "entities": entities,
    }

# ── PDF writer ────────────────────────────────────────────────────────────────

def write_contract_pdf(contract: dict):
    path = str(CONTRACT_DIR / contract["filename"])
    doc  = SimpleDocTemplate(
        path, pagesize=letter,
        leftMargin=1*inch, rightMargin=1*inch,
        topMargin=1*inch,  bottomMargin=1*inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ContractTitle", parent=styles["Title"],
        fontSize=16, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"),
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"],
        fontSize=11, spaceBefore=14, spaceAfter=4,
        textColor=colors.HexColor("#2E4057"), bold=True,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=15, spaceAfter=8, alignment=TA_JUSTIFY,
    )
    meta_style = ParagraphStyle(
        "Meta", parent=styles["Normal"],
        fontSize=9, textColor=colors.grey,
    )

    story = []
    meta = contract["metadata"]

    # Header table
    header_data = [
        ["Contract Type:", meta["contract_type"]],
        ["Effective Date:", meta["eff_date"]],
        ["Expiration Date:", meta["exp_date"]],
        ["Client:", meta["client"]],
        ["Provider:", meta["provider"]],
    ]
    header_table = Table(header_data, colWidths=[2*inch, 4.5*inch])
    header_table.setStyle(TableStyle([
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (0, 0), (0, -1), colors.HexColor("#2E4057")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#f0f4f8"), colors.white]),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ]))

    story.append(Paragraph(meta["contract_type"], title_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(header_table)
    story.append(Spacer(1, 0.15*inch))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2E4057")))
    story.append(Spacer(1, 0.1*inch))

    for section_title, section_body in contract["sections"]:
        if section_title == "SIGNATURE":
            story.append(Spacer(1, 0.2*inch))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(section_body, body_style))
            story.append(Spacer(1, 0.4*inch))

            sig_table = Table(
                [["___________________________", "___________________________"],
                 [f"Authorised Signatory",       f"Authorised Signatory"],
                 [meta["client"],                meta["provider"]]],
                colWidths=[3*inch, 3*inch]
            )
            sig_table.setStyle(TableStyle([
                ("FONTSIZE",  (0,0), (-1,-1), 9),
                ("FONTNAME",  (0,1), (-1,-1), "Helvetica"),
                ("FONTNAME",  (0,2), (-1,-1), "Helvetica-Bold"),
                ("ALIGN",     (0,0), (-1,-1), "CENTER"),
                ("TOPPADDING",(0,0), (-1,-1), 4),
            ]))
            story.append(sig_table)
        else:
            story.append(Paragraph(section_title, heading_style))
            # Handle sub-sections separated by \n\n
            for para in section_body.split("\n\n"):
                if para.strip():
                    story.append(Paragraph(para.strip(), body_style))

    # Footer note
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        f"Prepared with the assistance of {meta['law_firm']}. "
        f"Governing law: {meta['governing_law']}. CONFIDENTIAL.",
        meta_style
    ))

    doc.build(story)
    return path

# ── Training format converters ────────────────────────────────────────────────

def to_spacy_format(contracts):
    out = []
    for c in contracts:
        out.append({
            "text": c["text"],
            "entities": [
                {"start": e["start"], "end": e["end"], "label": e["label"]}
                for e in c["entities"]
            ]
        })
    return out


def to_bert_format(contracts):
    out = []
    for c in contracts:
        text = c["text"]
        char_labels = ["O"] * len(text)
        for e in c["entities"]:
            for i in range(e["start"], min(e["end"], len(text))):
                char_labels[i] = f"B-{e['label']}" if i == e["start"] else f"I-{e['label']}"

        tokens, token_labels = [], []
        remaining = text
        offset = 0
        for word in re.split(r"(\s+)", text):
            if not word:
                continue
            if re.match(r"\s+", word):
                offset += len(word)
                continue
            tokens.append(word)
            token_labels.append(char_labels[offset] if offset < len(char_labels) else "O")
            offset += len(word)

        out.append({"tokens": tokens, "labels": token_labels})
    return out

# ── Summary PDF ───────────────────────────────────────────────────────────────

def write_summary_pdf(contracts, path):
    doc = SimpleDocTemplate(
        path, pagesize=letter,
        leftMargin=0.85*inch, rightMargin=0.85*inch,
        topMargin=0.9*inch,   bottomMargin=0.9*inch,
    )
    styles = getSampleStyleSheet()

    title_s  = ParagraphStyle("T", parent=styles["Title"],   fontSize=18, spaceAfter=4,  textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER)
    sub_s    = ParagraphStyle("S", parent=styles["Normal"],  fontSize=11, spaceAfter=16, textColor=colors.HexColor("#4A90D9"), alignment=TA_CENTER)
    h1_s     = ParagraphStyle("H1",parent=styles["Heading1"],fontSize=13, spaceBefore=18,spaceAfter=6, textColor=colors.HexColor("#2E4057"))
    body_s   = ParagraphStyle("B", parent=styles["Normal"],  fontSize=10, leading=14, spaceAfter=6)
    small_s  = ParagraphStyle("Sm",parent=styles["Normal"],  fontSize=9,  leading=12, spaceAfter=4, textColor=colors.grey)
    code_s   = ParagraphStyle("C", parent=styles["Code"],    fontSize=8,  leading=11, fontName="Courier", backColor=colors.HexColor("#f5f5f5"))

    story = []
    story.append(Paragraph("LexiScan Auto", title_s))
    story.append(Paragraph("NER Training Dataset — Summary Report", sub_s))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2E4057")))
    story.append(Spacer(1, 0.15*inch))

    # Stats
    total_entities = sum(len(c["entities"]) for c in contracts)
    label_counts = {}
    for c in contracts:
        for e in c["entities"]:
            label_counts[e["label"]] = label_counts.get(e["label"], 0) + 1

    story.append(Paragraph("Dataset Overview", h1_s))
    stats = [
        ["Metric", "Value"],
        ["Total contracts generated", str(len(contracts))],
        ["Total annotated entities", str(total_entities)],
        ["Avg entities per contract", f"{total_entities / len(contracts):.1f}"],
        ["Entity types", "DATE, PARTY, AMOUNT, TERMINATION_CLAUSE"],
        ["Output formats", "PDF, annotations.json, train.json (SpaCy), train_bert.json (BERT)"],
    ]
    for label, count in sorted(label_counts.items()):
        stats.append([f"  {label} count", str(count)])

    tbl = Table(stats, colWidths=[3.2*inch, 3.5*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#2E4057")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.HexColor("#f0f4f8"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.2*inch))

    # File listing
    story.append(Paragraph("Generated Files", h1_s))
    file_data = [["File", "Type", "Purpose"]]
    for c in contracts:
        file_data.append([c["filename"], "PDF", f"{c['metadata']['contract_type']} — {len(c['entities'])} entities"])
    file_data.append(["annotations.json", "JSON", "Full ground-truth with char offsets"])
    file_data.append(["train.json",        "JSON", "SpaCy NER training format"])
    file_data.append(["train_bert.json",   "JSON", "BERT BIO token format"])
    file_data.append(["dataset_summary.pdf","PDF", "This report"])

    ftbl = Table(file_data, colWidths=[2.5*inch, 0.8*inch, 3.4*inch])
    ftbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#2E4057")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0f4f8"),colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
    ]))
    story.append(ftbl)
    story.append(Spacer(1, 0.2*inch))

    # Sample annotation
    story.append(Paragraph("Sample Ground-Truth Annotation (contract_001)", h1_s))
    sample = contracts[0]
    sample_json = json.dumps({
        "filename": sample["filename"],
        "entities": sample["entities"][:6]
    }, indent=2)
    story.append(Paragraph(sample_json.replace("\n","<br/>").replace(" ","&nbsp;"), code_s))
    story.append(Spacer(1, 0.2*inch))

    # How to use
    story.append(Paragraph("How to Use This Dataset", h1_s))
    steps = [
        ("Step 1 — Convert annotations",
         "python scripts/convert_doccano.py  (or use train.json / train_bert.json directly)"),
        ("Step 2 — Train SpaCy model",
         "python scripts/train_spacy_model.py   →  models/spacy_ner_model/"),
        ("Step 3 — Fine-tune LegalBERT",
         "python scripts/train_bert_model.py    →  models/bert_ner_model/"),
        ("Step 4 — Test on a PDF",
         "python demo.py   or   POST /extract  via the FastAPI server"),
    ]
    for title, desc in steps:
        story.append(Paragraph(f"<b>{title}</b>", body_s))
        story.append(Paragraph(desc, small_s))

    doc.build(story)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating dataset …")
    contracts = [build_contract(i + 1) for i in range(20)]

    # Write PDFs
    for c in contracts:
        p = write_contract_pdf(c)
        print(f"  [PDF] {p}  ({len(c['entities'])} entities)")

    # Write annotations.json
    annotations = [{"filename": c["filename"], "text": c["text"], "entities": c["entities"]} for c in contracts]
    with open(OUT_DIR / "annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"  [JSON] output/annotations.json")

    # Write SpaCy format
    with open(OUT_DIR / "train.json", "w") as f:
        json.dump(to_spacy_format(contracts), f, indent=2)
    print(f"  [JSON] output/train.json  (SpaCy format)")

    # Write BERT format
    with open(OUT_DIR / "train_bert.json", "w") as f:
        json.dump(to_bert_format(contracts), f, indent=2)
    print(f"  [JSON] output/train_bert.json  (BERT BIO format)")

    # Write summary PDF
    summary_path = str(OUT_DIR / "dataset_summary.pdf")
    write_summary_pdf(contracts, summary_path)
    print(f"  [PDF] {summary_path}")

    print(f"\nDone. {len(contracts)} contracts, {sum(len(c['entities']) for c in contracts)} total entities.")


if __name__ == "__main__":
    main()
