"""
Dataset Generator for LexiScan Auto - Legal Contract NER
Generates synthetic legal contracts with annotated entities:
  - DATE, PARTY, AMOUNT, TERMINATION_CLAUSE
"""

import random
import json
import os
import re
from datetime import date, timedelta

random.seed(42)

os.makedirs("data/raw_pdfs", exist_ok=True)
os.makedirs("data/annotated", exist_ok=True)

# ── Vocabulary pools ──────────────────────────────────────────
PARTY_NAMES = [
    "Apex Capital Partners LLC", "Nexgen Solutions Inc.",
    "BlueStar Financial Group", "Meridian Law Associates",
    "Global Trade Ventures Ltd.", "Pinnacle Asset Management",
    "Ironclad Technologies Corp.", "Redwood Equity Partners",
    "Solaris Consulting Group", "Vantage Point Holdings LLC",
    "Harbor Bridge Investments", "Crestview Legal Services",
    "Northern Star Enterprises", "SilverBay Advisors Inc.",
    "Pacific Rim Capital Group", "Quantum Legal Solutions",
    "Titan Financial Services", "Cascade Partners LLC",
    "EagleRock Ventures Inc.", "Summit Bridge Capital",
]

DOLLAR_AMOUNTS = [
    "$50,000", "$125,000", "$250,000", "$500,000", "$1,000,000",
    "$2,500,000", "$750,000", "$3,000,000", "$175,000", "$425,000",
    "$1,500,000", "$850,000", "$4,200,000", "$300,000", "$675,000",
]

TERMINATION_PHRASES = [
    "Either party may terminate this Agreement upon thirty (30) days written notice.",
    "This Agreement may be terminated immediately upon material breach by either party.",
    "Either party may terminate without cause upon sixty (60) days written notice to the other party.",
    "This Agreement shall terminate automatically upon completion of the services described herein.",
    "Either party may terminate this contract with ninety (90) days prior written notice.",
    "This Agreement may be terminated by mutual written consent of both parties.",
    "The Agreement terminates upon the occurrence of an Event of Default as defined in Section 12.",
    "Either party reserves the right to terminate this Agreement for convenience upon forty-five (45) days notice.",
]

GOVERNING_LAW = [
    "State of New York", "State of California", "State of Delaware",
    "State of Texas", "State of Florida", "Commonwealth of Massachusetts",
]

CONTRACT_TYPES = [
    "Service Agreement", "Non-Disclosure Agreement", "Consulting Agreement",
    "Master Services Agreement", "Partnership Agreement", "License Agreement",
    "Software Development Agreement", "Retainer Agreement",
]

SERVICES = [
    "legal advisory and document review services",
    "financial consulting and portfolio management",
    "software development and technology integration services",
    "strategic business consulting and market analysis",
    "compliance review and regulatory advisory services",
    "mergers and acquisitions due diligence services",
    "intellectual property management and licensing",
    "data analytics and business intelligence services",
]


def random_date(start_year=2020, end_year=2025):
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def format_date(d):
    formats = [
        d.strftime("%B %d, %Y"),       # January 15, 2023
        d.strftime("%d %B %Y"),         # 15 January 2023
        d.strftime("%Y-%m-%d"),         # 2023-01-15
        d.strftime("%m/%d/%Y"),         # 01/15/2023
    ]
    return random.choice(formats)


def generate_contract(idx):
    """Generate a single synthetic legal contract with entity annotations."""
    party1 = random.choice(PARTY_NAMES)
    party2 = random.choice([p for p in PARTY_NAMES if p != party1])
    exec_date = random_date()
    effective_date = exec_date + timedelta(days=random.randint(1, 15))
    end_date = effective_date + timedelta(days=random.randint(180, 730))
    amount1 = random.choice(DOLLAR_AMOUNTS)
    amount2 = random.choice(DOLLAR_AMOUNTS)
    retainer = random.choice(DOLLAR_AMOUNTS)
    term_clause = random.choice(TERMINATION_PHRASES)
    contract_type = random.choice(CONTRACT_TYPES)
    service = random.choice(SERVICES)
    gov_law = random.choice(GOVERNING_LAW)

    exec_date_str = format_date(exec_date)
    eff_date_str = format_date(effective_date)
    end_date_str = format_date(end_date)

    text = f"""{contract_type.upper()}

This {contract_type} (the "Agreement") is entered into as of {exec_date_str} (the "Execution Date"), 
by and between {party1} ("Party A") and {party2} ("Party B"), collectively referred to as the "Parties."

RECITALS

WHEREAS, Party A desires to engage Party B to provide {service};
WHEREAS, Party B is qualified and willing to provide such services under the terms set forth herein;

NOW, THEREFORE, in consideration of the mutual covenants and agreements set forth herein, 
the Parties agree as follows:

1. EFFECTIVE DATE AND TERM

This Agreement shall become effective on {eff_date_str} (the "Effective Date") and shall 
continue in full force until {end_date_str}, unless earlier terminated in accordance with 
the provisions of this Agreement.

2. SERVICES

Party B shall provide {service} as may be requested by Party A from time to time, 
subject to the terms and conditions of this Agreement.

3. COMPENSATION

3.1 In consideration for the services rendered, Party A agrees to pay Party B the sum 
of {amount1} upon execution of this Agreement.

3.2 Additional compensation of {amount2} shall be payable upon satisfactory completion 
of each project milestone as mutually agreed upon by the Parties in writing.

3.3 Party A shall also pay a monthly retainer fee of {retainer} for ongoing advisory 
services throughout the term of this Agreement.

4. CONFIDENTIALITY

Each Party acknowledges that it may receive confidential and proprietary information 
of the other Party. Each Party agrees to maintain the confidentiality of all such 
information and not to disclose it to any third party without prior written consent.

5. INTELLECTUAL PROPERTY

All work product, inventions, and deliverables created by Party B in connection with 
this Agreement shall be considered work-for-hire and shall be the exclusive property 
of Party A upon full payment of all fees.

6. TERMINATION

{term_clause}

Upon termination, Party B shall promptly deliver all work product and return or 
destroy all confidential information of Party A.

7. LIMITATION OF LIABILITY

IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, 
EXEMPLARY, OR CONSEQUENTIAL DAMAGES, HOWEVER CAUSED, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGES.

8. GOVERNING LAW

This Agreement shall be governed by and construed in accordance with the laws of 
the {gov_law}, without regard to its conflict of law provisions.

9. ENTIRE AGREEMENT

This Agreement constitutes the entire agreement between the Parties with respect to 
the subject matter hereof and supersedes all prior agreements and understandings, 
both written and oral, between the Parties.

IN WITNESS WHEREOF, the Parties have executed this Agreement as of the date 
first written above.

{party1.upper()}                    {party2.upper()}

By: _______________________         By: _______________________
Name: ____________________          Name: ____________________
Title: ____________________          Title: ____________________
Date: {exec_date_str}               Date: {exec_date_str}
"""

    # Build entity annotations
    entities = []

    def find_and_tag(search_text, label, entity_value):
        start = 0
        while True:
            pos = text.find(entity_value, start)
            if pos == -1:
                break
            entities.append({
                "start": pos,
                "end": pos + len(entity_value),
                "label": label,
                "text": entity_value
            })
            start = pos + 1

    find_and_tag(text, "PARTY", party1)
    find_and_tag(text, "PARTY", party2)
    find_and_tag(text, "DATE", exec_date_str)
    find_and_tag(text, "DATE", eff_date_str)
    find_and_tag(text, "DATE", end_date_str)
    find_and_tag(text, "AMOUNT", amount1)
    find_and_tag(text, "AMOUNT", amount2)
    find_and_tag(text, "AMOUNT", retainer)
    find_and_tag(text, "TERMINATION_CLAUSE", term_clause)

    # Remove duplicate spans
    seen = set()
    unique_entities = []
    for e in sorted(entities, key=lambda x: x["start"]):
        key = (e["start"], e["end"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(e)

    return {
        "id": f"contract_{idx:04d}",
        "text": text,
        "entities": unique_entities,
        "metadata": {
            "contract_type": contract_type,
            "party1": party1,
            "party2": party2,
            "governing_law": gov_law
        }
    }


def generate_dataset(n=300):
    print(f"Generating {n} annotated legal contracts...")
    contracts = []
    entity_counts = {"PARTY": 0, "DATE": 0, "AMOUNT": 0, "TERMINATION_CLAUSE": 0}

    for i in range(n):
        c = generate_contract(i)
        contracts.append(c)
        for e in c["entities"]:
            entity_counts[e["label"]] = entity_counts.get(e["label"], 0) + 1

        # Save individual text files
        with open(f"data/raw_pdfs/contract_{i:04d}.txt", "w") as f:
            f.write(c["text"])

        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{n} contracts")

    # Save full annotated dataset
    with open("data/annotated/contracts_annotated.json", "w") as f:
        json.dump(contracts, f, indent=2)

    # Train/val/test split
    random.shuffle(contracts)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits = {
        "train": contracts[:n_train],
        "val": contracts[n_train:n_train + n_val],
        "test": contracts[n_train + n_val:]
    }

    for split_name, data in splits.items():
        path = f"data/{split_name}/contracts_{split_name}.json"
        os.makedirs(f"data/{split_name}", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # Dataset info
    info = {
        "name": "LexiScan Auto - Legal Contract NER Dataset",
        "total_contracts": n,
        "splits": {k: len(v) for k, v in splits.items()},
        "entity_labels": ["PARTY", "DATE", "AMOUNT", "TERMINATION_CLAUSE"],
        "total_entity_mentions": entity_counts,
    }
    with open("data/dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✅ Dataset generated!")
    print(f"   Total contracts : {n}")
    for k, v in splits.items():
        print(f"   {k:5s}          : {len(v)}")
    print(f"\n   Entity mentions:")
    for label, count in entity_counts.items():
        print(f"     {label:20s}: {count}")

    return contracts, info


if __name__ == "__main__":
    generate_dataset(300)
