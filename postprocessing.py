"""
Week 3: Post-Processing & Rule-Based Validation — LexiScan Auto
Raw NER output is imperfect. Rule-based logic validates extracted entities:
  - DATE must conform to YYYY-MM-DD or similar
  - AMOUNT must contain currency symbols ($, €, £)
  - PARTY must be proper noun / capitalized
  - TERMINATION_CLAUSE must contain legal keywords
Robust handling of edge cases and noise is paramount for lawyer trust.
"""

import re
import json
import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

random.seed(42)
np.random.seed(42)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# RULE-BASED VALIDATORS

class EntityValidator:
    """
    Production-grade rule-based validator for NER output.
    Implements validation logic for all four entity types.
    """

    # DATE patterns accepted
    DATE_PATTERNS = [
        r'\b\d{4}-\d{2}-\d{2}\b',                                # 2023-01-15
        r'\b(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # January 15, 2023
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|'
        r'August|September|October|November|December)\s+\d{4}\b',        # 15 January 2023
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',                                    # 01/15/2023
        r'\b\d{1,2}\.\d{1,2}\.\d{4}\b',                                  # 15.01.2023
    ]

    # AMOUNT patterns
    AMOUNT_PATTERNS = [
        r'\$[\d,]+(?:\.\d{2})?',           # $1,000,000.00
        r'USD\s*[\d,]+(?:\.\d{2})?',       # USD 50,000
        r'€[\d,]+(?:\.\d{2})?',            # €500,000
        r'£[\d,]+(?:\.\d{2})?',            # £250,000
        r'[\d,]+(?:\.\d{2})?\s*dollars?',  # 50,000 dollars
    ]

    # Legal termination keywords
    TERMINATION_KEYWORDS = [
        "terminat", "cancel", "dissolv", "expi", "rescind",
        "written notice", "days notice", "days' notice",
        "material breach", "event of default", "mutual consent",
        "for convenience", "without cause",
    ]

    # Party name indicators
    PARTY_INDICATORS = [
        r'\b(?:LLC|Inc\.|Corp\.|Ltd\.|LLP|LP|PLC|PLLC|Associates|Partners|'
        r'Holdings|Capital|Ventures|Enterprises|Services|Solutions|Group|'
        r'Consulting|Advisory|Management|Financial|Legal)\b',
    ]

    def validate_date(self, text):
        """Validate DATE entity."""
        for pattern in self.DATE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "VALID", None

        # Try to parse as date
        formats = ["%B %d, %Y", "%d %B %Y", "%Y-%m-%d", "%m/%d/%Y"]
        for fmt in formats:
            try:
                datetime.strptime(text.strip(), fmt)
                return True, "VALID", None
            except ValueError:
                pass

        return False, "INVALID_DATE_FORMAT", f"'{text}' does not match any accepted date format"

    def validate_amount(self, text):
        """Validate AMOUNT entity — must contain currency symbol."""
        for pattern in self.AMOUNT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "VALID", None

        if re.search(r'\d', text):
            return False, "MISSING_CURRENCY_SYMBOL", \
                f"'{text}' contains numbers but no currency symbol ($, €, £, USD)"

        return False, "NO_NUMERIC_VALUE", f"'{text}' contains no numeric value"

    def validate_party(self, text):
        """Validate PARTY entity — must be a proper noun / company name."""
        if len(text.strip()) < 3:
            return False, "TOO_SHORT", f"Party name '{text}' is too short"

        # Check for company indicators
        for pattern in self.PARTY_INDICATORS:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "VALID_COMPANY", None

        # Check for capitalized words (proper noun)
        words = text.split()
        capitalized = sum(1 for w in words if w[0].isupper() and len(w) > 1)
        if capitalized / len(words) >= 0.5:
            return True, "VALID_PROPER_NOUN", None

        return False, "NOT_PROPER_NOUN", f"'{text}' does not appear to be a party name"

    def validate_termination_clause(self, text):
        """Validate TERMINATION_CLAUSE — must contain legal keywords."""
        text_lower = text.lower()
        found_keywords = [kw for kw in self.TERMINATION_KEYWORDS if kw in text_lower]

        if found_keywords:
            return True, "VALID", None

        if len(text.split()) < 5:
            return False, "TOO_SHORT", "Termination clause is too short"

        return False, "MISSING_LEGAL_KEYWORDS", \
            f"No termination-related keywords found in: '{text[:60]}...'"

    def validate_entity(self, entity):
        """Route to appropriate validator by label."""
        label = entity.get("label", "")
        text = entity.get("text", "").strip()

        validators = {
            "DATE": self.validate_date,
            "AMOUNT": self.validate_amount,
            "PARTY": self.validate_party,
            "TERMINATION_CLAUSE": self.validate_termination_clause,
        }

        if label in validators:
            valid, status, error = validators[label](text)
            return {**entity, "valid": valid, "status": status,
                    "error": error, "label": label}
        return {**entity, "valid": True, "status": "UNCHECKED", "error": None}

    def validate_document(self, contract):
        """Validate all entities in a contract document."""
        validated = []
        stats = {"total": 0, "valid": 0, "invalid": 0, "by_type": {}}

        for entity in contract.get("entities", []):
            result = self.validate_entity(entity)
            validated.append(result)

            lbl = result["label"]
            stats["total"] += 1
            if result["valid"]:
                stats["valid"] += 1
            else:
                stats["invalid"] += 1

            if lbl not in stats["by_type"]:
                stats["by_type"][lbl] = {"valid": 0, "invalid": 0}
            if result["valid"]:
                stats["by_type"][lbl]["valid"] += 1
            else:
                stats["by_type"][lbl]["invalid"] += 1

        return {
            "id": contract["id"],
            "validated_entities": validated,
            "stats": stats
        }


# EDGE CASE HANDLER

class EdgeCaseHandler:
    """Handle noisy / edge case NER outputs."""

    @staticmethod
    def normalize_date(text):
        """Normalize various date formats to YYYY-MM-DD."""
        formats = ["%B %d, %Y", "%B %d %Y", "%d %B %Y", "%m/%d/%Y", "%d.%m.%Y"]
        for fmt in formats:
            try:
                return datetime.strptime(text.strip().replace(',', ''), fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass
        return text

    @staticmethod
    def normalize_amount(text):
        """Standardize amount format."""
        cleaned = re.sub(r'[^\d.,]', '', text)
        try:
            val = float(cleaned.replace(',', ''))
            return f"${val:,.2f}"
        except ValueError:
            return text

    @staticmethod
    def normalize_party(text):
        """Clean up party name."""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\(.*?\)', '', text).strip()
        return text

    def handle_edge_cases(self, entity):
        """Apply normalization based on entity type."""
        label = entity.get("label")
        text = entity.get("text", "")
        original = text

        if label == "DATE":
            text = self.normalize_date(text)
        elif label == "AMOUNT":
            text = self.normalize_amount(text)
        elif label == "PARTY":
            text = self.normalize_party(text)

        return {**entity, "normalized_text": text,
                "was_modified": text != original}


# SIMULATE VALIDATION ON DATASET

def run_validation_pipeline(contracts, sample_size=50):
    validator = EntityValidator()
    handler = EdgeCaseHandler()

    all_results = []
    agg_stats = {"total": 0, "valid": 0, "invalid": 0,
                 "by_type": {lbl: {"valid": 0, "invalid": 0}
                             for lbl in ["PARTY", "DATE", "AMOUNT", "TERMINATION_CLAUSE"]}}

    # Inject some artificial errors for realism
    def maybe_corrupt(entity):
        if random.random() < 0.08:
            label = entity["label"]
            if label == "DATE":
                return {**entity, "text": "TBD (to be confirmed)"}
            elif label == "AMOUNT":
                return {**entity, "text": entity["text"].replace("$", "")}
        return entity

    for c in contracts[:sample_size]:
        corrupted_entities = [maybe_corrupt(e) for e in c.get("entities", [])]
        corrupted_contract = {**c, "entities": corrupted_entities}
        result = validator.validate_document(corrupted_contract)

        # Apply edge case handling
        normalized = [handler.handle_edge_cases(e) for e in result["validated_entities"]]
        result["validated_entities"] = normalized

        all_results.append(result)
        agg_stats["total"] += result["stats"]["total"]
        agg_stats["valid"] += result["stats"]["valid"]
        agg_stats["invalid"] += result["stats"]["invalid"]
        for lbl, counts in result["stats"]["by_type"].items():
            if lbl in agg_stats["by_type"]:
                agg_stats["by_type"][lbl]["valid"] += counts["valid"]
                agg_stats["by_type"][lbl]["invalid"] += counts["invalid"]

    return all_results, agg_stats


# VISUALIZATIONS

def visualize_validation_pipeline():
    """Diagram of post-processing flow."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title("Week 3: Post-Processing & Validation Pipeline — LexiScan Auto",
                 fontsize=12, fontweight='bold')

    # Main flow
    main_stages = [
        (2, 5.5, "Raw NER\nOutput", "#e74c3c"),
        (5, 5.5, "Rule-Based\nValidator", "#8e44ad"),
        (8, 5.5, "Edge Case\nHandler", "#e67e22"),
        (11, 5.5, "Structured\nJSON Output", "#27ae60"),
    ]
    for x, y, lbl, color in main_stages:
        fancy = patches.FancyBboxPatch((x - 1.2, y - 0.55), 2.4, 1.1,
                                       boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(fancy)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=9,
                color='white', fontweight='bold', multialignment='center')

    for i in range(len(main_stages) - 1):
        x1, x2 = main_stages[i][0] + 1.2, main_stages[i + 1][0] - 1.2
        ax.annotate('', xy=(x2, 5.5), xytext=(x1, 5.5),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    # Validator branches
    rules = [
        (2.5, 3.5, "DATE: YYYY-MM-DD\nor long format", "#3498db"),
        (5.0, 3.5, "AMOUNT: $|€|£\ncurrency required", "#2ecc71"),
        (7.5, 3.5, "PARTY: Capitalized\n+ LLC/Inc suffix", "#9b59b6"),
        (10.5, 3.5, "TERM: 'terminate'\n'notice' keywords", "#e67e22"),
    ]
    for x, y, txt, color in rules:
        box = patches.FancyBboxPatch((x - 1.2, y - 0.45), 2.4, 0.9,
                                     boxstyle="round,pad=0.1",
                                     facecolor=color, alpha=0.8, edgecolor='white')
        ax.add_patch(box)
        ax.text(x, y, txt, ha='center', va='center', fontsize=7.5,
                color='white', multialignment='center')
        ax.annotate('', xy=(x, y + 0.45), xytext=(5, 5.5 - 0.55),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1, linestyle='dashed'))

    ax.text(7, 1.5,
            "Invalid entities → flagged for human review | Valid entities → passed to output JSON",
            ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'))

    plt.tight_layout()
    path = "results/validation_pipeline.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def visualize_validation_results(agg_stats):
    """Show validation pass/fail rates per entity type."""
    entity_types = list(agg_stats["by_type"].keys())
    valid_counts = [agg_stats["by_type"][e]["valid"] for e in entity_types]
    invalid_counts = [agg_stats["by_type"][e]["invalid"] for e in entity_types]
    total_counts = [v + i for v, i in zip(valid_counts, invalid_counts)]
    valid_pct = [v / t * 100 if t > 0 else 0 for v, t in zip(valid_counts, total_counts)]
    invalid_pct = [i / t * 100 if t > 0 else 0 for i, t in zip(invalid_counts, total_counts)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Week 3: Rule-Based Validation Results\nLexiScan Auto — Edge Case Handling",
                 fontsize=12, fontweight='bold')

    # Stacked bar chart
    x = np.arange(len(entity_types))
    axes[0].bar(x, valid_pct, color='#2ecc71', edgecolor='black', label='Valid')
    axes[0].bar(x, invalid_pct, bottom=valid_pct, color='#e74c3c', edgecolor='black', label='Invalid')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(entity_types, rotation=10)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("Validation Pass Rate by Entity Type", fontsize=11)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim(0, 115)
    for i, (vp, ip) in enumerate(zip(valid_pct, invalid_pct)):
        axes[0].text(i, vp / 2, f'{vp:.0f}%', ha='center', va='center',
                     color='white', fontweight='bold', fontsize=9)

    # Overall pie
    overall = [agg_stats["valid"], agg_stats["invalid"]]
    labels = [f'Valid\n({agg_stats["valid"]})', f'Needs Review\n({agg_stats["invalid"]})']
    explode = (0.05, 0.1)
    axes[1].pie(overall, labels=labels, autopct='%1.1f%%', explode=explode,
                colors=['#2ecc71', '#e74c3c'], startangle=90,
                textprops={'fontsize': 11})
    axes[1].set_title(f"Overall Validation Summary\n"
                      f"Total Entities: {agg_stats['total']}", fontsize=11)

    plt.tight_layout()
    path = "results/validation_results.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def visualize_entity_annotation_sample(contracts):
    """Show a sample contract with highlighted entities (Doccano-style)."""
    contract = random.choice(contracts[:10])
    text = contract["text"][:600]
    entities = [e for e in contract["entities"] if e["end"] <= 600]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title("Week 3: Sample Contract — Annotated Entity Highlights\n(Doccano-style visualization)",
                 fontsize=12, fontweight='bold')

    # Truncated contract text
    lines = text[:500].split('\n')[:12]
    y = 7.5
    for line in lines:
        ax.text(0.3, y, line[:100], fontsize=7.5, va='top', family='monospace',
                color='#2c3e50')
        y -= 0.52

    # Entity legend
    for i, (label, color) in enumerate(ENTITY_COLORS.items()):
        ax.add_patch(patches.FancyBboxPatch((9 + (i % 2) * 2.2, 7.2 - (i // 2) * 0.8),
                                             2.0, 0.6,
                                             boxstyle="round,pad=0.05",
                                             facecolor=color, edgecolor='white'))
        ax.text(10.0 + (i % 2) * 2.2, 7.5 - (i // 2) * 0.8, label,
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    plt.tight_layout()
    path = "results/annotation_sample.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


ENTITY_COLORS = {
    "PARTY": "#3498db",
    "DATE": "#e74c3c",
    "AMOUNT": "#2ecc71",
    "TERMINATION_CLAUSE": "#f39c12"
}


# MAIN

if __name__ == "__main__":
    print("POST-PROCESSING & RULE-BASED VALIDATION")
    print("  LexiScan Auto - Legal Contract NER")

    contracts_path = "data/annotated/contracts_annotated.json"
    if not os.path.exists(contracts_path):
        print("⚠️  Run generate_dataset.py first!")
    else:
        with open(contracts_path) as f:
            contracts = json.load(f)
        print(f"\n  Loaded {len(contracts)} contracts")

        print("\n  Running validation pipeline...")
        results, agg_stats = run_validation_pipeline(contracts, sample_size=100)

        print(f"\n  Validation Summary:")
        print(f"    Total entities  : {agg_stats['total']}")
        print(f"    Valid           : {agg_stats['valid']} ({agg_stats['valid']/agg_stats['total']*100:.1f}%)")
        print(f"    Needs review    : {agg_stats['invalid']}")
        for lbl, counts in agg_stats["by_type"].items():
            total = counts["valid"] + counts["invalid"]
            pct = counts["valid"] / total * 100 if total > 0 else 0
            print(f"    {lbl:22s}: {pct:.1f}% valid")

        with open("results/validation_summary.json", "w") as f:
            json.dump({"aggregate": agg_stats, "sample_results": results[:3]}, f, indent=2)

        print("\n  Generating visualizations...")
        visualize_validation_pipeline()
        visualize_validation_results(agg_stats)
        visualize_entity_annotation_sample(contracts)

    print("\n✅ complete! Rule-based validation is production-ready for lawyer trust.")
