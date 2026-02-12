import os
import xml.etree.ElementTree as ET
import pandas as pd
from collections import Counter
from typing import Any

def parse_xmi(filepath: str) -> dict[str, Any] | None:
    """
    Parse a single XMI file and extract:
      - documentId
      - text
      - gold_label
      - annotator agreement stats
    """
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        # --- Extract text ---
        text = ""
        for el in root.iter():
            if el.tag.endswith("Sofa") and "sofaString" in el.attrib:
                text = el.attrib.get("sofaString", "").strip()
                break

        # --- Extract document ID ---
        document_id = None
        for el in root.iter():
            if el.tag.endswith("DocumentMetaData"):
                document_id = el.attrib.get("documentId")
                break
        if not document_id:
            document_id = os.path.basename(filepath)

        # --- Extract persuasiveness annotations ---
        pers_annotations = [
            el for el in root.iter() if el.tag.endswith("PersuasivenessAnnotationMetaData")
        ]

        gold_label = None
        annotator_labels: list[bool] = []

        for ann in pers_annotations:
            is_persuasive = ann.attrib.get("isPersuasive")
            is_gold = ann.attrib.get("isGold", "false").lower() == "true"
            annotator = ann.attrib.get("annotator")

            if is_persuasive is not None:
                label_bool = is_persuasive.lower() == "true"
                if annotator:
                    annotator_labels.append(label_bool)
                if is_gold:
                    gold_label = label_bool

        # --- Compute agreement ---
        if annotator_labels:
            c = Counter(annotator_labels)
            persuasive_votes = c.get(True, 0)
            non_persuasive_votes = c.get(False, 0)
            total = persuasive_votes + non_persuasive_votes
            agreement_ratio = max(persuasive_votes, non_persuasive_votes) / total if total > 0 else None
        else:
            persuasive_votes = non_persuasive_votes = agreement_ratio = None

        return {
            "documentId": document_id,
            "text": text,
            "gold_label": gold_label,
            "annotator_agreement": agreement_ratio,
            "persuasive_votes": persuasive_votes,
            "non_persuasive_votes": non_persuasive_votes,
            "num_annotators": len(annotator_labels),
        }

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def load_xmi_folder(folder_path: str):
    """
    Parse all .xmi files in a folder into a pandas DataFrame.
    """
    data: list[dict[str, Any]] = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".xmi"):
            fpath = os.path.join(folder_path, fname)
            record = parse_xmi(fpath)
            if record:
                data.append(record)
    return pd.DataFrame(data)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.normpath(os.path.join(BASE_DIR, "../data/download/gold.data.persuasive"))
    df = load_xmi_folder(folder)

    print(f"Parsed {len(df)} documents")
    print(df.head())

    output_file = os.path.normpath(os.path.join(BASE_DIR, "../data/sentences.csv"))
    df.to_csv(output_file, index=False)
    print(f"Saved to '{output_file}'")
