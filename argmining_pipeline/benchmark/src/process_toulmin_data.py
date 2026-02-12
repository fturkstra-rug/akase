import os
import csv
from lxml import etree # type: ignore (has no stub files)
import os

# Folder containing XMI files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folder = os.path.normpath(os.path.join(BASE_DIR, "../data/download/gold.data.toulmin"))

# Component types we care about
component_types = ["Claim", "Premise", "Backing", "Rebuttal", "Refutation"]

# Relation rules (directional)
relation_rules = {
    ("Premise", "Claim"): "support",
    ("Backing", "Claim"): "support",
    ("Backing", "Premise"): "support",
    ("Rebuttal", "Claim"): "attack",
    ("Refutation", "Rebuttal"): "attack",
}

components = []
relations = []

for file_name in os.listdir(folder):
    if not file_name.endswith(".xmi"):
        continue

    path = os.path.join(folder, file_name)
    # print(f"Processing {file_name}...")

    # Parse XML with lxml
    tree = etree.parse(path)
    root = tree.getroot()

    # Collect all namespaces automatically
    namespaces = {k: v for event, (k, v) in etree.iterparse(path, events=("start-ns",))}
    # print(f"  Detected namespaces: {namespaces}")

    # Extract sofaString (main text content)
    sofa = root.find(".//cas:Sofa", namespaces=namespaces)
    text = sofa.attrib.get("sofaString", "") if sofa is not None else ""

    # Extract components
    doc_components = []
    for comp_type in component_types:
        xpath_expr = f".//types:{comp_type}"
        for elem in root.findall(xpath_expr, namespaces=namespaces):
            begin = int(elem.attrib.get("begin", 0))
            end = int(elem.attrib.get("end", 0))
            comp_text = text[begin:end].strip()
            comp_id = elem.attrib.get("{http://www.omg.org/XMI}id", "")

            if comp_text:
                doc_components.append({
                    "id": comp_id,
                    "text": comp_text,
                    "type": comp_type,
                    "begin": begin,
                    "end": end
                })
                components.append({
                    "id": comp_id,
                    "text": comp_text,
                    "component_type": comp_type
                })

    # Sort by text order (begin offset)
    doc_components.sort(key=lambda x: x["begin"])

    # Build forward-only relations (A before B)
    for i, src in enumerate(doc_components):
        for j in range(i + 1, len(doc_components)):
            tgt = doc_components[j]
            rel_type = relation_rules.get((src["type"], tgt["type"]))
            if rel_type:
                relations.append({
                    "source_text": src["text"],
                    "source_component_type": src["type"],
                    "target_text": tgt["text"],
                    "target_component_type": tgt["type"],
                    "relation_type": rel_type
                })

# Save components.csv
components_file = os.path.normpath(os.path.join(BASE_DIR, "../data/components.csv"))
with open(components_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "text", "component_type"])
    writer.writeheader()
    writer.writerows(components)

# Save relations.csv
relations_file = os.path.normpath(os.path.join(BASE_DIR, "../data/relations.csv"))
with open(relations_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "source_text", "source_component_type",
        "target_text", "target_component_type",
        "relation_type"
    ])
    writer.writeheader()
    writer.writerows(relations)

print("Generated components.csv and relations.csv.")
