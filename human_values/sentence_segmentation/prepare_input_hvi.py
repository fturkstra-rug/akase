import json
import csv
from trankit import Pipeline

pipeline = Pipeline(lang='english')

with open('seed_data.json', 'r') as f:
    seed_data = json.load(f)

input_data = []
for row in seed_data:
    text = row.get("issue", "")
    text_id = row.get("uuid", "")

    if not text.strip():
        continue

    doc = pipeline(text)
    for i, sentence in enumerate(doc['sentences']):
        input_data.append(
            {
                "Text-ID": text_id,
                "Sentence-ID": i + 1,
                "Text": sentence['text']
            }
        )


with open('sentences.tsv', 'w', newline='', encoding='utf-8') as tsvfile:
    fieldnames = ['Text-ID', 'Sentence-ID', 'Text']
    writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')

    writer.writeheader()  # Write column headers
    for row in input_data:
        writer.writerow(row)

