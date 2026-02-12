import json
import csv
import spacy
from tqdm import tqdm
import argparse
from pathlib import Path
from constants import DOMAINS

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help="Seed data file (.json)", required=True, type=str)
    parser.add_argument("-o", "--output_file", help="File to which to save prepared model inputs", required=False, type=str)
    parser.add_argument("-d", "--domain", help="Domain to extract issues from", required=False, type=str, choices=DOMAINS)
    return parser.parse_args()

def sentence_tokenize(text, pipeline):
    doc = pipeline(text)
    return [sentence.text for sentence in doc.sents]

def format_data(text_id, sentences):
    data = []
    for i, sentence in enumerate(sentences):
        data.append({
            "Text-ID": text_id,
            "Sentence-ID": i + 1,
            "Text": sentence,
        })
    return data

def save(data, output_file):
    # Writes prepared model inputs to disk (.tsv)
    with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
        fieldnames = ['Text-ID', 'Sentence-ID', 'Text']
        writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader() 

        for row in data:
            writer.writerow(row)

def main():
    args = create_arg_parser()

    subgraph = args.domain is not None
    output_file = Path(args.output_file)
    input_file = Path(args.input_file)
    if input_file.exists():
        print("File already exists, exiting program.")
        exit()

    with open(input_file, "r") as f:
        seed_data = json.load(f)

    print("Loading SpaCy model...")
    nlp = spacy.load("en_core_web_sm")

    input_data = []
    for row in tqdm(seed_data, desc="Processing seed data"):
        issue_text = row.get("issue", "")
        issue_uuid = row.get("uuid", "")

        # Skip empty texts
        if not issue_text.strip():
            continue
        
        # Extract only issues related to domain
        if subgraph:
            domain = row.get("domain", [])
            if not (args.domain in domain):
                continue
        
        # Split into sentences and prepare input format
        issue_sentences = sentence_tokenize(issue_text, nlp) 
        issue_data = format_data(issue_uuid, issue_sentences)
        input_data.extend(issue_data)

        # Now do the same for arguments
        arguments = row.get("arguments", {})
        if not arguments:
            continue

        for stance in ["pro", "con"]:
            stance_arguments = arguments.get(f"{stance}_arguments", [])

            for i, argument in enumerate(stance_arguments):
                arg_id = f"{issue_uuid}-{stance}-{i + 1}"
                arg_sentences = sentence_tokenize(argument, nlp)
                arg_data = format_data(arg_id, arg_sentences)
                input_data.extend(arg_data)
    
    # Save data
    save(input_data, output_file)

if __name__ == "__main__":
    main()
