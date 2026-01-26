import json
from pathlib import Path
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parent = Path('experiments')

def merge_files():
    for dir in parent.iterdir():
        if "cohere_r" in dir.name:
            continue

        df1 = pd.read_json(dir / "model_outputs_og.jsonl", lines=True)
        df2 = pd.read_json(dir / "model_outputs_nf.jsonl", lines=True)
        
        max_id = df1['recordId'].max()
        # Offset the recordIds in df2
        df2['recordId'] = df2['recordId'] + max_id

        merged_df = pd.concat([df1, df2], ignore_index=True)
        print(len(merged_df))
        merged_df.to_json(dir / "model_outputs.jsonl", orient="records", lines=True)

merge_files()

def rename_files():
    for dir in parent.iterdir():
        for file in dir.iterdir():
            if file.is_file():
                if file.name == "model_inputs.jsonl":
                    new_file = file.with_name('model_inputs_og.jsonl')
                    file.rename(new_file)
                
                elif file.name == "model_outputs.jsonl":
                    file.rename(file.with_name('model_outputs_og.jsonl'))

                # df = pd.read_json(dir / 'eval.json')
                # print(dir.name, df['macro avg']['f1-score'])

            continue

def sort_output_files_by_record_id():
    with open(dir / "model_outputs.jsonl", "r") as f:
        # Load all lines into a list of dictionaries
        lines = [json.loads(line) for line in f]

    # Sort the lines based on the 'recordId' key (as a string)
    sorted_lines = sorted(lines, key=lambda x: int(x["recordId"]))

    # Write the sorted lines back to a new .jsonl file
    with open(dir / "model_outputs.jsonl", "w") as f:
        for line in sorted_lines:
            json.dump(line, f)
            f.write("\n")
