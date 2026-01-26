import os
import glob
import json
import re
from pathlib import Path


directory = "debates"
debate_files = glob.glob(os.path.join(directory, "*.json"))

validation_file = "test.json"
output_file = "debate_test.json"

with open(validation_file, 'r') as f:
    val_data = json.load(f)

output = {}
for debate_file in debate_files:
    with open(debate_file, 'r') as f:
        data = json.load(f)

        # Extract intervention
        debate_name = Path(debate_file).stem
        intervention_id = debate_name.removeprefix('debate_')

        # Extract questions
        questions = data["questions"]
        questions = re.findall(r'\d+\.\s(.*?)(?=\s\d+\.|$)', questions)

        # Extract indices
        verdict = data["verdict"]
        start = verdict.find('[')
        end = verdict.find(']')
        indices = eval(verdict[start+1:end])

        # Find useful questions
        useful_questions = [questions[i-1] for i in indices ]
        cqs = [{"id": i, "cq": cq} for i, cq in enumerate(useful_questions[:3])]

    intervention = val_data[intervention_id]
    del intervention["schemes"]
    intervention["cqs"] = cqs

    output[intervention_id] = intervention

with open(output_file, 'w') as f:
    json.dump(output, f, indent=4)

print(f"Saved results to {output_file}")
