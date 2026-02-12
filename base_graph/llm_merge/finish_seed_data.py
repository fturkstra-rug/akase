from collections import defaultdict
import json
from tqdm import tqdm
import re
import uuid


def extract_last_user_message(prompt: str) -> str:
    # Find all user message blocks
    user_blocks = re.findall(r"<\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|eot_id\|>", prompt, re.DOTALL)
    # Return the last one (strip for clean formatting)
    return user_blocks[-1].strip() if user_blocks else None

def get_arguments(cluster, indices):
    arguments = defaultdict(list)

    for idx in set(indices):
        if idx >= 0 and idx < len(cluster):
            args = cluster[idx].get('arguments', {})
            for key, value in args.items():
                for arg in value:
                    if not arg in arguments[key]:
                        arguments[key].append(arg)

    return arguments

def get_domains(cluster, indices):
    domains = set()

    for idx in set(indices):
        if idx >= 0 and idx < len(cluster):
            domain = cluster[idx].get('domain', '')
            if domain:
                domains.add(domain)

    return list(domains)

# Load and prepare data
with open('seed_data_raw.json', 'r') as raw_file, open('model_outputs.jsonl', 'r') as merged_file:
    raw_data = json.load(raw_file)
    merged_data = [json.loads(line) for line in merged_file]


# Preprocess the arguments in raw_data (all are now in the same format with pro and con arguments which are lists of strings, each string is a single argument)
for row in raw_data:
    url = row.get('url', '')

    if not url:
        print(row)
        exit('No URL found in raw_data. Please check the input data.')

    if 'isidewith' in url:
        row['arguments'] = {'pro_arguments': [], 'con_arguments': []}
    elif 'kialo-edu' in url:
        continue
    elif 'idebate' in url:
        # Replace the short arguments with the long arguments (because they do not map 1-1)
        arguments = row.get('arguments', {})

        row['arguments'] = {
            'pro_arguments': arguments.get('pro_arguments_long', []), 
            'con_arguments': arguments.get('con_arguments_long', [])
            }
    elif 'britannica' in url:
        # Add the short arguments as the first element of the long arguments
        arguments = row.get('arguments', {})

        for stance in ['pro', 'con']:
            short_arguments = arguments.get(f'{stance}_arguments', [])
            long_arguments = arguments.get(f'{stance}_arguments_long', [])

            new_arguments = []
            for short, long in zip(short_arguments, long_arguments):
                long.insert(0, short)
                argument = ' '.join(long)
                new_arguments.append(argument)

            row['arguments'][f'{stance}_arguments'] = new_arguments

        del row['arguments']['pro_arguments_long']
        del row['arguments']['con_arguments_long']
    else: 
        # Debatedata.io: each argument is a dict, remove the '_id' key. All other keys are separate arguments.
        arguments = row.get('arguments', {})

        for stance in ['pro', 'con']:
            args = arguments.get(f'{stance}_arguments', [])

            if not args:
                continue

            new_arguments = []
            for arg in args:
                try:
                    for key, value in arg.items():
                        if key != '_id':
                            new_arguments.append(value)
                except AttributeError:
                    # There is one entry with only None values as arguments
                    new_arguments = []
                    break
            
            row['arguments'][f'{stance}_arguments'] = new_arguments
                

# Link the arguments from the raw data to the merged issues in the model outputs
uuid_to_arguments = {row['uuid']: row.get('arguments', {}) for row in raw_data}
uuid_to_domains = {row['uuid']: row.get('domain', '') for row in raw_data}

clusters = defaultdict(list)
for row in raw_data:
    cluster_id = row['cluster']
    clusters[cluster_id].append(row)

seed_data = []
for row in tqdm(merged_data):
    generation = row.get('modelOutput').get('generation')
    record_id = row.get('recordId')

    # Noise data points have a uuid as record_id which has a length of 16 (>10).
    # Otherwise, the record_id is the same as the cluster_id.
    cluster_id = '-1' if len(str(record_id)) > 10 else record_id

    if cluster_id == '-1':
        row_data = {
            'issue': generation.strip(),
            'arguments': uuid_to_arguments.get(record_id, {}),
            'cluster_id': cluster_id,
            'uuid': record_id,
            'domain': [uuid_to_domains.get(record_id, '')]
        }
        seed_data.append(row_data)
    else:
        try:
            model_output = json.loads(generation.strip())
        except json.JSONDecodeError:
            print(f"Error decoding JSON for record ID {record_id}: {generation}")
            continue

        for idx, (motion, indices) in enumerate(model_output.items()):
            row_data = {
                'issue': motion,
                'arguments': get_arguments(clusters[int(cluster_id)], indices),
                'cluster_id': cluster_id,
                'uuid': uuid.uuid4().hex,
                'domain': get_domains(clusters[int(cluster_id)], indices)
            }
            seed_data.append(row_data)
    

with open('seed_data_panda.json', 'w') as seed_file:
    json.dump(seed_data, seed_file, indent=2, ensure_ascii=False)
