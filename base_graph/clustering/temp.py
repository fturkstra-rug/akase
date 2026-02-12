import json
from tqdm import tqdm

batch_size = 100

# Load the full clusters data first (since clusters['cluster_labels'] is likely a list of labels)
with open('cluster_results.json') as f2:
    clusters = json.load(f2)

total = len(clusters['cluster_labels'])

# Function to load motions in batches from motions.json
def load_motions_batch(start, size):
    with open('motions.json') as f:
        # Load only the required batch of motions (inefficient to load whole file, but JSON doesn't support streaming easily)
        # So load entire motions once then slice, or if very large file, use alternative storage like JSON lines or a DB.
        all_motions = json.load(f)
        return all_motions[start:start+size]

# Process and save in batches
output_file = 'clustered_motions.json'

# We'll write the JSON array opening bracket first
with open(output_file, 'w', encoding='utf-8') as f3:
    f3.write('[\n')  # start JSON list

for start in tqdm(range(0, total, batch_size)):
    motions_batch = load_motions_batch(start, batch_size)
    cluster_batch = clusters['cluster_labels'][start:start+batch_size]
    
    # Assign clusters
    for motion, cluster in zip(motions_batch, cluster_batch):
        motion['cluster'] = cluster
    
    # Append batch to output file
    with open(output_file, 'a', encoding='utf-8') as f3:
        # Dump the batch as JSON but remove the surrounding [] to continue the list
        batch_json = json.dumps(motions_batch, indent=4, ensure_ascii=False)
        # Remove first [ and last ] from batch_json
        batch_json = batch_json.strip()
        if batch_json.startswith('['):
            batch_json = batch_json[1:]
        if batch_json.endswith(']'):
            batch_json = batch_json[:-1]
        
        # If not the first batch, prepend a comma to separate JSON objects
        if start > 0:
            f3.write(',\n')
        f3.write(batch_json)

# Close the JSON list
with open(output_file, 'a', encoding='utf-8') as f3:
    f3.write('\n]')

print(f'Successfully saved {total} cluster labels in {output_file}')
