import json
import glob
import sys
import uuid

# Get the directory from command-line arguments
args = sys.argv[1:]
directory = args[0]

# Find all JSON files in the directory
file_paths = glob.glob(f"{directory}/*.json")

# List to store all entries
all_entries = []

# Read each JSON file and extend the list
for file_path in file_paths:
    if "raw" in file_path:
        continue

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                all_entries.extend(data)
            else:
                print(f"Warning: {file_path} does not contain a list of entries.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")

# Count before deduplication
total_before = len(all_entries)

# Remove entries with no "motion"
all_entries = [entry for entry in all_entries if "motion" in entry and entry["motion"] is not None]

# Deduplicate based on "motion"
seen_motions = set()
unique_entries = []
for entry in all_entries:
    motion = entry["motion"]
    if motion not in seen_motions:
        seen_motions.add(motion)
        entry["uuid"] = str(uuid.uuid4())  # Assign unique ID
        unique_entries.append(entry)

# Count after deduplication
total_after = len(unique_entries)

print(total_before, total_after)
if total_before == total_after:
    print("No duplicates found.")
else:
    print(f"Removed {total_before - total_after} duplicates.")

# Save the result to a JSON file
with open("motions.json", "w", encoding="utf-8") as f:
    json.dump(unique_entries, f, indent=2, ensure_ascii=False)

print("JSON files merged successfully!")
