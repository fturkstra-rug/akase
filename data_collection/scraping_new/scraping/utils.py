from collections import defaultdict
import json
import os

def merge():
    # Merge the raw scraping results from each platform

    graph = {
        "nodes": defaultdict(list),
        "edges": defaultdict(list)
    }

    for file in os.listdir("scraping/data/raw"):

        path = os.path.join("scraping/data/raw", file)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

            for node_type, item in data["nodes"].items():
                graph["nodes"][node_type].extend(item)

            for edge_type, item in data["edges"].items():
                graph["edges"][edge_type].extend(item)

        print("Finished processing:", file)

    # Save nodes and edges to separate files
    for entity_type in ["nodes", "edges"]:
        with open(f"scraping/data/raw/{entity_type}.json", "w", encoding="utf-8") as f:
            json.dump(graph[entity_type], f, ensure_ascii=False, indent=2)

def postprocess():
    for entity_type in ["nodes", "edges"]:
        raw_data = read_file(f"scraping/data/raw/{entity_type}.json")
        clean_data = deduplicate(entity_type, raw_data)

        # if entity_type == "nodes":
        #     clean_data = validate_text(clean_data)

        save(clean_data, f"scraping/data/{entity_type}.json")

def read_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def deduplicate(entity_type: str, data: dict) -> dict:
    result = {}

    for entity_subtype, entity_list in data.items():

        seen = set()
        deduped = []

        for entity in entity_list:
            key = get_key(entity_type, entity)
            if key not in seen:
                seen.add(key)
                deduped.append(entity)

        result[entity_subtype] = deduped

    return result
    
def get_key(entity_type: str, entity: dict) -> tuple:
    if entity_type == "nodes":
        return (entity["id"])
    elif entity_type == "edges":
        return (entity["source"], entity["target"]) 

def validate_text(nodes: dict) -> dict:
    result = {}

    for node_type, node_list in nodes.items():
        
        validated = []
        key = "name" if node_type in ["domain", "topic"] else "text"
        
        for node in node_list:
            # Remove empty strings and non-English languages (use ASCII filter as simple heuristic).
            if node[key].strip() and node[key].isascii():
                validated.append(node)

        result[node_type] = validated

    return result
