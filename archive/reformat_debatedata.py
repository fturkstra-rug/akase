import json
from collections import defaultdict
from graph.schema import *
from dataclasses import asdict
import re
import hashlib
from datetime import datetime

data = {
    "nodes": defaultdict(list),
    "edges": defaultdict(list)
}

def add_entity(value: Entity) -> None:
    global data
    data[value.kind()][value.type()].append(asdict(value))

def get_timestamp() -> str:
    return datetime.today().strftime("%Y-%m-%d %H:%M:%S")

def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def get_id(text: str) -> str:
    text = normalize(text)
    hash_obj = hashlib.sha256(text.encode("utf-8"))
    return hash_obj.hexdigest()[:16]

def get_domains(entry) -> list[Domain]:
    domains = []
    types = entry.get("Types")
    if not types:
        return []
        
    for name in types:
        domain = Domain(id=get_id(name), name=name)
        domains.append(domain)
        add_entity(domain)

    return domains

def get_issue(entry: dict) -> Optional[Issue]:
    text = entry.get("Motion")
    if not text:
        return None
        
    issue = Issue(
        id=get_id(text),
        text=text,
        source="debatedata",
        url=entry.get("URL", "https://debatedata.io/api/motion"),
        timestamp=get_timestamp(),
        context=entry.get("Infoslide", None)
    )
    add_entity(issue)
    return issue

def process_arguments(arguments: list[dict], relation_cls: Edge, issue_id: str, url: str) -> None:
    for arg in arguments:

        if not arg:
            continue
            
        text, context = "", ""
        for key, value in arg.items():
            if key == "Premise":
                text = value
            elif key == "_id":
                continue
            else:
                context += f"[{key}]\n{value}\n"

        if not text:
            continue

        argument = Argument(
            id=get_id(text),
            text=text,
            source="debatedata",
            url=url,
            timestamp=get_timestamp(),
            context=context or None
        )
        add_entity(argument)
        add_entity(relation_cls(source=argument.id, target=issue_id))

def main():
    global data

    with open("/home/frieso-turkstra/Documents/werk/akase/seed_data/data/scrape_data/debatedata_raw.json", "r") as f:
        raw_data = json.load(f)

    for entry in raw_data:
        domains = get_domains(entry)
        issue = get_issue(entry)

        if not issue:
            continue

        if domains:
            for domain in domains:
                add_entity(InDomain(source=issue.id, target=domain.id))

        url = "https://debatedata.io/api/motion/get-arguments"
        pro_arguments = entry.get('proArguments', [])
        con_arguments = entry.get('conArguments', [])

        process_arguments(pro_arguments, Supports, issue.id, url)
        process_arguments(con_arguments, Attacks, issue.id, url)

    with open(f"/home/frieso-turkstra/Documents/werk/akase/final/data/scraping/raw/debatedata.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
