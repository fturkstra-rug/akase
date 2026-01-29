from abc import ABC, abstractmethod
import requests
import json
from pathlib import Path
from datetime import datetime
import hashlib
from collections import defaultdict
from dataclasses import asdict
from schema import Entity
import re


class BaseScraper(ABC):
    data = {
        "nodes": defaultdict(list),
        "edges": defaultdict(list)
    }

    def _add_entity(self, value: Entity) -> None:
        self.data[value.kind()][value.type()].append(asdict(value))

    def _remove_entity(self, value: Entity) -> None:
        entities = self.data[value.kind()][value.type()]
        entities[:] = [e for e in entities if e["id"] != value.id] # modifies the list in place

    def print_entities(self):
        nodes = self.data["nodes"]
        edges = self.data["edges"]
        print("Issues:", len(nodes.get("issue", [])))
        print("Arguments:", len(nodes.get("argument", [])))
        print("Topics:", len(nodes.get("topic", [])))
        print("Domains:", len(nodes.get("domain", [])))
        print("Human Values:", len(nodes.get("human_value", [])))
        print("\nTotal number of nodes:", sum(len(v) for v in nodes.values()), "\n")
        print("Supports:", len(edges.get("supports", [])))
        print("Attacks:", len(edges.get("attacks", [])))
        print("InDomain:", len(edges.get("in_domain", [])))
        print("AboutTopic:", len(edges.get("about_topic", [])))
        print("Attains:", len(edges.get("attains", [])))
        print("Constrains:", len(edges.get("constrains", [])))
        print("\nTotal number of edges:", sum(len(v) for v in edges.values()), "\n")

    @property
    @abstractmethod
    def url(self) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def output_file(self) -> Path:
        return Path(f"{self.name}.json")

    def scrape(self) -> None:
        html = self._fetch(self.url)
        if html:
            self._parse(html)
            self.print_entities()
            self._save()

    def _get_timestamp(self) -> str:
        return datetime.today().strftime("%Y-%m-%d %H:%M:%S")

    def _get_id(self, text: str) -> str:
        text = self._normalize(text)
        hash_obj = hashlib.sha256(text.encode("utf-8"))
        return hash_obj.hexdigest()[:16]
    
    def _normalize(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    def _fetch(self, url: str) -> str:
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    @abstractmethod
    def _parse(self, html: str) -> None:
        pass

    def _save(self) -> None:
        with open(f"scraping/data/raw/{self.output_file}", "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
