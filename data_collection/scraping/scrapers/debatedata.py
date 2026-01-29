from .base_scraper import BaseScraper
from schema import *
import requests
import time
from typing import Optional
from tqdm import tqdm


class DebateDataScraper(BaseScraper):
    @property
    def url(self) -> str:
        return "https://debatedata.io/api/motion"
    
    @property
    def name(self) -> str:
        return "debatedata"
    
    def scrape(self) -> None:
        # First, get the motions, which are used as payloads to fetch the arguments for that motion.
        motions = self._get_motions()

        if not motions:
            print("No motions found.")
            return
        
        print(f"Total motions scraped: {len(motions)}")
        self._parse(motions)
        self.print_entities()
        self._save()
    
    def _get_motions(self) -> list[dict]:
        url = self.url + "/update-infinite-motions"

        all_motions = []
        page_number = 1

        while True:
            payload = {
                "sortDate": False,
                "motionTypes": [],
                "citiesActivated": [],
                "dateRange": [1981, 2025],
                "difficultyRange": None,
                "level": [],
                "pageLimit": 12,
                "pageNumber": page_number,
                "randomActivated": False,
                "searchText": "",
                "selectedAdjudicators": None,
                "style": [],
                "video": False,
            }

            data = self._call_api(url, payload)

            if not data.get("motions"):
                break

            all_motions.extend(data["motions"])
            print(f"Finished scraping page {page_number}. Total motions: {len(all_motions)}")
            page_number += 1
            time.sleep(1)

        return all_motions

    def _call_api(self, url, payload):
        headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    
    def _parse(self, motions) -> None:
        url = self.url + "/get-arguments"

        for motion in tqdm(motions):
            domains = self._add_domains(motion)
            issue = self._add_issue(motion)

            if not issue:
                continue

            if domains:
                for domain in domains:
                    self._add_entity(InDomain(source=issue.id, target=domain.id))

            argument_ids = motion.get("proArguments", []) + motion.get("conArguments", [])
            if argument_ids:
                payload = {"motion": motion}
                argument_data = self._call_api(url, payload)
                self._add_arguments(argument_data, issue.id)
                time.sleep(0.5)
    
    def _add_domains(self, motion: dict) -> list[Domain]:
        domains = []
        types = motion.get("Types")
        if not types:
            return []
        
        for name in types:
            domain = Domain(id=self._get_id(name), name=name)
            domains.append(domain)
            self._add_entity(domain)

        return domains

    def _add_issue(self, motion: dict) -> Optional[Issue]:
        text = motion.get("Motion")
        if not text:
            return None
        
        issue = Issue(
            id=self._get_id(text),
            text=text,
            source=self.name,
            url=motion.get("URL", self.url),
            timestamp=self._get_timestamp(),
            context=motion.get("Infoslide", None)
        )
        self._add_entity(issue)
        return issue
    
    def _process_arguments(self, arguments: list[dict], relation_cls: Edge, issue_id: str, url: str) -> None:
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
                id=self._get_id(text),
                text=text,
                source=self.name,
                url=url,
                timestamp=self._get_timestamp(),
                context=context or None
            )
            self._add_entity(argument)
            self._add_entity(relation_cls(source=argument.id, target=issue_id))
    
    def _add_arguments(self, data: dict, issue_id: str) -> None:
        url = self.url + "/get-arguments"
        pro_arguments = data.get('proArguments', [])
        con_arguments = data.get('conArguments', [])

        self._process_arguments(pro_arguments, Supports, issue_id, url)
        self._process_arguments(con_arguments, Attacks, issue_id, url)
