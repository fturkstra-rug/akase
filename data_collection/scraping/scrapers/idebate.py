from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
from bs4.element import Tag
from typing import Optional
import math
from collections import defaultdict
from schema import *
from tqdm import tqdm
import re


class IDebateScraper(BaseScraper):
    @property
    def url(self) -> str:
        return "https://idebate.net/resources/debatabase"
    
    @property
    def name(self) -> str:
        return "idebate"

    def _parse(self, html: str) -> None:
        soup = BeautifulSoup(html, "html.parser")

        for div in soup.find_all("div", class_="row debatabase-categories"):
            for a_tag in tqdm(div.find_all("a", class_="card-tools__title")):
                url = a_tag["href"]
                domain = self._get_domain(a_tag, url)

                num_pages = self._get_num_pages(url)

                for page in range(1, num_pages + 1):
                    page_url = f"{url}?page={page}"
                    issues = self._get_issues(page_url)

                    for issue in issues:
                        self._get_arguments(issue)

                        if domain:
                            self._add_entity(InDomain(source=issue.id, target=domain.id))

                print(f"Finished scraping {domain.name}.")

    def _get_domain(self, tag: Tag, url: str) -> Optional[Domain]:
        name = tag["title"].strip()
        if not name:
            return None
        
        domain = Domain(
            id=self._get_id(name),
            name=name,
        )
        self._add_entity(domain)
        return domain

    def _get_num_pages(self, url):
        html = self._fetch(url)
        soup = BeautifulSoup(html, "html.parser")

        counter_div = soup.find("div", class_="search-results__counter")
        if counter_div:
            count_text = counter_div.find("span", class_="font-weight-bold").get_text(" ", strip=True)
            total_results = int(count_text.split("of")[-1].strip())
            return math.ceil(total_results / 12)
        return 1
    
    def _get_issues(self, url: str) -> list[Issue]:
        html = self._fetch(url)
        soup = BeautifulSoup(html, "html.parser")
        issues = []

        for h3 in soup.find_all("h3"):
            issue_text = h3.get_text(" ", strip=True)
            if not issue_text:
                continue

            a_tag = h3.find_parent("a")
            if a_tag and a_tag.has_attr("href"):
                issue_url = "https://idebate.net" + a_tag["href"]
            else:
                continue
            
            issue = Issue(
                id=self._get_id(issue_text),
                text=issue_text,
                source=self.name,
                url=issue_url,
                timestamp=self._get_timestamp()
            )
            self._add_entity(issue)
            issues.append(issue)

        return issues
    
    def _get_arguments(self, issue: Issue) -> None:
        html = self._fetch(issue.url)
        soup = BeautifulSoup(html, "html.parser")

        # --- Step 0: Find and parse all accordion__item blocks ---
        accordion_items = soup.find_all("div", class_="accordion__item")
        parsed = [self.parse_accordion_div(div) for div in accordion_items]
        parsed = [x for x in parsed if x is not None]
            
        if not parsed:
            print(f"No accordion items found, skipping {issue.url}")
            return

        # --- Step 1: Group divs by accordion_number ---

        # Group by accordion_number
        grouped_by_id = defaultdict(list)
        for item in parsed:
            grouped_by_id[item["accordion_number"]].append(item)

        # Sort the unique accordion_numbers
        unique_ids = sorted(grouped_by_id.keys())

        # Warning if there are not exactly two unique ids
        if len(unique_ids) != 2:
            print(f"Warning: Expected 2 unique accordion IDs but found {len(unique_ids)}: {unique_ids}")

        # Assign groups
        pro_id = unique_ids[0]
        con_id = unique_ids[1] if len(unique_ids) > 1 else None  # Avoid crash if only one group

        group_pro = grouped_by_id[pro_id]
        group_con = grouped_by_id[con_id] if con_id is not None else []

        # --- Step 2: Generate arguments_1 and arguments_2 ---

        # Map title to entry
        def build_title_map(group):
            return {entry["title"]: entry for entry in group}

        pro_titles = build_title_map(group_pro)
        con_titles = build_title_map(group_con)

        # Collect all unique titles
        all_titles = set(pro_titles.keys()).union(con_titles.keys())

        for title in all_titles:
            in_pro = title in pro_titles
            in_con = title in con_titles

            if in_pro:
                entry = pro_titles[title]
            elif in_con:
                entry = con_titles[title]
            else:
                continue

            argument = Argument(
                id=self._get_id(title),
                text=title,
                source=self.name,
                url=issue.url,
                timestamp=self._get_timestamp(),
                context=entry["point"]
            )

            counter_point = entry["counterpoint"]
            sentences = re.split(r'(?<=[.!?])\s+', counter_point)
            text = sentences[0]
            context = " ".join(sentences[1:])

            counter_argument = Argument(
                id=self._get_id(text),
                text=text,
                source=self.name,
                url=issue.url,
                timestamp=self._get_timestamp(),
                context=context,
            )

            # Add nodes
            self._add_entity(argument)
            self._add_entity(counter_argument)

            # Add edges
            self._add_entity(Attacks(source=counter_argument.id, target=argument.id))

            if (in_pro and in_con) or in_pro:
                self._add_entity(Supports(source=argument.id, target=issue.id))
            elif in_con:
                self._add_entity(Attacks(source=argument.id, target=issue.id))

    def parse_accordion_div(self, div):
        result = {}

        # Extract the id from the accordion__head div to get the group
        head_div = div.find(class_="accordion__head")
        if head_div and "id" in head_div.attrs:
            match = re.search(r"accordion-(\d+)", head_div["id"])
            if match:
                result["accordion_number"] = int(match.group(1))
            else:
                return None

        h4 = div.find("h4", class_="accordion__subtitle")
        result["title"] = h4.get_text(" ", strip=True) if h4 else ""

        body_div = div.find(class_="accordion__body")
        if body_div:
            paragraphs = div.find_all("p")
            if len(paragraphs) >= 2:
                result["point"] = paragraphs[0].get_text(" ", strip=True)
                result["counterpoint"] = paragraphs[1].get_text(" ", strip=True)
        
        return result
