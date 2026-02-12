from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from schema import *
from typing import Optional


class BritannicaScraper(BaseScraper):    
    @property
    def url(self) -> str:
        return "https://www.britannica.com/procon"
    
    @property
    def name(self) -> str:
        return "britannica"

    def _parse(self, html: str) -> None:
        soup = BeautifulSoup(html, "html.parser")

        for h4 in tqdm(soup.find_all("h4")):
            domain = self._get_domain(h4)
            if domain.name == "New Features":
                self._remove_entity(domain)
                continue

            grid_div = h4.find_next_sibling(lambda tag: tag.name == "div" and "grid" in tag.get("class", []))
            if not grid_div:
                continue
            
            for div in grid_div.find_all("div", recursive=False):
                a_tag = div.find("a")
                if not a_tag:
                    continue

                url = a_tag.get("href")
                html = self._fetch(url)
                soup = BeautifulSoup(html, "html.parser")

                issue = self._get_issue(soup, url)
                if not issue:
                    continue
                
                topic = self._get_topic(soup)
                if topic:
                    self._add_entity(AboutTopic(source=issue.id, target=topic.id))
                    if domain:
                        self._add_entity(InDomain(source=topic.id, target=domain.id))
                elif domain:
                    self._add_entity(InDomain(source=issue.id, target=domain.id))

                self._get_arguments(soup, url, issue.id)
    
    def _get_domain(self, tag: BeautifulSoup) -> Optional[Domain]:
        name = tag.get_text(" ", strip=True)
        if not name:
            return None
            
        domain = Domain(name=name, id=self._get_id(name))
        self._add_entity(domain)
        return domain

    def _get_issue(self, soup: BeautifulSoup, url: str) -> Optional[Issue]:
        div = soup.find("div", class_=lambda c: c and c.startswith("topic-identifier"))
        if not div:
            return None
        
        text = div.get_text(" ", strip=True)
        if not (text and text.endswith("?")):
            return None
        
        issue = Issue(
            id=self._get_id(text),
            text=text,
            source=self.name,
            url=url,
            timestamp=self._get_timestamp()
        )
        self._add_entity(issue)
        return issue

    def _get_topic(self, soup: BeautifulSoup) -> Optional[Topic]:
        h1 = soup.find("h1")
        if not h1:
            return None
        
        name = h1.get_text(strip=True)
        if not name:
            return None
        
        topic = Topic(name=name, id=self._get_id(name))
        self._add_entity(topic)
        return topic

    def _get_arguments(self, soup: BeautifulSoup, url: str, issue_id: str) -> None:        
        for stance in ["pro", "con"]:
            sections = soup.find_all("section", class_=stance)

            for section in sections:
                for h2 in section.find_all("h2", class_="h2"):

                    # Remove any leading "Pro x:" or "Con x:"
                    raw_text = h2.get_text(strip=True)
                    text = re.sub(r"^(Pro|Con)\s*\d+:\s*", "", raw_text)

                    # Collect paragraphs until the next h2
                    paragraphs = []
                    next_tag = h2.find_next_sibling()
                    
                    while next_tag:
                        if next_tag.name == "h2":
                            break 

                        if next_tag.name == "p":
                            paragraph = next_tag.get_text(strip=True)
                            if paragraph:
                                paragraphs.append(paragraph)

                        next_tag = next_tag.find_next_sibling()

                    context = "\n\n".join(paragraphs)

                    argument = Argument(
                        id=self._get_id(text),
                        text=text,
                        source=self.name,
                        url=url,
                        timestamp=self._get_timestamp(),
                        context=context,
                    )
                    self._add_entity(argument)

                    if stance == "pro":
                        self._add_entity(Supports(source=argument.id, target=issue_id))
                    else:
                        self._add_entity(Attacks(source=argument.id, target=issue_id))
