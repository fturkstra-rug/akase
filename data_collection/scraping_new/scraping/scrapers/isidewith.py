from .base_scraper import BaseScraper
from schema import *
from bs4 import BeautifulSoup
from bs4.element import Tag
from typing import Optional
from tqdm import tqdm


class ISideWithScraper(BaseScraper):
    @property
    def url(self) -> str:
        return "https://www.isidewith.com/polls"
    
    @property
    def name(self) -> str:
        return "isidewith"
    
    def scrape(self) -> None:
        with open("data/scraping/isidewith.html", "r", encoding="utf-8") as f:
            html = f.read()

        if html:
            self._parse(html)
            self._save()
            self.print_entities()

    def _parse(self, html: str) -> None:
        soup = BeautifulSoup(html, "html.parser")

        for div in tqdm(soup.find_all("div", class_="sec_c")):
            
            domain = self._get_domain(div)
            if domain.name in ["Historical Importance", "Current Events Issues"]:
                self._remove_entity(domain)
                continue

            topic = self._get_topic(div)

            if topic and domain:
                self._add_entity(InDomain(source=topic.id, target=domain.id))

            for h3_tag in div.find_all("h3"):
                issue = self._get_issue(h3_tag)

                if topic:
                    self._add_entity(AboutTopic(source=issue.id, target=topic.id))
                elif domain:
                    self._add_entity(InDomain(source=issue.id, target=domain.id))

    def _get_domain(self, tag: Tag) -> Optional[Domain]:
        h2 = tag.find("h2")
        if not h2:
            return None
        
        name = h2.get_text(" ", strip=True)
        if not name:
            return None
            
        domain = Domain(name=name, id=self._get_id(name))
        self._add_entity(domain)
        return domain

    def _get_topic(self, tag: Tag) -> Optional[Topic]:
        topic_tag = tag.find("a", class_="name", rel="nofollow")
        if not topic_tag:
            return None
        
        name = topic_tag.get_text(" ", strip=True)
        if not name:
            return None
        
        topic = Topic(name=name, id=self._get_id(name))
        self._add_entity(topic)
        return topic

    def _get_issue(self, tag: Tag) -> Optional[Issue]:
        a_tag = tag.find("a")
        if not a_tag:
            return None
                
        text = a_tag.get_text(strip=True)
        if not text:
            return None
        
        url = "https://www.isidewith.com" + a_tag["href"]

        issue = Issue(
            id=self._get_id(text),
            text=text,
            source=self.name,
            url=url,
            timestamp=self._get_timestamp()
        )
        self._add_entity(issue)
        return issue
