from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
from bs4.element import Tag
from schema import *
from typing import Optional


class KialoBlogScraper(BaseScraper):
    @property
    def url(self) -> str:
        return "https://blog.kialo-edu.com/"
    
    @property
    def name(self) -> str:
        return "kialo_blog"
    
    def scrape(self) -> None:
        paths = [
            "debate-ideas/history-discussion-topics-for-the-classroom/",
            "debate-ideas/environmental-debate-topics/",
            "lesson-ideas/history-debate-topics/",
            "lesson-ideas/classroom-debate-ideas/",
            "debate-ideas/political-discussion-topics-for-the-classroom/",
            "debate-ideas/philosophical-debate-topics/",
            "debate-ideas/sports-debate-topics/",
            "lesson-ideas/political-debate-topics/",
            "debate-ideas/debate-topics-for-high-school-students/",
            "debate-ideas/debate-topics-for-college-students/",
            "debate-ideas/persuasive-writing-topics/",
            "lesson-ideas/science-debate-topics/",
            "debate-ideas/debate-topics-for-kids/",
            "lesson-ideas/literature-debate-topics-for-the-classroom/",
            "debate-ideas/fun-debate-topics/",
            "debate-ideas/debate-topics-for-middle-school-students/",
            "debate-ideas/discussion-topics-for-high-school-students/",
        ]

        for path in paths:
            url = self.url + path
            html = self._fetch(url)
            if html:
                self._parse(html, url)

        self.print_entities()
        self._save()

    def _parse(self, html: str, url: str) -> None:      
        soup = BeautifulSoup(html, "html.parser")

        domain = self._get_domain(soup)

        for element in soup.find_all(["h2", "ul"]):
            if element.name == "h2":
                topic = self._get_topic(element)

                if topic and domain:
                    self._add_entity(InDomain(source=topic.id, target=domain.id))

            elif element.name == "ul" and "wp-block-list" in element.get("class", []):
                for li in element.find_all("li"):
                    issue = self._get_issue(li, url)

                    if issue and topic:
                        self._add_entity(AboutTopic(source=issue.id, target=topic.id))
                    elif issue and domain:
                        self._add_entity(InDomain(source=issue.id, target=domain.id))

        print(f"Finished scraping {url}.")

    def _get_domain(self, soup: BeautifulSoup) -> Optional[Domain]:
        h1 = soup.find("h1")
        if not h1:
            return None
        
        name = h1.get_text(" ", strip=True)
        if not name:
            return None

        domain = Domain(id=self._get_id(name), name=name)
        self._add_entity(domain)
        return domain
    
    def _get_topic(self, tag: Tag) -> Optional[Topic]:
        name = tag.get_text(" ", strip=True)
        if not name:
            return None

        topic = Topic(id=self._get_id(name), name=name)
        self._add_entity(topic)
        return topic
    
    def _get_issue(self, tag: Tag, url: str) -> Optional[Issue]:
        a_tag = tag.find("a")
        text = (a_tag or tag).get_text(" ", strip=True)
        if not text:
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
