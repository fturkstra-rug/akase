from base_scraper import BaseScraper
from bs4 import BeautifulSoup


class KialoBlogScraper(BaseScraper):
    def scrape(self):
        blog_urls = [
            "https://blog.kialo-edu.com/debate-ideas/history-discussion-topics-for-the-classroom/",
            "https://blog.kialo-edu.com/debate-ideas/environmental-debate-topics/",
            "https://blog.kialo-edu.com/lesson-ideas/history-debate-topics/",
            "https://blog.kialo-edu.com/lesson-ideas/classroom-debate-ideas/",
            "https://blog.kialo-edu.com/debate-ideas/political-discussion-topics-for-the-classroom/",
            "https://blog.kialo-edu.com/debate-ideas/philosophical-debate-topics/",
            "https://blog.kialo-edu.com/debate-ideas/sports-debate-topics/",
            "https://blog.kialo-edu.com/lesson-ideas/political-debate-topics/",
            "https://blog.kialo-edu.com/debate-ideas/debate-topics-for-high-school-students/",
            "https://blog.kialo-edu.com/debate-ideas/debate-topics-for-college-students/",
            "https://blog.kialo-edu.com/debate-ideas/persuasive-writing-topics/",
            "https://blog.kialo-edu.com/lesson-ideas/science-debate-topics/",
            "https://blog.kialo-edu.com/debate-ideas/debate-topics-for-kids/",
            "https://blog.kialo-edu.com/lesson-ideas/literature-debate-topics-for-the-classroom/",
            "https://blog.kialo-edu.com/debate-ideas/fun-debate-topics/",
            "https://blog.kialo-edu.com/debate-ideas/debate-topics-for-middle-school-students/",
            "https://blog.kialo-edu.com/debate-ideas/discussion-topics-for-high-school-students/",
        ]

        for blog_url in blog_urls:
            html = self.fetch(blog_url)
            soup = BeautifulSoup(html, "html.parser")

            for element in soup.find_all(["h2", "ul"]):
                if element.name == "h2":
                    domain = element.get_text(strip=True)
                elif element.name == "ul" and "wp-block-list" in element.get(
                    "class", []
                ):
                    for li in element.find_all("li"):
                        motion = (
                            li.get_text(strip=True)
                            if not li.find("a")
                            else li.find("a").get_text(strip=True)
                        )
                        if motion:
                            domain = domain if domain is not None else ""
                            row = {
                                "domain": domain,
                                "topic": "",
                                "motion": motion,
                                "url": blog_url,
                                "arguments": {}
                            }
                            for key, value in row.items():
                                self.data[key].append(value)

            print(f"Finished scraping {blog_url}.")

        self.save()

    def post_process(self, df):
        print("Warning: the Kialo blogs contain over 150 domains, some of the domains contain another domain or a topic.")
        print("Therefore, the raw kialo blog data is manually post-processed.")
        input("Press <ENTER> to confirm you read this message.")
        return super().post_process(df)


if __name__ == "__main__":
    scraper = KialoBlogScraper(
        url="https://blog.kialo-edu.com/",
        output_file="kialo_blog.json"
    )
    scraper.scrape()