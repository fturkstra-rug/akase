from base_scraper import BaseScraper
from bs4 import BeautifulSoup
from tqdm import tqdm


class ISideWithScraper(BaseScraper):
    def parse(self, html):
        soup = BeautifulSoup(html, "html.parser")

        for div in tqdm(soup.find_all("div", class_="sec_c")):

            # Extract the domain
            domain = div.find("h2").get_text(strip=True) if div.find("h2") else ""

            # Skip the graph on the top of the page and the current events
            if domain == "Historical Importance" or domain == "Current Events Issues":
                continue

            # Extract the motions and their topics/urls
            for h3_tag in div.find_all("h3"):

                a_tag = h3_tag.find("a")
                if not a_tag:
                    continue

                motion = a_tag.get_text(strip=True)
                url = "https://www.isidewith.com" + a_tag["href"]

                topic_tag = div.find("a", class_="name", rel="nofollow")
                topic = topic_tag.get_text(strip=True) if topic_tag else ""

                row = {"domain": domain, "topic": topic, "motion": motion, "url": url, "arguments": {}}
                for key, value in row.items():
                    self.data[key].append(value)
    
    def post_process(self, df):
        df = super().post_process(df)
        remap = {
            'Social Issues': "Society & Culture",
            'Healthcare Issues': "Health",
            'Electoral Issues': "Politics & Government",
            'Immigration Issues': "International Relations",
            'Domestic Policy Issues': "Politics & Government",
            'Criminal Issues': "Law",
            'Education Issues': "Education",
            'Foreign Policy Issues': "International Relations",
            'Economic Issues': "Economy",
            'Science Issues': "Science & Technology",
            'Housing Issues': "Politics & Government",
            'Environmental Issues': "Environment",
            'National Security Issues': "Politics & Government",
            'Transportation Issues': "Politics & Government",
            'Technological Issues': "Science & Technology",
        }
        df["domain"] = df["domain"].apply(lambda domain: remap[domain])
        return df


if __name__ == "__main__":
    scraper = ISideWithScraper(
        url="https://www.isidewith.com/polls",
        output_file="isidewith.json",
    )
    scraper.scrape()