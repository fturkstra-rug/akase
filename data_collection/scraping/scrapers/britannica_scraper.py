from base_scraper import BaseScraper
from bs4 import BeautifulSoup
import re
from tqdm import tqdm


class BritannicaScraper(BaseScraper):
    def parse(self, html):
        soup = BeautifulSoup(html, "html.parser")

        for h4 in tqdm(soup.find_all("h4")):
            domain = h4.get_text(strip=True)

            if domain == "New Topics":
                continue

            parent_div = h4.find_parent("div")
            if not parent_div:
                continue

            ul = parent_div.find("ul")
            if not ul:
                continue

            for li in ul.find_all("li"):
                a_tag = li.find("a")
                if a_tag:
                    topic = a_tag.get_text(strip=True)
                    url = a_tag.get("href")
                    data = self.extract_data(url)

                    if data is not None:
                        row = {
                            "domain": domain,
                            "topic": topic,
                            "motion": data["motion"],
                            "url": url,
                            "arguments": data["arguments"]
                        }
                        for key, val in row.items():
                            self.data[key].append(val)
                                                

    def extract_data(self, url):
        html = self.fetch(url)
        soup = BeautifulSoup(html, "html.parser")
        data = {}

        # Find the motion
        div = soup.find("div", class_=lambda c: c and c.startswith("topic-identifier"))
        data["motion"] = div.get_text(strip=True) if div else None

        # And the arguments
        data["arguments"] = {
            "pro_arguments": [],
            "con_arguments": [],
            "pro_arguments_long": [],
            "con_arguments_long": []
        }

        for stance in ["pro", "con"]:
            sections = soup.find_all("section", class_=stance)

            for section in sections:
                for h2 in section.find_all('h2'):
                    # Get the clean title by removing "Pro x:" or "Con x:" if present
                    raw_argument = h2.get_text(strip=True)
                    clean_argument = re.sub(r'^(Pro|Con)\s*\d+:\s*', '', raw_argument)

                    # Prepare to collect the paragraphs after this <h2>
                    paragraphs = []
                    next_tag = h2.find_next_sibling()
                    
                    while next_tag:
                        if next_tag.name == 'h2':
                            break  # Reached the next section, stop collecting

                        if next_tag.name == 'p':
                            text = next_tag.get_text(strip=True)
                            if text:
                                paragraphs.append(text)

                        next_tag = next_tag.find_next_sibling()

                    # Add the section to the list if it has any paragraphs
                    data["arguments"][f"{stance}_arguments"].append(clean_argument)
                    data["arguments"][f"{stance}_arguments_long"].append(paragraphs)

        return data
    
    
    def post_process(self, df):
        df = super().post_process(df)
        remap = {
            'Digital Life, Science, & Technology': 'Science & Technology',
            'Economy & Taxes': 'Economy',
            'Elections & Presidents': 'Politics & Government',
            'Environment & Animal Rights': 'Environment',
            'Education': 'Education',
            'Government & Civics': 'Politics & Government',
            'Health & Medicine': 'Health',
            'Immigration & International Scene': 'International Relations',
            'Law & Order': 'Law',
            'Society & Holidays': 'Society & Culture',
            'Sports': 'Sports',
        }
        df["domain"] = df["domain"].apply(lambda domain: remap[domain])
        return df


if __name__ == "__main__":
    scraper = BritannicaScraper(
        url="https://www.britannica.com/procon",
        output_file="britannica.json",
    )
    scraper.scrape()
