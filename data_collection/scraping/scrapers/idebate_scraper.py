from base_scraper import BaseScraper
from bs4 import BeautifulSoup
import math
import json
from tqdm import tqdm
from collections import defaultdict
import re


class IDebateScraper(BaseScraper):
    def scrape(self):
        # Start by scraping motions
        # self.get_motions()
        # self.save()

        with open('idebate.json', 'r') as f:
            data = json.load(f)

        for row in data:
            self.data['domain'].append(row['domain'])
            self.data['topic'].append(row['topic'])
            self.data['motion'].append(row['motion'])
            self.data['url'].append(row['url'])

        self.data['arguments'] = self.get_arguments()
        self.save()

    def get_motions(self):
        html = self.fetch(self.url)
        soup = BeautifulSoup(html, "html.parser")

        for div in soup.find_all("div", class_="row debatabase-categories"):
            for a_tag in div.find_all("a", class_="card-tools__title"):
                domain = a_tag["title"]
                url = a_tag["href"]

                num_pages = self.get_num_pages(url)

                for page in range(1, num_pages + 1):
                    page_url = f"{url}?page={page}"
                    data = self.parse(page_url)
                    for entry in data:
                        row = {
                            "domain": domain,
                            "topic": "",
                            "motion": entry["motion"],
                            "url": "https://idebate.net" + entry["url"],
                            "arguments": {}
                        }
                        for key, value in row.items():
                            self.data[key].append(value)

                print(f"Finished scraping {domain}.")

    def get_num_pages(self, url):
        html = self.fetch(url)
        soup = BeautifulSoup(html, "html.parser")

        counter_div = soup.find("div", class_="search-results__counter")
        if counter_div:
            count_text = counter_div.find("span", class_="font-weight-bold").get_text(
                strip=True
            )
            total_results = int(count_text.split("of")[-1].strip())
            return math.ceil(total_results / 12)
        return 1
    
    def parse(self, url):
        html = self.fetch(url)
        soup = BeautifulSoup(html, "html.parser")

        data = []
        h3_tags = soup.find_all("h3")
        
        for h3 in h3_tags:
            text = h3.get_text(strip=True)

            a_tag = h3.find_parent("a")
            if a_tag and a_tag.has_attr("href"):
                url = a_tag["href"]
            else:
                url = self.url

            data.append({
                "motion": text,
                "url": url
            })
        return data
    
    def get_arguments(self):
        results = []
        unparsable_count = 0

        for url in tqdm(self.data["url"]):
            data = {
                "pro_arguments": [],
                "con_arguments": [],
                "pro_arguments_long": [],
                "con_arguments_long": []
            }

            if url == self.url:
                print(f"No unique url found, skipping: {url}")
                results.append(data)
                continue

            html = self.fetch(url)
            soup = BeautifulSoup(html, "html.parser")

            # --- Step 0: Find and parse all accordion__item blocks ---
            accordion_items = soup.find_all("div", class_="accordion__item")
            parsed = [self.parse_accordion_div(div) for div in accordion_items]
            
            if not parsed:
                print(f"No accordion items found, skipping {url}")
                unparsable_count += 1
                results.append(data)
                continue

            # --- Step 1: Group divs by accordion_number ---

            # Group by accordion_number
            grouped_by_id = defaultdict(list)
            for item in parsed:
                grouped_by_id[item['accordion_number']].append(item)

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

            # Helper: map title to entry for easy lookup
            def build_title_map(group):
                return {entry['title']: entry for entry in group}

            pro_titles = build_title_map(group_pro)
            con_titles = build_title_map(group_con)

            # Collect all unique titles
            all_titles = set(pro_titles.keys()).union(con_titles.keys())

            for title in all_titles:
                in_pro = title in pro_titles
                in_con = title in con_titles

                # Get the corresponding entry
                if in_pro:
                    entry = pro_titles[title]
                elif in_con:
                    entry = con_titles[title]
                else:
                    continue  # Shouldn't happen

                # Apply logic
                if (in_pro and in_con) or in_pro:
                    data["pro_arguments"].append(title)
                    data["pro_arguments_long"].append(entry['point'])
                    data["con_arguments_long"].append(entry['counterpoint'])
                elif in_con:
                    data["con_arguments"].append(title)
                    data["con_arguments_long"].append(entry['point'])
                    data["pro_arguments_long"].append(entry['counterpoint'])
            
            results.append(data)

        return results
            

    def parse_accordion_div(self, div):
        result = {}

        # Extract the id from the accordion__head div to get the group
        head_div = div.find(class_="accordion__head")
        if head_div and 'id' in head_div.attrs:
            match = re.search(r'accordion-(\d+)', head_div['id'])
            if match:
                result['accordion_number'] = int(match.group(1))

        # Extract the h4 title
        h4 = div.find('h4', class_='accordion__subtitle')
        result['title'] = h4.get_text(strip=True) if h4 else ''

        # Extract the body div
        body_div = div.find(class_='accordion__body')
        if body_div:
            body_text = body_div.get_text(separator="\n", strip=True)
            # Split the body text by POINT / COUNTERPOINT
            # Expecting format: "POINT\n<text>\nCOUNTERPOINT\n<text>"
            sections = re.split(r'\bPOINT\b|\bCOUNTERPOINT\b', body_text)
            if len(sections) >= 2:
                result['point'] = sections[1].strip()
            if len(sections) >= 3:
                result['counterpoint'] = sections[2].strip()
        
        return result

    
    def post_process(self, df):
        df = super().post_process(df)
        remap = {
            "Culture": "Society & Culture",
            "Digital Freedoms": "Science & Technology",
            "Economy": "Economy",
            "Education": "Education",
            "Environment": "Environment",
            "Free Speech Debate": "Law",
            "Health": "Health",
            "International": "International Relations",
            "Law": "Law",
            "Philosophy": "Philosophy",
            "Politics": "Politics & Government",
            "Religion": "Religion",
            "Science": "Science & Technology",
            "Society": "Society & Culture",
            "Sport": "Sports",
        }
        # df["domain"] = df["domain"].apply(lambda domain: remap[domain])
        return df
    

if __name__ == "__main__":
    scraper = IDebateScraper(
        url="https://idebate.net/resources/debatabase",
        output_file="idebate_args.json",
    )
    scraper.scrape()
