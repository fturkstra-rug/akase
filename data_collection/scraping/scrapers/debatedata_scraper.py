from base_scraper import BaseScraper
import requests
import json
import time
import pandas as pd
from tqdm import tqdm


class DebateDataScraper(BaseScraper):
    def scrape(self):
        # Start by scraping motions
        # motions = self.get_motions()

        # if not motions:
        #     print("Sorry, no motions here mate.")
        #     return
        
        # print(f"Total motions scraped: {len(motions)}")

        # Temporary
        with open("../../data/raw_scraping_data/debatedata_raw.json", "r", encoding="utf-8") as f:
            motions = json.load(f)

        # Then add arguments to each motion and save everything
        motions_with_arguments = self.get_arguments(motions)
        self.save(motions_with_arguments)
        

    def get_motions(self):
        # Directly call the underlying API endpoint
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

            data = self.call_api(url, payload)

            if not data.get("motions"):  # Stop if there are no more motions
                break

            all_motions.extend(data["motions"])
            print(f"Finished scraping page {page_number}. Total motions: {len(all_motions)}")
            page_number += 1
            time.sleep(1)

        return all_motions
    
    def call_api(self, url, payload):
        headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    def get_arguments(self, motions):
        url = self.url + "/get-arguments"

        for i, motion in enumerate(tqdm(motions)):
            if i < 30900: # temporary skip
                continue
            argument_ids = motion.get("proArguments", []) + motion.get("conArguments", [])
            if argument_ids:
                payload = {"motion": motion}
                argument_data = self.call_api(url, payload)
                motion['proArguments'] = argument_data.get('proArguments', [])
                motion['conArguments'] = argument_data.get('conArguments', [])
                time.sleep(0.5)

            if i % 100 == 0:
                try:
                    with open(f"temp_save_{i}.json", 'w', encoding='utf-8') as f:
                        json.dump(motions, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Error saving progress at count {i}: {e}")
        
        try:
            with open(f"fifth_run.json", 'w', encoding='utf-8') as f:
                json.dump(motions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving final progress", e)

        return motions

    def post_process(self, df):
        df = super().post_process(df)
        remap = {
            'Culture': 'Society & Culture',
            'Labor': 'Economy',
            'Development': 'Economy',
            'Asia': 'International Relations',
            'Art': 'Arts',
            'LGBTQ+': 'Society & Culture',
            'Economics': 'Economy',
            'Privacy': 'Law',
            'Democracy': 'Politics & Government',
            'Philosophy': 'Philosophy',
            'Charity': 'Society & Culture',
            'International Relations': 'International Relations',
            'Media': 'Society & Culture',
            'Education/Academia': 'Education',
            'Cities': 'Society & Culture',
            'Historical Memory': 'History',
            'Children': 'Society & Culture',
            'Environment': 'Environment',
            'Europe': 'International Relations',
            'Criminal Justice': 'Law',
            'Law': 'Law',
            'Feminism': 'Society & Culture',
            'Colonialism': 'History',
            'Medical': 'Health',
            'Africa': 'International Relations',
            'Military': 'Politics & Government',
            'Animal Rights': 'Environment',
            'Immigration': 'International Relations',
            'Corruption': 'Politics & Government',
            'Romance/Sex': 'Society & Culture',
            'Family': 'Society & Culture',
            'Nationalism': 'Politics & Government',
            'Indigenous People': 'Society & Culture',
            'Class': 'Society & Culture',
            'Agriculture': 'Economy',
            'Minority Communities': 'Society & Culture',
            'Ethics': 'Philosophy',
            'Other': '',
            'Crime': 'Law',
            'Science/Technology': 'Science & Technology',
            'Poverty': 'Economy',
            'Latin America': 'International Relations',
            'Health': 'Health',
            'Business': 'Economy',
            'Demography': 'Society & Culture',
            'Religion': 'Religion',
            'Climate Change': 'Environment',
            'Corporate Culture': 'Economy',
            'Social Justice': 'Society & Culture',
            'Civil Service': 'Politics & Government',
            'Ethics/Philosophy': 'Philosophy',
            'Civil Rights': 'Law',
            'Technology': 'Science & Technology',
            'Romance Sex': 'Other',
            'Moral Philosophy': 'Philosophy',
            'Social Policy': 'Politics & Government',
            'Sports': 'Sports',
            'Middle East': 'International Relations',
            'Terrorism': 'Politics & Government',
            'Private Property': 'Economy',
            'Cultural': 'Society & Culture',
            'Police': 'Law',
        }
        df["domain"] = df["domain"].apply(lambda domain: remap[domain])
        return df
    
    def json2df(self, data):
        rows = []
        
        for item in data:
            domain = "Other" if not (types := item["Types"]) else types[0]
            topic = ""
            motion = item["Motion"]
            url = item.get("URL", "https://debatedata.io/")
            arguments = {
                "pro_arguments": item["proArguments"],
                "con_arguments": item["conArguments"]
            }
            rows.append([domain, topic, motion, url, arguments])

        df = pd.DataFrame(rows, columns=["domain", "topic", "motion", "url", "arguments"])
        return df

    def save(self, data):
        # Save raw scraping data
        # raw_output_file = self.output_file.stem + "_raw" + self.output_file.suffix
        # with open(raw_output_file, "w", encoding="utf-8") as f:
        #     json.dump(data, f, ensure_ascii=False, indent=4)
        # print(f"Saved raw scraping data to `{raw_output_file}`.")

        # Transform json into a dataframe
        with open('debatedata_raw.json', 'r') as f:
            data = json.load(f)
        df = self.json2df(data)

        # Save post_processed data
        df = self.post_process(df)
        data = df.to_dict(orient='records')
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Saved post-processed data to `{self.output_file}`.")


if __name__ == "__main__":
    scraper = DebateDataScraper(
        url="https://debatedata.io/api/motion/",
        output_file="debatedata.json",
    )
    scraper.scrape()
