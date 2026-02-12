import requests
from datetime import datetime
import pandas as pd
from pathlib import Path
import json


class BaseScraper:
    def __init__(self, url: str, output_file: str):
        self.url = url
        self.output_file = Path(output_file)
        self.data = {"domain": [], "topic": [], "motion": [], "url": [], "arguments": []}

    def fetch(self, url=None) -> str:
        url = self.url if url is None else url
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def parse(self, html: str) -> dict:
        raise NotImplementedError("Subclasses must implement `parse` method unless the subclass overwrites the `scrape` method.")

    def post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        General post-processing:
        - Remove duplicate motions
        - Remove rows with empty strings or NA values for the motion or url
        - Remove non-English text (using a simple heuristic by keeping ASCII text only)
        - Add the date of access

        Subclasses can extend this method to implement website-specific post-processing such as remapping domains.
        """
        df = df.drop_duplicates(subset="motion", keep="first")  
        df = df.replace("", pd.NA).dropna(subset=['motion', 'url'])
        df = df[df['motion'].apply(lambda x: x.isascii())]
        df['access_date'] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        return df
    
    def save(self) -> None:
        df = pd.DataFrame(self.data)
        df = self.post_process(df)
        data = df.to_dict(orient='records')
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Saved scraping data to `{self.output_file}`.")

    def scrape(self) -> None:
        html = self.fetch()
        if html:
            self.parse(html)
            self.save()
