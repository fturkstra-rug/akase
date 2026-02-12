from base_scraper import BaseScraper
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
from tqdm import tqdm


class KialoEduScraper(BaseScraper):
    def scrape(self):
        # Use Selenium to click the drop down menu and select one of the following domains
        domains = [
            "Arts",
            "Civics & Society",
            "Economics & Business",
            "History",
            "Human Geography",
            "Just for Fun / Icebreakers",
            "Literature",
            "Philosophy",
            "Pop Culture & Entertainment",
            "Religious Studies",
            "Science & Technology",
            "Social-Emotional Learning",
        ]

        # Ensure you have ChromeDriver installed in the correct location
        driver = self.setup_driver()
        driver.get(self.url)

        # Allow time for elements to load
        time.sleep(5)

        for domain in domains:
            # Step 1: Click the dropdown button to reveal the options
            dropdown_button = driver.find_element(
                By.CSS_SELECTOR, "div.dropdown__button--generic"
            )
            dropdown_button.click()

            # Step 2: Wait for the dropdown to expand and the category button to be visible
            category_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, f"//button[span[text()='{domain}']]")
                )
            )

            # Scroll to the category button to ensure it's clickable
            driver.execute_script("arguments[0].scrollIntoView(true);", category_button)
            time.sleep(1)  # Allow some time for scrolling

            # Step 3: Click the category button
            category_button.click()

            # Allow time for the page to reload/filter based on the selected category
            time.sleep(3)

            # Step 4: Locate all discussion topic elements for this category
            topic_elements = driver.find_elements(
                # By.CSS_SELECTOR, "li.card-grid__card"
                By.CSS_SELECTOR, "div.line-discussion-card__container"
            )

            for topic in topic_elements:
                title_element = topic.find_element(
                    By.CSS_SELECTOR, "div.discussion-card-title__title-wrapper"
                )
                motion = title_element.text.strip() if title_element else None

                link_element = topic.find_element(
                    By.CSS_SELECTOR, "a.discussion-card-title--link"
                )
                url = link_element.get_attribute("href") if link_element else self.url

                if motion:
                    row = {"domain": domain, "topic": "", "motion": motion, "url": url, "arguments": {}}
                    for key, value in row.items():
                        self.data[key].append(value)

            print(f"Finished scraping domain: {domain}")

        arguments, motions = self.get_arguments(driver)
        self.data["arguments"] = arguments
        self.data["motion"] = motions
        driver.quit()
        self.save()

    def get_arguments(self, driver):
        arguments_data = []
        motions = []

        for url in tqdm(self.data["url"]):
            driver.get(url)

            # Data structure for this URL
            extracted = {
                "arguments": {
                    "pro_arguments": [],
                    "con_arguments": []
                }
            }
            motion_text = ""

            try:
                # Wait for the iframe and switch into it
                iframe = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "iframe.embedded-discussion-dialog__iframe"))
                )
                driver.switch_to.frame(iframe)

                # Extract the motion
                motion_span = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "h2.rich-text.claim-text__content span"))
                )
                motion_text = motion_span.text.strip()

                # Extract arguments
                for stance_key, aria_label in [("pro_arguments", "Pros"), ("con_arguments", "Cons")]:
                    try:
                        ul = WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, f'ul[aria-label="{aria_label}"]'))
                        )
                        h3_elements = ul.find_elements(By.CSS_SELECTOR, "h3.rich-text.claim-text__content")
                        for h3 in h3_elements:
                            try:
                                span = h3.find_element(By.TAG_NAME, "span")
                                if span and span.text.strip():
                                    extracted["arguments"][stance_key].append(span.text.strip())
                            except:
                                continue
                    except:
                        continue

            except Exception as e:
                print(f"[WARN] Failed to extract from: {url}\nReason: {e}")
            finally:
                # Always switch back to main content
                try:
                    driver.switch_to.default_content()
                except:
                    pass

            arguments_data.append(extracted)
            motions.append(motion_text)

        return arguments_data, motions

    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("start-maximized")
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")

        # service = Service(executable_path="/usr/local/bin/chromedriver")
        # driver = webdriver.Chrome(service=service, options=chrome_options)
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        return driver

    def post_process(self, df):
        df = super().post_process(df)
        remap = {
            "Arts": "Arts",
            "Civics & Society": "Society & Culture",
            "Economics & Business": "Economy",
            "History": "History",
            "Human Geography": "Science & Technology",
            "Just for Fun / Icebreakers": "Other",
            "Literature": "Literature",
            "Philosophy": "Philosophy",
            "Pop Culture & Entertainment": "Society & Culture",
            "Religious Studies": "Religion",
            "Science & Technology": "Science & Technology",
            "Social-Emotional Learning": "Society & Culture",
        }
        df["domain"] = df["domain"].apply(lambda domain: remap[domain])
        return df


if __name__ == "__main__":
    scraper = KialoEduScraper(
        url="https://www.kialo-edu.com/debate-topics-and-argumentative-essay-topics",
        output_file="kialo_edu.json",
    )
    scraper.scrape()
