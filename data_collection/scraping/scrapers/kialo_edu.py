from .base_scraper import BaseScraper
from schema import *
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
    @property
    def url(self) -> str:
        return "https://www.kialo-edu.com/debate-topics-and-argumentative-essay-topics"
    
    @property
    def name(self) -> str:
        return "kialo_edu"
    
    def scrape(self):
        # Use Selenium to click the drop down menu and select one of the following domains
        domain_names = [
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
        driver = self._setup_driver()
        driver.get(self.url)

        # Allow time for elements to load
        time.sleep(5)

        pages = []
        for domain_name in domain_names:

            domain = Domain(id=self._get_id(domain_name), name=domain_name)
            self._add_entity(domain)

            # Step 1: Click the dropdown button to reveal the options
            dropdown_button = driver.find_element(
                By.CSS_SELECTOR, "div.dropdown__button--generic"
            )
            dropdown_button.click()

            # Step 2: Wait for the dropdown to expand and the category button to be visible
            category_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, f"//button[span[text()='{domain.name}']]")
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
            discussion_cards = driver.find_elements(By.CSS_SELECTOR, "div.line-discussion-card__container")

            for i in range(len(discussion_cards)):
                cards = driver.find_elements(By.CSS_SELECTOR, "div.line-discussion-card__container")
                card = cards[i]
                # title_element = topic.find_element(
                #     By.CSS_SELECTOR, "div.discussion-card-title__title-wrapper"
                # )
                # motion = title_element.text.strip() if title_element else None

                link_element = card.find_element(By.CSS_SELECTOR, "a.discussion-card-title--link")
                if not link_element:
                    continue

                url = link_element.get_attribute("href")
                pages.append({"url": url, "domain_id": domain.id})

            print(f"Finished scraping domain: {domain.name}")

        for page in tqdm(pages):
            self._parse(driver, page["url"], page["domain_id"])

        driver.quit()
        self.print_entities()
        self._save()

    def _parse(self, driver: webdriver.Chrome, url: str, domain_id: str) -> None:
        driver.get(url)

        try:
            # Wait for the iframe and switch into it
            iframe = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "iframe.embedded-discussion-dialog__iframe"))
            )
            driver.switch_to.frame(iframe)

            # Extract the issue
            issue_span = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h2.rich-text.claim-text__content span"))
            )
            issue_text = issue_span.text.strip()
            if not issue_text:
                return 
            
            issue = Issue(
                id=self._get_id(issue_text),
                text=issue_text,
                source=self.name,
                url=url,
                timestamp=self._get_timestamp()
            )
            self._add_entity(issue)
            self._add_entity(InDomain(source=issue.id, target=domain_id))

            # Extract arguments
            for aria_label in ("Pros", "Cons"):
                try:
                    ul = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, f'ul[aria-label="{aria_label}"]'))
                    )
                    h3_elements = ul.find_elements(By.CSS_SELECTOR, "h3.rich-text.claim-text__content")
                    for h3 in h3_elements:
                        try:
                            span = h3.find_element(By.TAG_NAME, "span")
                            if span and span.text.strip():
                                argument_text = span.text.strip()
                                argument = Argument(
                                    id=self._get_id(argument_text),
                                    text=argument_text,
                                    source=self.name,
                                    url=url,
                                    timestamp=self._get_timestamp()
                                )
                                self._add_entity(argument)
                                if aria_label == "Pros":
                                    self._add_entity(Supports(source=argument.id, target=issue.id))
                                else:
                                    self._add_entity(Attacks(source=argument.id, target=issue.id))
                        except:
                            continue
                except:
                    continue

        except Exception as e:
            print(f"Failed to extract from: {url}\nReason: {e}")
        finally:
            # Always switch back to main content
            try:
                driver.switch_to.default_content()
            except:
                pass

    def _setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("start-maximized")
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")

        # service = Service(executable_path="/usr/local/bin/chromedriver")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
