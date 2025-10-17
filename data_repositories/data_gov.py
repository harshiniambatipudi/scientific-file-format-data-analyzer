import re
from playwright.sync_api import sync_playwright
from data_repositories.data_repository import DataRepository


class DataGovDataRepository(DataRepository):
    def __init__(self):
        super().__init__("https://catalog.data.gov")

    def get_repository_metadata(self):
        format_counts = {}

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            page.goto(f"{self.base_url}/dataset/?_res_format_limit=0")
            page.wait_for_load_state()

            format_items = page.query_selector_all(
                "nav[aria-label='Formats'] li.nav-item"
            )
            for item in format_items:
                format_label = item.query_selector(".item-label").inner_text()
                count = int(item.query_selector(".item-count").inner_text())
                format_cleaned = re.sub(r"[^a-z0-9]", "", format_label.strip().lower())
                format_counts[format_cleaned] = count

            browser.close()
        self.format_counts.clear()
        self.format_counts.update(format_counts)
        self.file_sizes = {}


if __name__ == "__main__":
    print("Fetching Data.gov file format distribution...")
    repo = DataGovDataRepository()
    print("\nGenerating plot...")
    repo.plot_format_counts()
