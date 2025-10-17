import re
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from data_repositories.data_repository import DataRepository


class IEEEDataPortDataRepository(DataRepository):
    def __init__(self):
        super().__init__("https://ieee-dataport.org")

    @staticmethod
    def _get_dataset_urls(page):
        dataset_links = page.query_selector_all("a[href^='/documents/']")
        return [
            f"https://ieee-dataport.org{link.get_attribute('href')}"
            for link in dataset_links
        ]

    @staticmethod
    def _get_formats(page):
        format_link = page.query_selector("a[href^='/data-formats/']")
        if not format_link:
            return []

        text = format_link.inner_text()
        print("original format links :", text)
        text = text.replace(";", " ")
        text = re.sub(r'[\[\]*/|,()]+', ' ', text)
        file_formats = text.split()
        print("after split  :", file_formats)
        return [
            clean
            for file_format in file_formats
            if (clean := re.sub(r"[^a-z0-9]", "", file_format.lower()))
        ]

    def get_repository_metadata(self):
        format_counts, file_sizes, start_page = self.load_latest_checkpoint()
        format_counts_dataset = {}
        ieee_validation = {}
       # max_pages = 34
        max_pages = 39
        # If the length of format_counts is zero then there is no check point data
        # ,and it is a fresh new start.
        if len(format_counts) == 0:
            start_page = 0

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            processed_datasets = sum(format_counts.values()) if format_counts else 0
            for current_page in range(start_page, max_pages + 1):
                start_time = time.time()
                try:
                    page.goto(f"{self.base_url}/datasets?page={current_page}")
                    page.wait_for_load_state()

                    dataset_urls = self._get_dataset_urls(page)
                    print(f"Page {current_page}: {len(dataset_urls)} datasets")

                    for idx, url in enumerate(dataset_urls):
                        format_counts_dataset.clear()
                        dataset_name = (
                            url.split("/documents/", 1)[1]
                            if "/documents/" in url
                            else url
                        )
                        print("*********************************************")
                        print(
                            f"Processing dataset {idx + 1}/{len(dataset_urls)}: {dataset_name}"
                        )

                        page.goto(url)
                        page.wait_for_load_state()

                        file_formats = self._get_formats(page)
                        for file_format in file_formats:
                            format_counts[file_format] = (
                                format_counts.get(file_format, 0) + 1
                            )
                            # need to get the complete metadata for repository
                            format_counts_dataset[file_format] = format_counts_dataset.get(file_format, 0) + 1

                        end_time = time.time()
                        elapsed = end_time - start_time
                        result = {
                            "formats": {
                                ext: {
                                    "count": format_counts_dataset.get(ext, 0),
                                    "sizes": []
                                }
                                for ext in set(format_counts_dataset)
                            },
                            "execution_time_sec": round(elapsed, 2)
                        }
                        ieee_validation[dataset_name] = result
                        print("File format counts:")
                        for file_format, count in sorted(format_counts_dataset.items()):
                            print(f"\t{file_format}: {count}")
                        print()

                        time.sleep(1.5)

                    self.save_checkpoint(format_counts, file_sizes, current_page)
                    time.sleep(5)

                except PlaywrightTimeoutError:
                    print(f"Timeout on page {current_page}")
                    continue
                except Exception as e:
                    print(f"Error on page {current_page}: {e}")
                    continue

            browser.close()
        self.data_validation.clear()
        self.format_counts.clear()
        self.file_sizes = {}

        self.data_validation.update(ieee_validation)
        self.format_counts.update(format_counts)


if __name__ == "__main__":
    print("Fetching IEEE Dataport file format distribution...")
    repo = IEEEDataPortDataRepository()
    print("\nGenerating plots...")
    repo.plot_format_counts()
    # repository data validation and plot execution time distribution is
    # only for new run.
    if not (repo.from_format_counts or repo.from_check_points):
        repo.repository_data_validation()
        repo.plot_execution_time_distribution()

