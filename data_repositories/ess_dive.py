import os
import time
import requests
from dotenv import load_dotenv
from data_repositories.data_repository import DataRepository

load_dotenv()


class ESSDiveDataRepository(DataRepository):
    def __init__(self):
        load_dotenv()
        token = os.getenv("ESS_DIVE_AUTH_TOKEN")
        if not token:
            raise RuntimeError("Set ESS_DIVE_TOKEN in token.env")
        self.headers = {"Authorization": f"bearer {os.getenv('ESS_DIVE_AUTH_TOKEN')}"}
        super().__init__("https://api.ess-dive.lbl.gov")

    def get_repository_metadata(self):
        ess_complete_format_counts, ess_complete_file_sizes, row_start = self.load_latest_checkpoint()
        ess_file_sizes = {}
        ess_format_counts = {}
        # ess_validation is used to validate data after fetching the entire metadata
        # to make sure we got the corrct file formats and sizes
        ess_validation = {}
        page_size = 30 # this is always fixed -The number of datasets to return per request.
        total_packages = row_start-1  # rowStart: The row number to start on. Use this for paging results
        max_datasets = 10

        while True:
            params = {"isPublic": True, "pageSize": page_size, "rowStart": row_start}
            response = requests.get(
                f"{self.base_url}/packages", headers=self.headers, params=params
            ).json()
            packages = response["result"]
            num_packages = len(packages)

            print(
                f"Retrieved {num_packages} packages (page starting at row {row_start})"
            )
            processed_in_this_page = 1
            for idx, package in enumerate(packages):
                print("*********************************************")
                package_id = package["id"]
                print(f"Processing package {idx + 1}/{num_packages}: {package_id}")
                start_time = time.time()
                try:
                    response = requests.get(
                        f"{self.base_url}/packages/{package_id}", headers=self.headers
                    ).json()

                    if (
                        "dataset" not in response
                        or "distribution" not in response["dataset"]
                    ):
                        continue
                    # clear format_Counts and File sizes for each run
                    ess_format_counts.clear()
                    ess_file_sizes.clear()
                    for file in response["dataset"]["distribution"]:
                        if "name" in file and "." in file["name"]:
                            extension = file["name"].split(".")[-1].lower()
                            size = file.get("contentSize")

                            if size is None:
                                continue
                            # Start writing the data into dictionary objects
                            ess_format_counts[extension] = (
                                ess_format_counts.get(extension, 0) + 1
                            )
                            ess_file_sizes.setdefault(extension, []).append(float(size))
                            ess_complete_format_counts[extension] = ess_complete_format_counts.get(extension, 0) + 1
                            ess_complete_file_sizes.setdefault(extension, []).append(round(float(size)))

                            # capture end time  and elapsed for metric and store it per dataset raw json file.
                            end_time = time.time()
                            elapsed = end_time - start_time
                            # validation store file_Sizes and file formats for the packege_id
                            result = {
                                "formats": {
                                    ext: {
                                        "count": ess_format_counts.get(ext, 0),
                                        "sizes": ess_file_sizes.get(ext, [])
                                    }
                                    for ext in set(ess_format_counts) | set(ess_file_sizes)
                                },
                                "execution_time_sec": round(elapsed, 2)
                            }
                            ess_validation[package_id] = result
                    total_packages += 1
                    print("format counts:", ess_format_counts)
                    print("file sizes:", ess_file_sizes)
                    print(f"Processing completed for package")

                except Exception as e:
                    print(f"Error processing package {package_id}: {e}")
                    continue

            if num_packages < page_size:
                break

            row_start += num_packages
            self.save_checkpoint(ess_complete_format_counts, ess_complete_file_sizes, total_packages)
            time.sleep(1)

        self.data_validation.clear()
        self.format_counts.clear()
        self.file_sizes.clear()

        self.data_validation.update(ess_validation)
        self.format_counts.update(ess_complete_format_counts)
        self.file_sizes.update(ess_complete_file_sizes)


if __name__ == "__main__":
    print("Fetching ESS-DIVE file format distribution...")
    repo = ESSDiveDataRepository()
    print("\nGenerating plots...")
    repo.plot_format_counts()
    repo.plot_file_sizes()
    # repository data validation and plot execution time distribution is
    # only for new run.
    if not (repo.from_format_counts or repo.from_check_points):
        repo.repository_data_validation()
        repo.plot_execution_time_distribution()
