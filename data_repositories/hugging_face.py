import os
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi
from data_repositories.data_repository import DataRepository

load_dotenv()


class HuggingFaceDataRepository(DataRepository):
    def __init__(self, pages=30, per_page=30):
        self.api = HfApi(token=os.getenv("HF_TOKEN"))
        self.pages = pages
        self.per_page = per_page
        super().__init__("https://huggingface.co/datasets")

    @staticmethod
    def clean_file_ext(path: str) -> str:
        file = Path(path)
        name = file.name.lower()

        # Whitelist of valid scientific extensions
        valid_file_extensions = {
            "hdf5", "parquet", "tfrecord", "json", "jsonl",
            "csv", "tsv", "xml", "yaml", "yml", "nc", "mat",
            "jpg", "jpeg", "png", "tif", "tiff", "xlsx", "xls"
        }
        map_files = {
            "json": "json", "jsonl": "json",
            "yaml": "yaml", "yml": "yaml",
            "jpg": "jpg", "jpeg": "jpg",
            "tif": "tif", "tiff": "tif",
            "xls": "xls", "xlsx": "xls",
            "h5": "hdf5", "hdf5": "hdf5", "geojson": "json",
            "htm": "html", "stp": "step", "step": "step", "h5ad": "hdf5"
        }

        # Case 0: no extension
        if not file.suffix:
            return "noext"

        # Case X: multiple suffixes → invalid
        suffixes = file.suffixes
        if len(suffixes) > 1:
            return "invalidext"

        # Case 1: single extension
        ext = file.suffix[1:]  # strip dot
        ext = ext.lower()

        # --- Normalize TFRecord shard extensions: tfrecord-<i>-of-<n> ---
        # e.g., "tfrecord-00008-of-00032" or "tfrecord-8-of-32"
        if re.fullmatch(r"tfrecord-\d+-of-\d+", ext):
            return "tfrecord"

        # Special rules
        if ext.isdigit():
            return "numericext"
        if len(ext) <= 2:
            return "shortext"

        # Hex-like or weird extensions → invalid
        if ext not in valid_file_extensions:
            if re.fullmatch(r"[a-f0-9]{6,}", ext):
                return "invalidext"
            if re.fullmatch(r"[a-z0-9]{4,}", ext):
                return "invalidext"

        # Strip numeric suffixes like ".csv1"
        base = re.sub(r"[-_]?\d+$", "", ext)
        if len(base) <= 2:
            return "shortext"

        # Apply normalization mapping at the end
        return map_files.get(base, base)

    def _get_page_datasets(self, page_num):
        start = (page_num - 1) * self.per_page
        print(f"Fetching dataset list page {page_num} (size={self.per_page})...")

        try:
            ds = list(self.api.list_datasets(limit=self.per_page, offset=start))
            return [d.id for d in ds]
        except TypeError:
            ds = list(self.api.list_datasets(limit=start + self.per_page))
            return [d.id for d in ds[start : start + self.per_page]]

    def get_repository_metadata(self):
        hf_complete_format_counts, hf_complete_file_sizes, page_start = self.load_latest_checkpoint()
        hf_file_sizes = {}
        hf_format_counts = {}
        hf_validation = {}

        for page in range(page_start, self.pages + 1):
            counter = 0
            dataset_ids = self._get_page_datasets(page)

            if not dataset_ids:
                break

            for idx, dataset_id in enumerate(dataset_ids):
                print(f"Processing dataset {idx + 1}/{len(dataset_ids)}: {dataset_id}")

                start_time = time.time()
                try:
                    info = self.api.dataset_info(
                        dataset_id, revision="main", files_metadata=True
                    )
                    hf_format_counts.clear()
                    hf_file_sizes.clear()
                    for sibling in info.siblings:
                        size = getattr(sibling, "size", None)
                        if size is None:
                            continue

                        path = sibling.rfilename
                        file_ext = self.clean_file_ext(path)
                        # Start writing the data into dictionary objects
                        hf_format_counts[file_ext] = hf_format_counts.get(file_ext, 0) + 1
                        # Initialize list if not present
                        if file_ext not in hf_file_sizes:
                            hf_file_sizes[file_ext] = []

                        # Only keep at most 100 sizes
                        if len(hf_file_sizes[file_ext]) < 100:
                            hf_file_sizes[file_ext].append(int(size))

                        hf_complete_format_counts[file_ext] = hf_complete_format_counts.get(file_ext, 0) + 1

                        # Initialize list if not present
                        if file_ext not in hf_complete_file_sizes:
                            hf_complete_file_sizes[file_ext] = []

                        # Only keep at most 100 sizes
                        if len(hf_complete_file_sizes[file_ext]) < 100:
                            hf_complete_file_sizes[file_ext].append(int(size))
                            # ---------- print summary ONCE per dataset ----------
                    print(f"Format counts list :", hf_format_counts)
                    print(f"File size list:", hf_file_sizes)
                    elapsed = time.time() - start_time
                    result = {
                        "formats": {
                            ext: {
                                "count": hf_format_counts.get(ext, 0),
                                "sizes": hf_file_sizes.get(ext, [])
                            }
                            for ext in set(hf_format_counts) | set(hf_file_sizes)
                        },
                        "execution_time_sec": round(elapsed, 2)
                    }
                    hf_validation[dataset_id] = result

                except Exception as e:
                    print(f"Error processing dataset {dataset_id}: {e}")
                    continue

            self.save_checkpoint(hf_complete_format_counts, hf_complete_file_sizes, page)

        self.data_validation.clear()
        self.format_counts.clear()
        self.file_sizes.clear()

        self.data_validation.update(hf_validation)
        self.format_counts.update(hf_complete_format_counts)
        self.file_sizes.update(hf_complete_file_sizes)




if __name__ == "__main__":
    print("Fetching HuggingFace file format distribution...")
    repo = HuggingFaceDataRepository(pages=30, per_page=30)
    print("\nGenerating plots...")
    repo.plot_format_counts()
    repo.plot_file_sizes()
    # repository data validation and plot execution time distribution is
    # only for new run.
    if not (repo.from_format_counts or repo.from_check_points):
        repo.repository_data_validation()
        repo.plot_execution_time_distribution()
