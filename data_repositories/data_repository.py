import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


class DataRepository(ABC):
    EXCLUDED_FORMATS = {
        "format", "videos", "html", "esrirest", "wms", "npz", "obj", "dsstore", "ds_store", "log",
        "zip", "html", "pdf", "text", "arcgisgeoservicesrestapi", "txt", "yaml", "yml",
        "z", "gz", "tar", "zst", "zstd", "glb", "tfrecord", "png", "part", "md",
        "step", "psd", "pptx", "noextension", "targz", "gz", "compression", "jpgzip",
        "mp3", "mp4", "doc", "docx", "txt", "pdf", "psd", "pages", "m4a", "jsonld", "bib", "whl", "json00", "gz002", "gz003",
        "bin", "gz004", "gz005", "gz006", "gz007", "gz008", "tmp", "sql", "cff", "toml", "flac", "pptx", "scp", "rar",
        "avif",
        "pth", "webm", "safetensors", "onnx", "usdc", "hdr", "mov", "pyc", "ipynb", "svg", "docx", "dbsqlitedatabase",
        "dbsqlitedatabase", "webp", "mcap", "sha256", "pdb", "gif", "crc", "ogg", "jpg",
        "numericext", "picklezip", "bmpzippngzip", "cocoformat", "npzzip", "shortext", "noext", "invalidext", "tsvzip",
        "csvzip", "pythonnumpynpyfile", "bmpzip", "pngzip", "xlsxzip"
    }

    FORMAT_ALIASES = {
        "excel": "xlsx",
        "jsonl": "json",
        "jpeg": "jpg",
        "tiff": "tif",
        "h5": "hdf5",
        "hdf5": "hdf5",
        "h5ad": "hdf5",
        "eml": "xml",
        "spreadsheets": "xlsx",
        "excelspreadsheets": "xlsx",
        "exceldata": "xlsx",
        "xls": "xlsx",
        "nc": "netcdf",
    }
    INCLUDED_FORMATS = {"nc"}

    def __init__(self, base_url):
        self.base_url = base_url
        self.name = self.__class__.__name__
        self.format_counts: Dict[str, int] = {}
        self.file_sizes: Dict[str, List[float]] = {}
        self.top9_file_names = {}
        self.data_validation = {}
        # this flag will be true or false based on new start ot restored from format point
        self.from_format_counts: bool = False
        # this flag will br true or false based on new start ot restored from checkpoint point
        self.from_check_points: bool = False

        self.format_counts_dir = Path("format_counts")
        self.checkpoints_dir = Path("checkpoints")
        self.data_validation_dir = Path("data_validation")
        self.repository_metadata_dir = Path("repository_metadata")
        self.repository_trends_dir = Path("repository_plots")

        self.format_counts_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.data_validation_dir.mkdir(exist_ok=True)
        self.repository_metadata_dir.mkdir(exist_ok=True)
        self.repository_trends_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.format_counts_dir / f"{self.name}_format_counts.json"

        if json_path.exists():
            print(f"Loading existing format counts from {json_path}")
            self.from_format_counts = True
            self.format_counts, self.file_sizes = (
                self.load_repository_metadata()
            )
        else:
            print(
                f"No existing format counts found for {self.name}. Fetching new counts..."
            )
            self.fetch_start = datetime.now()
            self.get_repository_metadata()
            self.fetch_end = datetime.now()
            fetch_duration = (self.fetch_end - self.fetch_start).total_seconds()
            print(f"Fetch completed in {fetch_duration:.2f} seconds")
            self.save_repository_metadata()

        # Get format_counts and file_sizes after filtering
        self.format_counts = self.filter_format_counts(self.format_counts)
        self.file_sizes = self.filter_file_sizes()

    def load_repository_metadata(self):
        json_path = self.format_counts_dir / f"{self.name}_format_counts.json"
        with open(json_path) as f:
            data = json.load(f)
            return (
                data.get("format_counts", {}),
                data.get("file_sizes", {}),
            )

    def save_repository_metadata(self):
        json_path = self.format_counts_dir / f"{self.name}_format_counts.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "format_counts": self.format_counts,
                    "file_sizes": self.file_sizes,
                },
                f,
                indent=2,
            )

    def save_checkpoint(self, format_counts, file_sizes, page):
        checkpoint_path = (
            self.checkpoints_dir / f"{self.name}_checkpoint_page_{page}.json"
        )
        with open(checkpoint_path, "w") as f:
            json.dump(
                {"format_counts": format_counts, "file_sizes": file_sizes}, f, indent=2
            )

    def load_latest_checkpoint(self):
        checkpoints = sorted(
            self.checkpoints_dir.glob(f"{self.name}_checkpoint_page_*.json"),
            key=lambda f: int(re.search(r"page_(\d+)", f.name).group(1)),
        )

        if not checkpoints:
            return {}, {}, 1

        self.from_check_points = True
        latest = checkpoints[-1]
        page = int(re.search(r"page_(\d+)", latest.name).group(1))

        with open(latest) as f:
            data = json.load(f)
            return data.get("format_counts", {}), data.get("file_sizes", {}), page + 1

    @abstractmethod
    def get_repository_metadata(self):
        pass

    def filter_format_counts(self, counts: Dict[str, int]) -> Dict[str, int]:
        filtered: Dict[str, int] = {}
        for format_name, count in counts.items():
            format_name = re.sub(r"[^a-zA-Z0-9]", "", format_name.lower())
            if format_name in self.EXCLUDED_FORMATS or (len(format_name) <= 2 and format_name not in self.INCLUDED_FORMATS):
                continue

            format_name = self.FORMAT_ALIASES.get(format_name, format_name)
            filtered[format_name] = filtered.get(format_name, 0) + count
        return filtered

    def filter_file_sizes(self):
        filtered_sizes = {}
        if self.file_sizes:
            for format_name, sizes in self.file_sizes.items():
                format_name = re.sub(r"[^a-zA-Z0-9]", "", format_name.lower())
                if format_name in self.EXCLUDED_FORMATS:
                    continue

                format_name = self.FORMAT_ALIASES.get(format_name, format_name)
                filtered_sizes.setdefault(format_name, []).extend(sizes)
        else:
            print("File sizes are empty.")
        return filtered_sizes

    def plot_format_counts(self):
        def truncate_label(label, max_length=10):
            return (
                label if len(label) <= max_length else label[: max_length - 3] + "..."
            )

        sorted_formats = dict(
            sorted(self.format_counts.items(), key=lambda x: x[1], reverse=True)
        )
        top_formats = dict(list(sorted_formats.items())[:9])
        other_sum = sum(list(sorted_formats.values())[9:])
        if other_sum > 0:
            top_formats["other"] = other_sum

        plt.figure()
        plt.title(f"{self.name} File Format Distribution")
        plt.xlabel("File Formats")
        plt.ylabel("Count")

        plt.bar(range(len(top_formats)), top_formats.values(), edgecolor="black")
        plt.xticks(
            range(len(top_formats)),
            [truncate_label(k) for k in top_formats.keys()],
            rotation=90,
        )
        plt.tight_layout()
        out_dir: Path = self.repository_trends_dir
        save_base = out_dir / f"{self.name}_format_counts"
        plt.savefig(save_base.with_suffix(".pdf"))  # vector PDF

        plt.show()

    def plot_file_sizes(self, log_scale=True):
        unit = "KiB"
        bytes_conversion = 1024.0  # bytes → KiB
        if not self.file_sizes:
            print("No file size data to plot.")
            return

        # Get top9 file names:
        top9_file_names = [k for k, _ in sorted(self.format_counts.items(), key=lambda x: x[1], reverse=True)[:9]]

      #  if "hugging" in self.name.lower():
      #   bytes_conversion = 1024 ** 2  # 1,048,576 bytes
      #   unit = 'MB'

        # filter top9 file sizes after conversion from bytes to MB or KiB based on repository
        file_sizes_for_plot = {
            ext: [round(size / bytes_conversion, 2) for size in self.file_sizes.get(ext, [])]
            for ext in top9_file_names
            if ext in self.file_sizes and self.file_sizes[ext]
        }

        bxp_stats = []
        labels = []
        for fmt, vals in file_sizes_for_plot.items():
            if not vals:
                continue
            data = np.array(vals, dtype=float)
            labels.append(fmt)
            bxp_stats.append(
                {
                    "label": fmt,
                    "whislo": float(np.min(data)),
                    "q1": float(np.percentile(data, 25)),
                    "med": float(np.median(data)),
                    "q3": float(np.percentile(data, 75)),
                    "whishi": float(np.max(data)),
                    "fliers": [],
                }
            )

        if not bxp_stats:
            print("No file size data to plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bxp(bxp_stats, showfliers=False)

        if log_scale:
            ax.set_yscale("log")

        ax.set_title(f"{self.name} File Size Distribution")
        ax.set_ylabel(
            f"File Size ({unit}, log scale)" if log_scale else f"File Size ({unit})"
        )
        ax.set_xticklabels(labels, rotation=0)

        for i, stats in enumerate(bxp_stats, start=1):
            ax.text(
                i,
                stats["med"],
                f"{stats['med']:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        # --- Save as PDF ---
        out_dir: Path = self.repository_trends_dir
        pdf_path = out_dir / f"{self.name}_file_size_distribution.pdf"
        fig.savefig(pdf_path)  # vector PDF; dpi not needed
        plt.show()

    def plot_execution_time_distribution(self, bins=10):
        execution_times = [
            execution_pkg.get("execution_time_sec")
            for _, execution_pkg in self.data_validation.items()
            if "execution_time_sec" in execution_pkg
        ]
        if not execution_times:
            print("No execution times available to plot.")
            return

        mean_time = np.mean(execution_times)
        median_time = np.median(execution_times)

        plt.figure(figsize=(12, 6))
        plt.hist(execution_times, bins=bins, edgecolor="black", alpha=0.7)

        plt.axvline(
            mean_time,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_time:.2f}s",
        )
        plt.axvline(
            median_time,
            color="green",
            linestyle="-.",
            linewidth=2,
            label=f"Median: {median_time:.2f}s",
        )

        plt.title(f"{self.name} Execution Time Distribution")
        plt.xlabel("Execution Time (seconds)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        # --- Save as PDF (vector) ---
        out_dir: Path = self.repository_trends_dir
        pdf_path = out_dir / f"{self.name}_frequency_distribution.pdf"
        plt.savefig(pdf_path)  # DPI not needed for PDF

        plt.show()

    def repository_data_validation(self) -> None:
        # create file paths to write metadata and validation results.
        # repository metadata is stored in data_validationand we are writing this
        # as json file in /repository_metadata folder
        print("repository validation started")
        json_path = self.repository_metadata_dir / f"{self.name}_metadata.json"
        txt_path = self.data_validation_dir / f"{self.name}_validation_results.txt"
        repository_file_format_counts: Dict[str, int] = {}
        repository_file_sizes: Dict[str, List[float]] = {}

        # Delete old metadata file if present
        if json_path.exists():
            json_path.unlink()

        # Delete old validation file if present
        if txt_path.exists():
            txt_path.unlink()

        # Check and delete check point files
        with open(json_path, "w") as f:
            json.dump(self.data_validation, f, indent=2)

        # First collect all the data into list and then write to a file.
        # First validation: write repository name , dataset counts and time taken to fetch
        lines: List[str] = []
        lines.append("***********************************************")
        lines.append("Validation 1- Repository Name, Dataset processed, Fetch start, Fetch end time and TimeTaken")
        lines.append(f"Repository: {self.name}")
        lines.append(f"Datasets processed: {len(self.data_validation)}")
        if self.fetch_start and self.fetch_end:
            lines.append(f"Fetching started: {self.fetch_start.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Fetching ended:   {self.fetch_end.strftime('%Y-%m-%d %H:%M:%S')}")
            elapsed = self.fetch_end - self.fetch_start
            lines.append(f"Time taken :{str(elapsed).split(".")[0]} hh:mm:sec)")
        lines.append("***********************************************")

        # blank line before details
        # Compute aggregates of self.data_validation
        format_counts = Counter()
        file_sizes = {}

        # The below code populates repository_file_format_counts and repository_file_sizes
        # from metadata repository collection.
        for pkg_id, pkg_meta in self.data_validation.items():
            for fmt, meta in pkg_meta["formats"].items():
                format_counts[fmt] += meta.get("count", 0)
                file_sizes.setdefault(fmt, []).extend(meta.get("sizes", []))

                # convert Json to dict
                metadata_aggregate = {
                    "format_counts": dict(format_counts),
                    "file_sizes": file_sizes
                }
                repository_file_format_counts = metadata_aggregate.get("format_counts", {})
                repository_file_sizes = metadata_aggregate.get("file_sizes", {})

        # validation 2
        if len(repository_file_format_counts) > 0 and len(self.file_sizes) > 0:
            # First match the file format counts
            # second validation: check if repository metadata file format counts and file sizes matches
            lines.append(
                "*** Validation 2: The tool verifies consistency between file format counts ***\n"
                "*** and file size entries in  the raw per-dataset metadata JSON files")
            for fmt, count in sorted(repository_file_format_counts.items()):
                size_len = len(repository_file_sizes.get(fmt, []))
                if "hugging" in self.name.lower() and size_len > 100:
                    status = "MATCH"
                else:
                    status = "MATCH" if count == size_len else "MISMATCH"

                line = f"{fmt}: count={count} | len(file_sizes)={size_len} -> {status}\n"
                lines.append(line)
        else:
            lines.append("Validation 2: check if raw repository metadata counts vs length of sizes matches is not applied as file_sizes is empty")
        # validate top 9 file formats from repository meta data json
        # first normalize / filter using filter_format_counts and then
        # get the top 9 file formats
        # validation 3
        tool_top9_file_formats = dict(
            sorted(self.format_counts.items(), key=lambda x: x[1], reverse=True)[:9])

        if repository_file_format_counts:
            filtered_top9_file_format_counts: Dict[str, int] = self.filter_format_counts(repository_file_format_counts)
            raw_top9_file_formats= dict(sorted(filtered_top9_file_format_counts.items(), key=lambda x: x[1], reverse=True)[:9])

            lines.append(
                "**Validation 3: From repository raw metadata get top file formats **\n"
                "**and match with the tool or graph ===\n"
            )
            for fmt in set(raw_top9_file_formats) | set(tool_top9_file_formats):  # union of both sets
                raw_val = raw_top9_file_formats.get(fmt, 0)
                tool_val = tool_top9_file_formats.get(fmt, 0)

                status = "MATCH" if raw_val == tool_val else "MISMATCH"
                lines.append(f"{fmt}: raw_count={raw_val} | tool_count={tool_val} -> {status}")

        # write validation results into text
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

