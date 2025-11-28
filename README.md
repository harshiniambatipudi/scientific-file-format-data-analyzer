# Scientific File Format Data Analyzer

Analyzes file format distributions across major scientific data repositories: Data.gov, ESS-DIVE, HuggingFace, and IEEE Dataport.  
This project was developed under the guidance of **Dr. Suren Byna**, Professor in the Department of Computer Science and Engineering (CSE) at The Ohio State University (OSU).

## Features

- Fetches file format counts from multiple data repositories
- Tracks file sizes where available
- Generates distribution plots
- Checkpointing for resumable long-running fetches
- Execution time tracking

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install Playwright browsers (required for Data.gov and IEEE)
uv run playwright install chromium
```

## Configuration

Create a `.env` file in the project root:

```bash
# Required for ESS-DIVE
ESS_DIVE_AUTH_TOKEN=your_token_here

# Required for HuggingFace
HF_TOKEN=your_token_here
```

**Getting API Tokens:**
- ESS-DIVE: Visit [ESS-DIVE API](https://api.ess-dive.lbl.gov) and register
- HuggingFace: Get token from [HuggingFace Settings](https://huggingface.co/settings/tokens)

## Usage

Run individual data sources:

```bash
# Data.gov (no token required)
uv run python -m data_repositories.data_gov

# ESS-DIVE (requires token)
uv run python -m data_repositories.ess_dive

# HuggingFace (requires token)
uv run python -m data_repositories.hugging_face

# IEEE Dataport (no token required)
uv run python -m data_repositories.ieee_dataport
```

## Output

The analyzer generates:

- **`format_counts/`** - JSON files with raw format counts and file sizes
- **`checkpoints/`** - Checkpoint files for resuming interrupted fetches
- **Plots** - Interactive matplotlib visualizations showing:
  - File format distribution (top 9 + others)
  - File size distribution (box plots, where available)
  - Execution time distribution (per-dataset processing time)


## Notes

- First run fetches fresh data (can take hours for large repositories)
- Subsequent runs load from cached JSON files
- Delete `format_counts/*.json` to force a fresh fetch
- Checkpoints allow resuming if fetch is interrupted
- Raw data is saved to disk; normalization happens in-memory only

 
