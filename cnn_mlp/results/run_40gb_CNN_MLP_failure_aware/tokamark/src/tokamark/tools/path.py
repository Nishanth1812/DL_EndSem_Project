"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import os
from pathlib import Path


# ======================================================================================================================
# Default directories

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent  # Dynamically find the repo root (no hardcoding!)
PACKAGE_ROOT_DIR = Path(__file__).parent.parent  # Dynamically find the repo root (no hardcoding!)

TASKS_CONFIGS_DIR = os.path.join(PACKAGE_ROOT_DIR, "tasks_configs")
PACKAGE_METADATA_DIR = os.path.join(PACKAGE_ROOT_DIR, "metadata")

RANDOM_SPLIT_SIGNALS_STATS_FILE = os.path.join(PACKAGE_METADATA_DIR, "random_split_signals_stats.yaml")
TEMPORAL_SPLIT_SIGNALS_STATS_FILE = os.path.join(PACKAGE_METADATA_DIR, "temporal_split_signals_stats.yaml")

RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE = os.path.join(PACKAGE_METADATA_DIR, "TokaMark_data_splits.csv")
TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE = os.path.join(PACKAGE_METADATA_DIR, "TokaMark_temporal_data_splits.csv")


# ======================================================================================================================
if __name__ == "__main__":
    # One can print to verify when developing

    print()
    print("Repo root dir:", PROJECT_ROOT_DIR)
    print("Package root dir:", PACKAGE_ROOT_DIR)
    print("Tasks configs dir:", TASKS_CONFIGS_DIR)
    print("Metadata dir:", PACKAGE_METADATA_DIR)
