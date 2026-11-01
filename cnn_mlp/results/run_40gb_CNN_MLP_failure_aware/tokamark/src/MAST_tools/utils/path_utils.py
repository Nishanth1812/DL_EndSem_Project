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

PACKAGE_METADATA_DIR = os.path.join(PACKAGE_ROOT_DIR, "metadata")

# ======================================================================================================================
# Default file paths

DEFAULT_SIGNAL_AVAILABILITY_FILE = os.path.join(PACKAGE_METADATA_DIR, "signal_availability.csv")
DEFAULT_SOURCES_WITH_SIGNALS_FILE = os.path.join(PACKAGE_METADATA_DIR, "dict_sources_with_signals.yaml")

MANUAL_DICT_OUTLIER_METADATA_FILE = os.path.join(PACKAGE_METADATA_DIR, "manual_outlier_metadata.yaml")
RANDOM_SPLIT_OUTLIER_METADATA_FILE = os.path.join(PACKAGE_METADATA_DIR, "random_split_outlier_metadata.yaml")
TEMPORAL_SPLIT_OUTLIER_METADATA_FILE = os.path.join(PACKAGE_METADATA_DIR, "temporal_split_outlier_metadata.yaml")

# ======================================================================================================================
if __name__ == "__main__":
    # One can print to verify when developing

    print()
    print("Repo root:", PROJECT_ROOT_DIR)
    print("Package root dir:", PACKAGE_ROOT_DIR)
    print("Package metadata dir:", PACKAGE_METADATA_DIR)
