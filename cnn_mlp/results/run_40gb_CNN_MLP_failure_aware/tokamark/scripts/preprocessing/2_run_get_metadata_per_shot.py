"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import os
import csv
import argparse
import numpy as np
from typing import Any
from multiprocessing import cpu_count

import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

from tokamark.tools.utils import get_config_from_yaml
from MAST_tools.utils.general_utils import warning_print
from MAST_tools.utils.path_utils import MANUAL_DICT_OUTLIER_METADATA_FILE
from MAST_tools.MAST_dataset import MastDataset
from tokamark.data import initialize_MAST_dataset
from tokamark.data_split import get_train_test_val_shots

from preproc_paths import (
    DEFAULT_SHOTS_STATS_ALL_FILE,
)

# ----------------------------------------------------------------------------------------------------------------------

CSV_HEADER = [
    "shot_idx",
    "shot_id",
    "variable",
    "n_dim_shot",
    "n_nans_shot",
    "mean",
    "variance",
    "min",
    "max",
    "median",
]


# ----------------------------------------------------------------------------------------------------------------------
def identity_collate(batch: Any) -> Any:
    """Identity collate function"""

    return batch


# ======================================================================================================================
class ShotStatsDataset(Dataset):
    """
    Dataset to get shot stats.

    Attributes
    ----------
    dataset : MastDataset
        Target dataset.
    sources_signals : list[str]
        List of source-signal items.

    Methods
    -------
    __len__()
        Return the size of the dataset.
    __getitem__(idx)
        Return shot stats by shot index.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, dataset: MastDataset) -> None:
        """
        Initialize class attributes

        Parameters
        ----------
        dataset : MastDataset
            Input dataset.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors.

        """

        self.dataset = dataset
        self.sources_signals = list(dataset[0].keys())

    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        """
        Return the size of the dataset.

        Returns
        -------
        int
            Length of self.dataset.

        """

        return len(self.dataset)

    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(  # NOSONAR - Ignore cognitive complexity
        self, idx: int
    ) -> dict[str, Any]:
        """
        Return shot stats by shot index.

        Parameters
        ----------
        idx : int
            Shot index.

        Returns
        -------
        dict[str, Any]
            Dictionary with shot statistics.

        """

        shot = self.dataset[idx]
        shot_id = self.dataset.get_shot_id(idx=idx)
        stats: dict[str, Any] = {"shot_idx": idx, "shot_id": shot_id}

        for var in self.sources_signals:
            values = shot[var].get("values", [])

            # Initialize nested dict
            stats[var] = {
                "n_dim_shot": None,
                "n_nans_shot": None,
                "mean": None,
                "variance": None,
                "min": None,
                "max": None,
                "median": None,
            }

            if len(values) > 0:
                mean_sample = np.nanmean(values)
                variance_sample = np.nanvar(values, ddof=0)  # Unbiased variance

                # Handle all-NaN arrays
                if np.isnan(mean_sample):
                    mean_sample = None
                if np.isnan(variance_sample):
                    variance_sample = None

                stats[var]["n_dim_shot"] = values.size if hasattr(values, "size") else None
                stats[var]["n_nans_shot"] = np.isnan(values).sum() if hasattr(values, "size") else None
                stats[var]["mean"] = float(mean_sample) if mean_sample is not None else None
                stats[var]["variance"] = float(variance_sample) if variance_sample is not None else None
                stats[var]["min"] = float(np.nanmin(values)) if values.size else None
                stats[var]["max"] = float(np.nanmax(values)) if values.size else None
                stats[var]["median"] = float(np.nanmedian(values)) if values.size else None

        return stats


# ----------------------------------------------------------------------------------------------------------------------
def compute_mean_std_per_shot_to_csv(
    dataset: MastDataset, csv_path: str, num_workers: int = 32, batch_size: int = 32
) -> None:
    """
    Compute Mean and STD per shot in provided dataset, and save results to target CSV file.

    Parameters
    ----------
    dataset : MastDataset
        Target MAST dataset.
    csv_path : str
        Target CSV path.
    num_workers : int
        Target number of workers.
        Optional. Default: 32.
    batch_size : int
        Target batch size.
        Optional. Default: 32.

    Returns
    -------
    None

    """

    stats_dataset = ShotStatsDataset(dataset=dataset)

    loader = DataLoader(
        dataset=stats_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=identity_collate,
        persistent_workers=num_workers > 0,
    )

    # Ensure directory exists
    os.makedirs(name=os.path.dirname(p=csv_path), exist_ok=True)

    # Overwrite CSV
    with open(csv_path, "w", newline="") as ff:
        writer = csv.writer(ff)

        # Write header
        writer.writerow(CSV_HEADER)

        counter = 0
        for batch in loader:
            for shot_stats in batch:
                counter += 1
                if counter % 10 == 0:
                    print(f"Processed {counter} shots")

                for var in stats_dataset.sources_signals:
                    row = [
                        shot_stats["shot_idx"],
                        shot_stats["shot_id"],
                        var,
                        shot_stats[var].get("n_dim_shot", None),  # Safe fallback
                        shot_stats[var].get("n_nans_shot", None),  # Safe fallback
                        shot_stats[var].get("mean", None),  # Safe fallback
                        shot_stats[var].get("variance", None),  # Safe fallback
                        shot_stats[var].get("min", None),  # Safe fallback
                        shot_stats[var].get("max", None),  # Safe fallback
                        shot_stats[var].get("median", None),  # Safe fallback
                    ]
                    writer.writerow(row)

            # Optional: flush every batch
            ff.flush()
            os.fsync(ff.fileno())


# ======================================================================================================================
if __name__ == "__main__":
    print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method(method="spawn", force=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="./config_get_metadata.yaml",
        help="Path to the YAML file with configuration to get metadata.",
    )
    parser.add_argument(
        "--shots_stats_saving_file_path",
        type=str,
        default=DEFAULT_SHOTS_STATS_ALL_FILE,
        help="Path to the CSV file where shots statistics will be saved.",
    )
    parser.add_argument("--demo_mode", action="store_true", help="Activate demo mode.")
    parser.add_argument(
        "--demo_suffix", type=str, default="_DEMO", help="Suffix used in demo mode when saving results."
    )

    args, _ = parser.parse_known_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Experiment configuration
    # ------------------------------------------------------------------------------------------------------------------

    # Load Task YAML config
    config = get_config_from_yaml(file_path=args.config_file)

    # REMARK: Demo mode could be enforced (e.g., for testing) by uncommenting the following line.
    # args.demo_mode = True

    # If required, override values with settings for demo mode
    if args.demo_mode:
        warning_print("Running in demo mode.")

        filename_demo_suffix = f"{args.demo_suffix}.csv"
        args.shots_stats_saving_file_path = str(args.shots_stats_saving_file_path).replace(".csv", filename_demo_suffix)

        config["get_shots_settings"]["max_index"] = 2
        config["get_shots_settings"]["shuffle"] = True

        config["dataloader_settings"]["batch_size"] = 10
        config["dataloader_settings"]["num_workers"] = 10

    # ------------------------------------------------------------------------------------------------------------------
    # Specific Data Preprocessing for LCFS profiles
    # ------------------------------------------------------------------------------------------------------------------

    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(**config["get_shots_settings"])

    # ------------------------------------------------------------------------------------------------------------------
    # Create unstandardized dataset
    # ------------------------------------------------------------------------------------------------------------------

    print("\nProcessing dataset... with local=", config["local"])

    unstandardized_all_dataset = initialize_MAST_dataset(
        config_task=config["task_configuration"],
        shots_list=train_shots_ + test_shots_ + val_shots_,
        local_flag=config["local"],
        use_std_scaling=False,  # <- To get unstandardized dataset
        use_nan_filling=False,
        remove_outliers=True,  # <- To remove manually found outliers
        outlier_metadata_file=MANUAL_DICT_OUTLIER_METADATA_FILE,
        remove_bad_efit_rating=True,
        store_manager_settings=config["store_manager_settings"],
    )

    print("Computing Mean/STD values for all shots in dataset...")

    compute_mean_std_per_shot_to_csv(
        dataset=unstandardized_all_dataset, csv_path=args.shots_stats_saving_file_path, **config["dataloader_settings"]
    )

    print(f"Mean/STD results saved at {args.shots_stats_saving_file_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # Ending
    # ------------------------------------------------------------------------------------------------------------------

    print("\nAll computations done.")
