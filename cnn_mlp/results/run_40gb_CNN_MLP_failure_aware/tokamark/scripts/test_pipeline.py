"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import time
import argparse
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Sequence, Optional
from multiprocessing import cpu_count

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate  # noqa - Ignore "access to protected method" warning

from MAST_tools.utils.general_utils import warning_print
from tokamark.tools.utils import get_device, get_config_from_yaml
from tokamark.data_split import get_train_test_val_shots
from tokamark.tasks import get_task_metadata, get_task_config
from tokamark.data import initialize_MAST_dataset, initialize_TokaMark_dataset

# ----------------------------------------------------------------------------------------------------------------------

CONFIG_FILES_DIR = Path("./config_files")
BASE_CONFIG_TEST_PIPELINE_FILE = CONFIG_FILES_DIR / "config_test_pipeline.yaml"
BASE_CONFIG_TEST_TASK_FILE = CONFIG_FILES_DIR / "config_test_task.yaml"

# ------------------------------------------------------------------------------------------------------------------
# Preliminaries

start = time.perf_counter()

device = get_device()


# print(f"Using device: {device}\n")


# ======================================================================================================================
class ModelSpecificTransform:  # TEMPLATE
    """
    Model specific transform.

    Attributes
    ----------
    verbose : bool
        If True, activate verbose mode.

    Methods
    -------
    __call__(shot)
        Call method.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, verbose=False) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        verbose : bool
            If True, activate verbose mode.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors, as this is a callable class.

        """

        self.verbose = verbose

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Mapping[str, Any]) -> dict[str, Any]:
        """
        Call method.

        Parameters
        ----------
        shot : Dict[str, Any]
            Target shot.

        Returns
        -------
        dict[str, Any]
            Dictionary with "x" and "y" keys and values from `shot["input"] + shot["actuator"]` and `shot["output"]`
            items, respectively.

        """

        return {
            "x": (
                [data["values"] for var, data in shot["input"].items()]
                + [data["values"] for var, data in shot["actuator"].items()]
            ),
            "y": [data["values"] for var, data in shot["output"].items()],
        }


# ----------------------------------------------------------------------------------------------------------------------
def model_collate_fn(batch: Sequence, verbose: bool = False) -> Optional[Any]:
    """
    Model collate function.

    Parameters
    ----------
    batch : Sequence
        Input batch.
    verbose : bool
        If True, activate verbose mode.
        Optional. Default: False.

    Returns
    -------
    Optional[Any]
        Default collate function evaluated on flattened batch if feasible, None otherwise.

    """

    flattened_batch = [(item["shot_id"], item["window_index"], item["x"], item["y"]) for item in batch]

    if verbose:
        print(f"\nNumber of shots in a batch = {len(batch)}; number of samples (segments) = {len(flattened_batch)}")
        if len(flattened_batch) == 0:
            print("batch is None")

    return default_collate(flattened_batch) if (len(flattened_batch) > 0) else None


# ======================================================================================================================
if __name__ == "__main__":
    print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method(method="spawn", force=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "test_task",
            "task_1-1",
            "task_1-2",
            "task_1-3",
            "task_2-1",
            "task_2-2",
            "task_2-3",
            "task_3-1",
            "task_3-2",
            "task_3-3",
            "task_4-1",
            "task_4-2",
            "task_4-3",
            "task_4-4",
            "task_4-5",
        ],
        default="test_task",
        help="The name of the task available in the benchmark.",
    )
    parser.add_argument(
        "--pipeline_config_file_path",
        type=str,
        default=BASE_CONFIG_TEST_PIPELINE_FILE,
        help="Path to the model YAML config file.",
    )
    parser.add_argument("--demo_mode", action="store_true", help="Activate demo mode.")
    parser.add_argument(
        "--demo_suffix", type=str, default="_DEMO", help="Suffix used in demo mode when saving results."
    )

    args, _ = parser.parse_known_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Experiment configuration
    # ------------------------------------------------------------------------------------------------------------------

    # Load the config YAML file
    pipeline_config = get_config_from_yaml(file_path=args.pipeline_config_file_path)

    # REMARK: Demo mode could be enforced (e.g., for testing) by uncommenting the following line.
    # args.demo_mode = True

    if args.demo_mode:
        warning_print("Running in demo mode.")

        pipeline_config["get_shots_settings"]["max_index"] = 2
        pipeline_config["get_shots_settings"]["shuffle"] = True

        pipeline_config["dataloader_settings"]["batch_size"] = 4
        pipeline_config["dataloader_settings"]["num_workers"] = 0

        pipeline_config["paths"]["output_dir"] += args.demo_suffix

    if args.task == "test_task":
        # Instead of loading benchmark task, here we load the configuration for a test task
        config_task = get_config_from_yaml(file_path=BASE_CONFIG_TEST_TASK_FILE)
    else:
        # Otherwise, use the provided benchmark task
        config_task = get_task_config(task_name=args.task)

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize datasets and metadata
    # ------------------------------------------------------------------------------------------------------------------

    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(**pipeline_config["get_shots_settings"])

    local_flag = pipeline_config["local"]

    train_MAST_dataset = initialize_MAST_dataset(
        config_task=config_task,
        shots_list=train_shots_,
        local_flag=local_flag,
        **pipeline_config["mast_dataset_init_settings"],
        store_manager_settings=pipeline_config["store_manager_settings"],
        verbose=True,
    )

    val_MAST_dataset = initialize_MAST_dataset(
        config_task=config_task,
        shots_list=val_shots_,
        local_flag=local_flag,
        **pipeline_config["mast_dataset_init_settings"],
        store_manager_settings=pipeline_config["store_manager_settings"],
        verbose=True,
    )

    test_MAST_dataset = initialize_MAST_dataset(
        config_task=config_task,
        shots_list=test_shots_,
        local_flag=local_flag,
        **pipeline_config["mast_dataset_init_settings"],
        store_manager_settings=pipeline_config["store_manager_settings"],
        verbose=True,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize task-specific metadata
    # ------------------------------------------------------------------------------------------------------------------

    dict_task_metadata = get_task_metadata(config_task=config_task, verbose=False)

    # ------------------------------------------------------------------------------------------------------------------
    # EXAMPLE WITH MODEL SPECIFIC PIPELINE
    # ------------------------------------------------------------------------------------------------------------------

    model_specific_transform = ModelSpecificTransform()  # <- Likely depends on dict_task_metadata

    train_model_dataset = initialize_TokaMark_dataset(
        dataset=train_MAST_dataset,
        task_metadata=dict_task_metadata,
        config_metadata=config_task,
        custom_transform=model_specific_transform,
        **pipeline_config["tokamark_dataset_init_settings"],
        verbose=False,
    )

    train_dataloader = DataLoader(
        dataset=train_model_dataset,  # noqa - Ignore expected type warning
        collate_fn=model_collate_fn,
        # drop_last=True,
        **pipeline_config["dataloader_settings"],
        pin_memory=True,
    )

    # Similarly, DataLoader instances can be created for val and test, i.e., val_dataloader and test_dataloader.

    # ..................................................................................................................
    # Evaluation loop for train_dataloader
    # ..................................................................................................................

    for batch_idx, batch_ in enumerate(train_dataloader):
        print(f"\nBatch {batch_idx}")
        shot_id, window_index, x_train, y_train = batch_  # noqa - Right number of values to unpack

        print(f"The list of shot ID is {shot_id}")
        print(f"The list of window index is {window_index}")

        print("The x_train has been collated to shape (B, ..., T), ", [arr.shape for arr in x_train])
        # print("Mean x_train", [torch.nanmean(arr) for arr in x_train])
        # print("Std x_train", [np.nanstd(arr) for arr in x_train])

        print("The y_train has been collated to shape (B, ..., T), ", [arr.shape for arr in y_train])
        # print("Mean y_train", [torch.nanmean(arr) for arr in y_train])
        # print("Std y_train", [np.nanstd(arr) for arr in y_train])

        print("____________________________________________________\n")

    # print(x_train[0][0:10])
    # print("\n\n\n")
    # print(y_train[0][0:10])

    # ..................................................................................................................
    # Evaluation loop for val and test
    # ..................................................................................................................

    # Similarly, evaluation loops can be put in place for val_dataloader and test_dataloader instances.

    # ------------------------------------------------------------------------------------------------------------------

end = time.perf_counter()

print("\n-----------------------------")
print(f"Elapsed time: {end - start:.4f} seconds")
