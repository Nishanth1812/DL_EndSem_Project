"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import os
from typing import cast
import pandas as pd
import random

from tokamark.tools.path import RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE


# ----------------------------------------------------------------------------------------------------------------------
def get_train_test_val_shots(
    max_index: int | None = None,
    max_index_for_train: int | None = None,
    max_index_for_val: int | None = None,
    max_index_for_test: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    data_splits_file_path: str = RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE,
) -> tuple[list, list, list]:
    """
    Generate lists of shot IDs for training, testing, and validation. These lists can be subsets of the corresponding
    complete lists.

    Parameters
    ----------
    max_index : int | None
        If not None, all lists will have the same length given by max_index.
        Optional. Default: None.
    max_index_for_train : int | None
        Number of shot IDs for the training set. Overrides max_index.
        Optional. Default: None.
    max_index_for_val : int | None
        Number of shot IDs for the validation set. Overrides max_index.
        Optional. Default: None.
    max_index_for_test : int | None
        Number of shot IDs for the testing set. Overrides max_index.
        Optional. Default: None.
    shuffle : bool
        True if we need shuffled samples, which shuffling done before selecting subset of shot IDs to use.
        Optional. Default: False.
    seed : int
        For reproducibility of the rnd sequence.
        Optional. Default: 42.
    data_splits_file_path : str
        Path to the data splits CSV file.
        Optional. Default: `tokamark.tools.path.RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE`.

    Returns
    -------
    tuple[list, list, list]
        Three lists of shot IDs for training, testing, and validation, respectively.

    """

    # Read full data splits
    train_set_full, test_set_full, val_set_full = read_data_split_csv(csv_path=data_splits_file_path)

    if shuffle:
        if not isinstance(seed, int):
            raise ValueError(f"Seed must be integer, got {type(seed).__name__}")  # noqa - Ignore unreach code warning

        random.seed(seed)
        random.shuffle(train_set_full)
        random.shuffle(test_set_full)
        random.shuffle(val_set_full)

    train_set = train_set_full
    test_set = test_set_full
    val_set = val_set_full

    # If max_index is provided, override all other limits
    if (max_index is not None) and (max_index > 0):
        train_set = train_set_full[:max_index]
        val_set = val_set_full[:max_index]
        test_set = test_set_full[:max_index]

    # Apply individual limits if provided and positive
    if (max_index_for_train is not None) and (max_index_for_train > 0):
        train_set = train_set_full[:max_index_for_train]

    if (max_index_for_val is not None) and (max_index_for_val > 0):
        val_set = val_set_full[:max_index_for_val]

    if (max_index_for_test is not None) and (max_index_for_test > 0):
        test_set = test_set_full[:max_index_for_test]

    return train_set, test_set, val_set


# ----------------------------------------------------------------------------------------------------------------------
def read_data_split_csv(csv_path: str) -> tuple[list, list, list]:
    """
    Read the CSV file containing the lists of shot IDs for training, testing, and validation.

    Parameters
    ----------
    csv_path : str
        Target CSV path.

    Returns
    -------
    tuple[list, list, list]
        List of shot IDs for training, testing, and validation.

    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)  # noqa - Ignore missing parameter(s)
    df = cast(pd.DataFrame, df)

    shot_ids_for_train = df[df["train"] == True]["shot_id"].tolist()  # noqa - Ignore E712, "is" comparison fails
    shot_ids_for_test = df[df["test"] == True]["shot_id"].tolist()  # noqa - Ignore E712, "is" comparison fails
    shot_ids_for_val = df[df["val"] == True]["shot_id"].tolist()  # noqa - Ignore E712, "is" comparison fails

    return shot_ids_for_train, shot_ids_for_test, shot_ids_for_val


# ----------------------------------------------------------------------------------------------------------------------
