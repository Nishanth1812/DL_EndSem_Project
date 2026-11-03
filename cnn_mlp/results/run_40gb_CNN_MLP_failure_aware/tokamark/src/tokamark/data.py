"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

from collections.abc import Callable, Mapping
from typing import Any

from tokamark.tools.TokaMark_dataset import TokaMarkDataset
from tokamark.tools.MAST_composite_transform import build_common_signal_transform_map
from tokamark.tools.path import RANDOM_SPLIT_SIGNALS_STATS_FILE
from MAST_tools.MAST_dataset import MastDataset
from MAST_tools.utils.data_utils import StoreManagerParametersType
from MAST_tools.utils.path_utils import RANDOM_SPLIT_OUTLIER_METADATA_FILE


# ----------------------------------------------------------------------------------------------------------------------
def initialize_MAST_dataset(  # noqa - Ignore lowercase warning
    config_task: Mapping[str, Any],
    shots_list: list[int],
    local_flag: bool = True,
    use_std_scaling: bool = True,
    stats_metadata_file_path: str = RANDOM_SPLIT_SIGNALS_STATS_FILE,
    use_nan_filling: bool = False,
    remove_outliers: bool = True,
    outlier_metadata_file: str = RANDOM_SPLIT_OUTLIER_METADATA_FILE,
    remove_bad_efit_rating: bool = True,
    *,
    store_manager_settings: StoreManagerParametersType | None = None,
    verbose: bool = False,
) -> MastDataset:
    """
    Initialize and return MAST dataset.

    Parameters
    ----------
    config_task : Mapping[str, Any],
        Task configuration dictionary.
    shots_list : list[int]
        List of target shots.
    local_flag : bool
        If True, local mode is used.
        Optional. Default: True.
    use_nan_filling : bool
        If True, NaN filling is used.
        Optional. Default: True.
    use_std_scaling : bool
        If True, standard scaling is used.
        Optional. Default: True.
    stats_metadata_file_path : str
        Path to suitable YAML file with stats metadata.
        Optional. Default: `tokamark.tools.path.RANDOM_SPLIT_SIGNALS_STATS_FILE`.
    remove_outliers : bool
        If True, outliers are removed.
        Optional. Default: True.
    outlier_metadata_file : str
        Path to suitable YAML file with outlier metadata.
        Optional. Default: `MAST_tools.utils.path_utils.RANDOM_SPLIT_OUTLIER_METADATA_FILE`.
    remove_bad_efit_rating : bool
        If True, bad EFIT ratings are removed.
        Optional. Default: True.
    store_manager_settings : StoreManagerParametersType | None
        Settings for the store manager instance provided as a kwargs dictionary, with keywords and required value
        types as defined in `MAST_tools.utils.data_utils.StoreManagerParameters`. Only valid (keyword, value) pairs
        are used to update default values, e.g. {"target_fsspec_protocol": "s3"}.
        Optional. Default: None, which results in the default values for all the keywords as defined in
        `MAST_tools.store_utils.MASTStorageManager.__init__`.
    verbose : bool
        If True, verbose mode is activated.
        Optional. Default: False.

    """

    # ..................................................................................................................
    # Get unique source-signal

    source_signal_list = (
        (config_task["sources_and_signals"].get("input_name") or [])
        + (config_task["sources_and_signals"].get("actuator_name") or [])
        + (config_task["sources_and_signals"].get("output_name") or [])
    )
    source_signal_list = [s for i, s in enumerate(source_signal_list) if s not in source_signal_list[:i]]  # Unicity

    # ..................................................................................................................
    # Create common transform map

    signal_transform_map = build_common_signal_transform_map(
        source_signal_list=source_signal_list,
        use_std_scaling=use_std_scaling,
        stats_metadata_file_path=stats_metadata_file_path,
        use_nan_filling=use_nan_filling,
    )

    mast_dataset = MastDataset(
        local=local_flag,
        shots_list=shots_list,
        source_signal_list=source_signal_list,
        signal_level_transform_map=signal_transform_map,
        remove_outliers=remove_outliers,
        outlier_metadata_file=outlier_metadata_file,
        remove_bad_efit_rating=remove_bad_efit_rating,
        store_manager_settings=store_manager_settings,
        verbose=verbose,
    )

    return mast_dataset


# ----------------------------------------------------------------------------------------------------------------------
def initialize_TokaMark_dataset(  # noqa - Ignore lowercase warning
    dataset: MastDataset | None,
    task_metadata: Mapping[str, Any],
    config_metadata: Mapping[str, Any],
    custom_transform: Callable | None = None,
    test_mode: bool = False,
    shuffle_windows: bool = True,
    shuffle_buffer_size: int = 512,
    *,
    verbose: bool = False,
) -> TokaMarkDataset | None:
    """
    #
    Wrap a single baseline shot-level dataset with TokaMarkDataset.
    This method is intended to be called explicitly for each split (train/val/test) from entrypoint scripts.

    Parameters
    ----------
    dataset : MastDataset | None
        Baseline shot-level dataset (e.g., MastDataset) for one split, or None.
    task_metadata : Mapping[str, Any]
        Metadata dictionary produced by the baseline pipeline (dt, shapes, etc.).
    config_metadata : Mapping[str, Any]
        Task configuration dictionary containing `task_window_segmenter` (keys, lengths, delta).
    custom_transform : Callable | None
        Optional model-specific transform chain applied per window.
    test_mode : bool
        If True, activates test mode. This forces at least one input to be full, and all outputs to be full.
        Optional. Default: False.
    shuffle_windows : bool
        If True, windows are shuffled before returning them.
        Optional. Default: True.
    shuffle_buffer_size : int
        Size of the shuffle buffer.
        Optional. Default: 512
    verbose : bool
        If True, enables verbose prints in the wrapper.

    Returns
    -------
    TokaMarkDataset | None
        Wrapped dataset, or None if input dataset is None.

    """

    if dataset is None:
        return None

    return TokaMarkDataset(
        base_dataset=dataset,
        task_metadata=task_metadata,
        config_metadata=config_metadata,
        custom_transform=custom_transform,
        test_mode=test_mode,
        shuffle_windows=shuffle_windows,
        shuffle_buffer_size=shuffle_buffer_size,
        verbose=verbose,
    )
