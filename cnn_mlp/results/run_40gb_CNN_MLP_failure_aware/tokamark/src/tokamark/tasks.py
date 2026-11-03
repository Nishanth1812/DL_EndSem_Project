"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import yaml
import numpy as np
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from tokamark.tools.utils import get_config_from_yaml
from tokamark.tools.path import TASKS_CONFIGS_DIR, RANDOM_SPLIT_SIGNALS_STATS_FILE

# ----------------------------------------------------------------------------------------------------------------------

GROUP_TASKS = {  # REMARK: Unused here, but imported elsewhere.
    1: ["task_1-1", "task_1-2", "task_1-3"],
    2: ["task_2-1", "task_2-2", "task_2-3"],
    3: ["task_3-1", "task_3-2", "task_3-3"],
    4: ["task_4-1", "task_4-2", "task_4-3", "task_4-4", "task_4-5"],
}

TASKS_CONFIGS_MAP = {
    "task_1-1": "group_1_reconstruction/task_1-1.yaml",
    "task_1-2": "group_1_reconstruction/task_1-2.yaml",
    "task_1-3": "group_1_reconstruction/task_1-3.yaml",
    "task_2-1": "group_2_magnetics_dynamics/task_2-1.yaml",
    "task_2-2": "group_2_magnetics_dynamics/task_2-2.yaml",
    "task_2-3": "group_2_magnetics_dynamics/task_2-3.yaml",
    "task_3-1": "group_3_profiles_dynamics/task_3-1.yaml",
    "task_3-2": "group_3_profiles_dynamics/task_3-2.yaml",
    "task_3-3": "group_3_profiles_dynamics/task_3-3.yaml",
    "task_4-1": "group_4_mhd_activity/task_4-1.yaml",
    "task_4-2": "group_4_mhd_activity/task_4-2.yaml",
    "task_4-3": "group_4_mhd_activity/task_4-3.yaml",
    "task_4-4": "group_4_mhd_activity/task_4-4.yaml",
    "task_4-5": "group_4_mhd_activity/task_4-5.yaml",
}


# ----------------------------------------------------------------------------------------------------------------------
def get_task_config(task_name: str) -> dict[str, Any]:
    """
    Get task configuration by task name.

    Parameters
    ----------
    task_name : str
        Name of the target task.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary from the corresponding task-specific YAML file.

    """

    task_path = TASKS_CONFIGS_MAP[task_name]
    file_path = Path(TASKS_CONFIGS_DIR) / task_path

    return get_config_from_yaml(file_path=file_path)


# ----------------------------------------------------------------------------------------------------------------------
def get_signals_metadata(file_path: str = RANDOM_SPLIT_SIGNALS_STATS_FILE) -> dict[str, Any]:
    """
    Read signals metadata file and return content as a dictionary.

    Parameters
    ----------
    file_path : str
        Target file path for signals metadata.
        Optional. Default: `tokamark.tools.path.RANDOM_SPLIT_SIGNALS_STATS_FILE`.

    Returns
    -------
    dict[str, Any]
        Dictionary with signals metadata from the target metadata file.

    """

    with open(file_path, "r") as f:
        dict_stats_metadata = yaml.safe_load(f)

    return dict_stats_metadata


# ----------------------------------------------------------------------------------------------------------------------
def get_task_metadata(  # NOSONAR - Ignore cognitive complexity
    config_task: Mapping[str, Any], verbose: bool = False
) -> dict[str, Any]:
    """
    Get task metadata for target configuration task.

    Parameters
    ----------
    config_task : Mapping[str, Any]
        Dictionary with task configuration.
    verbose : bool
        If True, verbose mode is activated.
        Default: False.

    Returns
    -------
    dict[str, Any]
        Output dictionary with task metadata.

    """

    # ..................................................................................................................
    # Import data metadata
    # ..................................................................................................................

    dict_stats_metadata = get_signals_metadata()

    # ..................................................................................................................
    # Import task-specific roles
    # ..................................................................................................................

    # Input
    input_keys = [f"{source}-{signal}" for source, signal in (config_task["task_window_segmenter"]["input_keys"] or [])]

    # Actuator
    actuator_keys = [
        f"{source}-{signal}" for source, signal in (config_task["task_window_segmenter"]["actuator_keys"] or [])
    ]

    # Output
    output_keys = [
        f"{source}-{signal}" for source, signal in (config_task["task_window_segmenter"]["output_keys"] or [])
    ]

    # ..................................................................................................................
    # Import task-specific values
    # ..................................................................................................................

    input_length = config_task["task_window_segmenter"]["input_length"]
    output_length = config_task["task_window_segmenter"]["output_length"]
    delta = config_task["task_window_segmenter"]["delta"]

    # Get stride from config file
    stride_in_secs = config_task["stride_window"]  # min([dict_metadata[key]["dt"] for key in output_keys])

    # Split into role-scoped dicts (avoid overwriting)
    out = {
        "stride_in_secs": stride_in_secs,  # Time in seconds between two consecutive windows.
        "input": {},
        "actuator": {},
        "output": {},
    }

    # ..................................................................................................................
    # Get information
    # ..................................................................................................................

    # Input
    length_in_secs = input_length
    for key in input_keys:
        out["input"][key] = dict_stats_metadata[key]
        out["input"][key]["length_in_seconds"] = length_in_secs
        out["input"][key]["length_in_time_stamps"] = int(np.round(length_in_secs / out["input"][key]["dt"]))
        out["input"][key]["stride_in_time_stamps"] = int(np.round(stride_in_secs / out["input"][key]["dt"]))

    # Actuator
    length_in_secs = input_length + delta + output_length
    for key in actuator_keys:
        out["actuator"][key] = dict_stats_metadata[key]
        out["actuator"][key]["length_in_seconds"] = length_in_secs
        out["actuator"][key]["length_in_time_stamps"] = int(np.round(length_in_secs / out["actuator"][key]["dt"]))
        out["actuator"][key]["stride_in_time_stamps"] = int(np.round(stride_in_secs / out["actuator"][key]["dt"]))

    # Output
    length_in_secs = output_length
    for key in output_keys:
        out["output"][key] = dict_stats_metadata[key]
        out["output"][key]["length_in_seconds"] = length_in_secs
        out["output"][key]["length_in_time_stamps"] = int(np.round(length_in_secs / out["output"][key]["dt"]))
        out["output"][key]["stride_in_time_stamps"] = int(np.round(stride_in_secs / out["output"][key]["dt"]))

    # ..................................................................................................................
    # Relevant prints, if verbose mode is activated
    # ..................................................................................................................

    if verbose:
        for role in ("input", "actuator", "output"):
            for key, val in out[role].items():
                print(f"\nSignal: {key}")
                if val["dt"] is not None:
                    print(f"  dt: {val['dt']:.5f} s")
                else:
                    print("  dt: None")
                print(f"  Values shape: {val['values_shape']}")

    return out

    # ..................................................................................................................
