"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import random
import numpy as np
from typing import Any, Union
from collections.abc import Callable, Mapping, Generator

from torch.utils.data import IterableDataset, get_worker_info

from MAST_tools.MAST_dataset import MastDataset


# ======================================================================================================================
# HELPERS


# ----------------------------------------------------------------------------------------------------------------------
def _all_vars_have_all_nans(dict_obj: Mapping[str, Any]) -> bool:
    """
    Check if all variables in an input mapping have all Nan values.

    Parameters
    ----------
    dict_obj : Mapping[str, Any]
        Input mapping to be checked.

    Returns
    -------
    bool
        If True, all variables in input mapping have all NaN values.

    """

    return all(np.isnan(np.asarray(dict_obj[var]["values"])).all() for var in dict_obj.keys())


# ----------------------------------------------------------------------------------------------------------------------
def _all_vars_have_any_nans(dict_obj: Mapping[str, Any]) -> bool:
    """
    Check if all variables in an input mapping have any Nan values.

    Parameters
    ----------
    dict_obj : Mapping[str, Any]
        Input mapping to be checked.

    Returns
    -------
    bool
        If True, all variables in input mapping have NaN values.

    """

    return all(np.isnan(np.asarray(dict_obj[var]["values"])).any() for var in dict_obj.keys())


# ----------------------------------------------------------------------------------------------------------------------
def _any_vars_have_all_nans(dict_obj: Mapping[str, Any]) -> bool:
    """
    Check if at least one variable in an input mapping has all Nan values.

    Parameters
    ----------
    dict_obj : Mapping[str, Any]
        Input mapping to be checked.

    Returns
    -------
    bool
        If True, at least one variable in input mapping has all NaN values.

    """

    return any(np.isnan(np.asarray(dict_obj[var]["values"])).all() for var in dict_obj.keys())


# ----------------------------------------------------------------------------------------------------------------------
def _any_vars_have_any_nans(dict_obj: Mapping[str, Any]) -> bool:
    """
    Check if any variables in an input mapping have any Nan values.

    Parameters
    ----------
    dict_obj : Mapping[str, Any]
        Input mapping to be checked.

    Returns
    -------
    bool
        If True, at least one variable in input mapping have at least one NaN values.

    """

    return any(np.isnan(np.asarray(dict_obj[var]["values"])).any() for var in dict_obj.keys())


# ----------------------------------------------------------------------------------------------------------------------
def _shuffle_buffer(iterator: Generator, buffer_size: int = 512) -> Generator:
    """
    Streaming shuffle buffer.

    Parameters
    ----------
    iterator : Generator
        Input iterator.
    buffer_size : int
        Size of the buffer.
        Optional. Default: 512.

    Returns
    -------
    Generator
        A Generator item.

    """

    buffer = []
    for item in iterator:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            idx = random.randrange(len(buffer))
            yield buffer.pop(idx)

    while buffer:
        idx = random.randrange(len(buffer))
        yield buffer.pop(idx)


# ======================================================================================================================
class TokaMarkDataset(IterableDataset):
    """
    Iterable dataset class for TokaMark (MAST) data.

    Attributes
    ----------
    base : MastDataset
        Base dataset.
    shots_list : list[int]
        List of shot IDs in the dataset.
    task_metadata : Mapping[str, Any]
        Dictionary with task metadata.
    data_metadata : Mapping[str, Any]
        Dictionary with data metadata (namely, "dt" and "shape_values").
    task_type : str
        Type of task, either "markovian" or "non_markovian".
    input_keys : list[str]
        List of input keys in the "{source}-{signal}" format.
    actuator_keys : list[str]
        List of actuator keys in the "{source}-{signal}" format.
    output_keys : list[str]
        List of output keys in the "{source}-{signal}" format.
    input_length : float
        Input length in seconds.
    output_length : float
        Output length in seconds.
    delta : float
        Distance in seconds between the end on the input interval and the beginning of the output interval.
        For forecasting, `delta` should be 0, and for reconstruction, `delta` should be `-input_length`.
    stride : float
        Distance between two consecutive windows.
    custom_transform : Callable | None
        Custom model-specific transform chain applied per window.
    test_mode : bool
        If True, activates test mode. This forces at least one input to be full, and all outputs to be full.
    shuffle_windows : bool
        If True, shuffling of windows is performed.
    shuffle_buffer_size : int
        Size of the shuffle buffer.
    verbose : bool
        If True, verbose mode is activated.

    Methods
    -------
    __iter__()
        Worker-safe iterator.
    _iterate_shots(start, end)
        Shot iterator.
    _process_shot(shot_idx)
        Shot-processing generator.
    _build_window(sample, global_start_time, t_cut, type_window)
        Window-builder method.
    _pad_time_series_to_interval(times, values, dt, t_start, t_end, shape_values)
        Pad an input time series to a target interval.
    _pad_sample_to_interval(sample, t_start, t_end)
        Pad an input sample to a target interval.
    get_shot_id(idx)
        Return shot ID from shot index.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        base_dataset: MastDataset,
        task_metadata: Mapping[str, Any],
        config_metadata: Mapping[str, Any],
        custom_transform: Callable | None = None,
        test_mode: bool = False,
        shuffle_windows: bool = False,
        shuffle_buffer_size: int = 512,
        verbose: bool = False,
    ) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        base_dataset : MastDataset
            Base dataset.
        task_metadata : Mapping[str, Any]
             Dictionary with task metadata.
        config_metadata : Mapping[str, Any]
            Dictionary with configuration metadata.
        custom_transform : Callable | None
            Custom model-specific transform chain applied per window.
            Optional. Default: None.
        test_mode : bool
            If True, activates test mode. This forces at least one input to be full, and all outputs to be full.
            Optional. Default: False.
        shuffle_windows : bool
            If True, shuffling of windows is performed.
            Optional. Default: False.
        shuffle_buffer_size : int
            Size of the shuffle buffer.
            Optional. Default: 512.
        verbose : bool
            If True, verbose mode is activated.
            Optional. Default: False.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors.

        """

        super().__init__()

        self.base = base_dataset
        self.shots_list = self.base.shots_list
        self.task_metadata = task_metadata

        self.data_metadata: dict[Any, dict[str, Any]] = {
            key: {"dt": meta["dt"], "shape_values": tuple(meta["values_shape"])}
            for d in [task_metadata["input"], task_metadata["actuator"], task_metadata["output"]]
            for key, meta in d.items()
        }

        self.task_type = config_metadata["task_type"]
        segmenter_metadata = config_metadata["task_window_segmenter"]

        self.input_keys = [f"{s}-{k}" for s, k in (segmenter_metadata["input_keys"] or [])]
        self.actuator_keys = [f"{s}-{k}" for s, k in (segmenter_metadata["actuator_keys"] or [])]
        self.output_keys = [f"{s}-{k}" for s, k in (segmenter_metadata["output_keys"] or [])]

        self.input_length = segmenter_metadata["input_length"]
        self.output_length = segmenter_metadata["output_length"]
        self.delta = segmenter_metadata["delta"]

        self.stride = float(self.task_metadata["stride_in_secs"])

        self.custom_transform = custom_transform
        self.test_mode = test_mode
        self.shuffle_windows = shuffle_windows
        self.shuffle_buffer_size = shuffle_buffer_size
        self.verbose = verbose

    # ------------------------------------------------------------------------------------------------------------------
    def __iter__(self) -> Generator:
        """
        Worker-safe iterator.

        Returns
        -------
        Generator
            A Generator item.

        """

        worker_info = get_worker_info()

        if worker_info is None:
            start = 0
            end = len(self.base)
        else:
            per_worker = int(np.ceil(len(self.base) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.base))

        iterator = self._iterate_shots(start=start, end=end)

        if self.shuffle_windows:
            iterator = _shuffle_buffer(iterator=iterator, buffer_size=self.shuffle_buffer_size)

        yield from iterator

    # ------------------------------------------------------------------------------------------------------------------
    def _iterate_shots(self, start: int, end: int) -> Generator:
        """
        Shot iterator.

        Parameters
        ----------
        start : int
            Start shot index.
        end : int
            End shot index.

        Returns
        -------
        Generator
            A Generator item.

        """

        for shot_idx in range(start, end):
            yield from self._process_shot(shot_idx=shot_idx)

    # ------------------------------------------------------------------------------------------------------------------
    def _process_shot(  # NOSONAR - Ignore cognitive complexity
        self, shot_idx: int, filled_test_mode: bool = False
    ) -> Generator:
        """
        Shot-processing generator.

        Parameters
        ----------
        shot_idx : int
            Shot index.
        filled_test_mode : bool
            If True and `self.test_mode` is True, use filled test mode. If False and `self.test_mode` is True, use
            sparse test mode. Ignored if `self.test_mode` is False.
            Optional. Default: True.

        Returns
        -------
        Generator
            A Generator item.

        """

        sample = self.base[shot_idx]

        # ..............................................................................................................
        # Determine global_start_time based on input and actuator

        t_start = []
        for var in self.input_keys + self.actuator_keys:
            t = sample[var]["time"]
            v = sample[var]["values"]

            valid_mask = ~np.all(np.isnan(v), axis=tuple(range(v.ndim - 1)))
            if not np.any(valid_mask):
                continue
            t_valid = t[valid_mask]
            t_start.append(t_valid[0])

        if not t_start:
            return

        global_start_time = float(np.min(t_start))

        # ..............................................................................................................
        # Determine global_end_time based on output

        t_end = []
        for var in self.output_keys:
            t = sample[var]["time"]
            v = sample[var]["values"]

            if t.size == 0 or v.size == 0:
                if self.test_mode:
                    if self.verbose:
                        print(f"Skipping shot {self.get_shot_id(idx=shot_idx)} (empty output '{var}')")
                    return  # ← skip whole shot
                continue

            valid_mask = ~np.all(np.isnan(v), axis=tuple(range(v.ndim - 1)))
            if not np.any(valid_mask):
                continue
            t_valid = t[valid_mask]
            t_end.append(t_valid[-1])

        if not t_end:
            return

        global_end_time = float(np.max(t_end))

        # ..............................................................................................................
        # Skip shot if global times are non-finite or the window range is too short for even one cut

        if not (np.isfinite(global_start_time) and np.isfinite(global_end_time)):
            if self.verbose:
                print(f"Skipping shot {self.get_shot_id(idx=shot_idx)}: non-finite global times.")
            return

        t_cut_start = global_start_time + self.input_length
        t_cut_stop = global_end_time - self.delta - self.output_length
        if t_cut_start >= t_cut_stop:
            if self.verbose:
                print(f"Skipping shot {self.get_shot_id(idx=shot_idx)}: window range too short for any cut.")
            return

        # --------------------------------------------------------------------------------------------------------------
        # Pad sample for consistency

        sample = self._pad_sample_to_interval(sample=sample, t_start=global_start_time, t_end=global_end_time)

        # ..............................................................................................................
        # Build windows

        t_cuts = np.arange(
            t_cut_start,
            stop=t_cut_stop,
            step=self.stride,
        )

        for idx_t, t_cut in enumerate(t_cuts):
            input_slice = self._build_window(
                sample=sample,
                global_start_time=global_start_time,
                t_cut=t_cut,  # noqa - Ignore expected type warning
                type_window="input",
            )

            actuator_slice = self._build_window(
                sample=sample,
                global_start_time=global_start_time,
                t_cut=t_cut,  # noqa - Ignore expected type warning
                type_window="actuator",
            )

            output_slice = self._build_window(
                sample=sample,
                global_start_time=global_start_time,
                t_cut=t_cut,  # noqa - Ignore expected type warning
                type_window="output",
            )

            obj: dict[str, Any] = {
                "input": input_slice,
                "actuator": actuator_slice,
                "output": output_slice,
                "t_cut": t_cut,
                "shot_id": self.get_shot_id(idx=shot_idx),
                "window_index": idx_t,
            }

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Filtering

            if self.test_mode:
                if filled_test_mode:
                    # Filled test_mode
                    window_valid = not (
                        _all_vars_have_any_nans(obj["input"]) and _all_vars_have_any_nans(obj["actuator"])
                    ) and (not _any_vars_have_any_nans(obj["output"]))
                else:
                    # Sparse test_mode
                    window_valid = not (
                        _all_vars_have_all_nans(obj["input"]) and _all_vars_have_all_nans(obj["actuator"])
                    ) and (not _any_vars_have_all_nans(obj["output"]))

                if not window_valid:
                    if self.verbose:
                        print(f"Window {obj['window_index']} of shot {obj['shot_id']} is not valid.")
                    continue

            obj2 = self.custom_transform(obj) if self.custom_transform else obj
            if obj2 is None:
                continue

            yield {"shot_id": obj["shot_id"], "window_index": idx_t, **obj2}

    # ------------------------------------------------------------------------------------------------------------------
    def _build_window(  # NOSONAR - Ignore cognitive complexity
        self,
        sample: Mapping[str, Any],
        global_start_time: float,
        t_cut: Union[float, np.float16, np.float32, np.float64],
        type_window: str,
    ) -> dict[str, np.ndarray]:
        """
        Window-builder method.

        Parameters
        ----------
        sample : Mapping[str, Any]
            Input sample.
        global_start_time : float
            Global starting time for the target window.
        t_cut : float
            Cutting time for the target window.
        type_window : str
            Target window type. Valid options: "input", "actuator", "output"

        Returns
        -------
        dict
            Built window from passed values.

        """

        out = {}

        keys = {"input": self.input_keys, "actuator": self.actuator_keys, "output": self.output_keys}

        for key in keys[type_window]:
            md = self.task_metadata[type_window][key]

            dt = md["dt"]
            shape_values = tuple(md["values_shape"])

            ts_in = int(round(self.input_length / dt))
            ts_out = int(round(self.output_length / dt))
            ts_delta = int(round(self.delta / dt))

            ts_len = {"input": ts_in, "actuator": ts_in + ts_delta + ts_out, "output": ts_out}

            times = sample[key]["time"]
            values = sample[key]["values"]

            if (times.size == 0) or (values.size == 0):
                selected_times = np.full(ts_len[type_window], np.nan)
                selected_values = np.full(shape_values + (ts_len[type_window],), np.nan)

            else:
                # ......................................................................................................
                # Index Logic

                idx = None
                if type_window == "input":
                    end_idx = int(round((t_cut - times[0]) / dt))

                    if self.task_type == "markovian":
                        idx = np.arange(end_idx - ts_in, end_idx)
                    else:
                        global_start_idx = int(round((global_start_time - times[0]) / dt))
                        idx = np.arange(global_start_idx, end_idx)

                elif type_window == "output":
                    cut_time = t_cut + self.delta
                    start_idx = int(round((cut_time - times[0]) / dt))
                    idx = np.arange(start_idx, start_idx + ts_out)

                elif type_window == "actuator":
                    cut_idx = int(round((t_cut - times[0]) / dt))

                    if self.task_type == "markovian":
                        idx_in = np.arange(cut_idx - ts_in, cut_idx)
                    else:
                        global_start_idx = int(round((global_start_time - times[0]) / dt))
                        idx_in = np.arange(global_start_idx, cut_idx)

                    idx_out = np.arange(cut_idx, cut_idx + ts_delta + ts_out)

                    idx = np.concatenate([idx_in, idx_out])

                # ......................................................................................................

                idx = np.clip(idx, 0, len(times) - 1)
                selected_times = times[idx]
                selected_values = values[..., idx]

            if (selected_values.ndim == 2) and (selected_values.shape[0] == 1):
                selected_values = selected_values[0]

            out[key] = {"time": selected_times, "values": selected_values}

        return out

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _pad_time_series_to_interval(
        times: np.ndarray, values: np.ndarray, dt: float, t_start: float, t_end: float, shape_values: tuple
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pad an input time series to a target interval.

        Parameters
        ----------
        times : np.ndarray
            Input array of times.
        values : np.ndarray
            Input array of values to be padded.
        dt : float
            Input time granularity.
        t_start : float
            Starting time for the target interval.
        t_end : float
            Ending time for the target interval.
        shape_values : tuple
            Target shape for the padded values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (times, values) tuple.

        """

        if not np.issubdtype(values.dtype, np.floating):
            values = values.astype(float)

        # ..............................................................................................................
        # Left pad

        if times[0] > t_start:
            n_pad = int(np.ceil((times[0] - t_start) / dt)) + 1

            left_times = times[0] - dt * np.arange(n_pad, 0, -1)
            times = np.concatenate([left_times, times])

            pad_shape = shape_values + (n_pad,)  # noqa - Ignore missing attribute warning
            left_pad = np.full(shape=pad_shape, fill_value=np.nan)

            values = np.concatenate([left_pad, values], axis=-1)

        # ..............................................................................................................
        # Right pad

        if times[-1] < t_end:
            n_pad = int(np.ceil((t_end - times[-1]) / dt)) + 1

            right_times = times[-1] + dt * np.arange(1, n_pad + 1)
            times = np.concatenate([times, right_times])

            pad_shape = shape_values + (n_pad,)  # noqa - Ignore missing attribute warning
            right_pad = np.full(pad_shape, np.nan)

            values = np.concatenate([values, right_pad], axis=-1)

        return times, values

    # ------------------------------------------------------------------------------------------------------------------
    def _pad_sample_to_interval(self, sample: Mapping[str, Any], t_start: float, t_end: float) -> dict[str, Any]:
        """
        Pad an input sample to a target interval.

        Parameters
        ----------
        sample : Mapping[str, Any]
            Input sample to be padded to a target interval.
        t_start : float
            Starting time of the target interval.
        t_end : float
            Ending time of the target interval.

        Returns
        -------
        dict[str, Any]
            Padded sample.

        """

        padded_sample = dict(sample)

        for key in sample.keys():
            times = sample[key]["time"]
            values = sample[key]["values"]

            if times.size == 0 or values.size == 0:
                continue

            dt: float = self.data_metadata[key]["dt"]
            shape_values = self.data_metadata[key]["shape_values"]

            times, values = self._pad_time_series_to_interval(
                times=times, values=values, dt=dt, t_start=t_start, t_end=t_end, shape_values=shape_values
            )

            padded_sample[key]["time"] = times
            padded_sample[key]["values"] = values

        return padded_sample

    # ------------------------------------------------------------------------------------------------------------------
    def get_shot_id(self, idx: int) -> int:
        """
        Return shot ID from shot index.

        Parameters
        ----------
        idx : int
            Shot index.

        Returns
        -------
        int
            Shot ID corresponding to the provided shot index.

        """

        return self.base.shots_list[idx]

    # ------------------------------------------------------------------------------------------------------------------
