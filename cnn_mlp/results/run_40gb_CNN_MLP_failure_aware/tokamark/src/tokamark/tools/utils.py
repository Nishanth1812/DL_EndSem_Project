"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import yaml
from pathlib import Path
from filelock import FileLock
import pandas as pd
from typing import Any, Union

import torch


# ----------------------------------------------------------------------------------------------------------------------
def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Return the best available torch device.

    Parameters
    ----------
    prefer_mps : bool
        Whether to prefer Apple Metal Performance Shaders (MPS) over CPU.
        Optional. Default: True.

    Returns
    -------
    torch.device
        Torch device.

    """

    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------------------------------------------------
def get_config_from_yaml(file_path: str | Path) -> dict[str, Any]:
    """
    Get configuration dictionary from YAML file.

    Parameters
    ----------
    file_path : str | Path
        Target file path.

    Returns
    -------
    dict[str, Any]
        Loaded YAML file as dictionary.

    Raises
    ------
    ValueError
        If provided `file_path` resulted in loaded configuration that is not a mapping (dict).

    """

    # Load YAML config
    with open(file_path, "r") as f:  # noqa - Ignore unexpected type
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Provided `file_path` {file_path!r} resulted in loaded configuration of type {type(config).__name__}, "
            f"expected dict."
        )

    return config


# ======================================================================================================================
class AutoAppendingDataFrame:
    """
    Auto-saving DataFrame that buffers rows and writes atomically after N batches.

    Attributes
    ----------
    path : str
        Path to the target CSV file.
    buffer_size : int
        Number of rows to buffer before saving.
    lock : BaseFileLock
        Lock for concurrent appending.
    buffer : list
        Buffer for DataFrame rows.
    columns : Optional[list]
        List of DataFrame columns.

    Methods
    -------
    append(df_rows)
        Append rows to buffer and commit if threshold reached.
    _commit()
        Commit buffered rows to disk atomically.
    flush()
        Force commit of any buffered rows.
    view()
        Return a copy of the current DataFrame.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, path: Union[Path, str], buffer_size: int = 1) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the target CSV file.
        buffer_size : int
            Number of rows to buffer before saving.
            Optional. Default: 1.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors.

        """

        self.path = Path(path)
        self.buffer_size = buffer_size
        self.lock = FileLock(str(self.path) + ".lock")
        self.buffer = []
        self.columns = None

    # ------------------------------------------------------------------------------------------------------------------
    def append(self, df_rows: pd.DataFrame) -> None:
        """
        Append rows to buffer and commit if threshold reached.

        Parameters
        ----------
        df_rows : pd.DataFrame
            Input dataframe.

        Returns
        -------
        None

        """

        if self.columns is None:
            self.columns = list(df_rows.columns)

            # checking for column consistency
            if list(df_rows.columns) != self.columns:
                raise ValueError(f"Column mismatch: expected {self.columns}, got {list(df_rows.columns)}")

        self.buffer.append(df_rows)

        if len(self.buffer) >= self.buffer_size:
            self._commit()

    # ------------------------------------------------------------------------------------------------------------------
    def _commit(self) -> None:
        """Commit buffered rows to disk atomically."""

        if not self.buffer:
            return

        # Merge buffer into main DataFrame
        df_new_data = pd.concat(self.buffer, ignore_index=True)
        self.buffer.clear()

        # Concurrent appending
        with self.lock:
            file_exists = self.path.exists() and (self.path.stat().st_size > 0)
            # If file does not exist or is empty, write header once
            df_new_data.to_csv(
                path_or_buf=self.path,
                mode="a",  # Append
                header=not file_exists,  # Write header only on first write
                index=False,
            )

    # ------------------------------------------------------------------------------------------------------------------
    def flush(self) -> None:
        """Force commit of any buffered rows."""

        self._commit()

    # ------------------------------------------------------------------------------------------------------------------
