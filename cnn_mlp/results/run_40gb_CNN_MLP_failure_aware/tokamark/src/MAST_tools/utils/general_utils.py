"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import random
import string
from urllib.error import HTTPError
import pandas as pd


# ----------------------------------------------------------------------------------------------------------------------
def get_random_string(n=4):
    """
    Get random alphanumeric string of a given length.

    Parameters
    ----------
    n : int
        Length of target string.

    Returns
    -------
    str

    """

    return "".join(random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for _ in range(n))


# ----------------------------------------------------------------------------------------------------------------------
def _read_parquet_data(path: str) -> pd.DataFrame:
    """
    Read MAST data from a target file path using parquet pipeline.

    Parameters
    ----------
    path : str
        Target file path.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with the target parquet data.

    Raises
    ------
    FileNotFoundError
       If no parquet data is available for the provided path.

    """

    try:
        return pd.read_parquet(path=path)
    except HTTPError as ee:
        raise FileNotFoundError(f"No data available for path {path} ({ee}).")
    except FileNotFoundError as ee:
        raise FileNotFoundError(f"No data available for path {path} ({ee}).")


# ----------------------------------------------------------------------------------------------------------------------
def warning_print(input_string: str, prefix: str = "[WARNING] "):
    """Print warning text."""
    print(f"\033[93m{prefix}{input_string}\033[0m")
