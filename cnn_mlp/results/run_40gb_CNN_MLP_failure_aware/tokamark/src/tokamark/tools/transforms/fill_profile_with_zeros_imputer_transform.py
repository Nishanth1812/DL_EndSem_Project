"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import numpy as np
from collections.abc import Mapping
from typing import Any


# ======================================================================================================================
class FillProfileWithZerosTransform:
    """
    Transform to fill profile with zeros.

    Attributes
    ----------
    None.

    Methods
    -------

    __call__(dict_)
        Call method for the class instances to behave like a function.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, dict_: Mapping[str, Any]) -> dict[str, Any]:
        """
        Call method for the class instances to behave like a function.

        Parameters
        ----------
        dict_ : Mapping[str, Any]
            Torch dict with "time" and "key" keys and corresponding values.

        Returns
        -------
        dict[str, Any]
            Torch dict with "time" and "key" keys where the NaNs of profiles (i.e., when one full channel in the
            profile is missing) are filled with zeros.

        """

        time = dict_["time"]
        values = dict_["values"].copy()

        # Checking for indexes with NaN values
        nan_ind = np.isnan(values)

        # Excluding columns comprised of NaNs only
        nan_cols = np.isnan(values).all(axis=0)
        nan_ind[:, nan_cols] = False

        # Replacing NaNs in profile components
        values[nan_ind] = 0

        return {"time": time, "values": values}

    # ------------------------------------------------------------------------------------------------------------------
