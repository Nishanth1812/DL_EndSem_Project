"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import numpy as np
from collections.abc import Mapping
from typing import Any


# ======================================================================================================================
class ClipXPointTransform:
    """
    Transform to clip non-physical x points.

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
            Torch dict with "time" and "key" keys, where non-physical timestamps have been replaced with NaN values.

        """

        time = dict_["time"]
        values = dict_["values"]

        # Replace values > 2 or < -2 with NaN
        values = np.where((values > 2) | (values < -2), np.nan, values)

        return {"time": time, "values": values}

    # ------------------------------------------------------------------------------------------------------------------
