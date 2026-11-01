"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

from collections.abc import Mapping
from typing import Any


# ======================================================================================================================
class StdScalingTransform:
    """
    STD scaling transform.

    Methods
    -------
    __call__(dict_)
        Call method for the class instances to behave like a function. It normalizes each sample individually: subtract
        mean, divide by STD.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, mean: float, std: float) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        mean : float
            Input mean.
        std : float
            Input STD.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors, as this is a callable class.

        """

        self.mean = mean
        self.std = std

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, dict_: Mapping[str, Any]) -> dict[str, Any]:
        """
        Call method for the class instances to behave like a function. It normalizes each sample individually: subtract
        mean, divide by STD.

        Parameters
        ----------
        dict_ : Mapping[str, Any]
            Dictionary with "time" and "values" keys with corresponding values.

        Returns
        -------
        dict[str, Any]
            Augmented input dictionary with values normalized per feature.

        """

        values = dict_["values"]
        if values is not None:
            values = (values - self.mean) / self.std

        return {"time": dict_["time"], "values": values}

    # ------------------------------------------------------------------------------------------------------------------
