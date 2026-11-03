"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

from collections.abc import Callable
from typing import Any, Optional


# ======================================================================================================================
class ComposeTransforms(object):
    """
    Compose transforms and apply them in series checking for None return values.

    Attributes
    ----------
    transforms : list[callable[tuple]]
        List containing the names of the transforms

    Methods
    -------
    __call__(sample)
        Call method for the class instances to behave like a function.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, transforms: list[Callable]) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        transforms : list[Callable]
            List containing the names of the transforms.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors, as this is a callable class.

        """

        self.transforms = transforms

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, sample: Optional[Any]) -> Optional[Any]:
        """
        Call method for the class instances to behave like a function.

        Parameters
        ----------
        sample : Optional[Any]
            Input sample.

        Returns
        -------
        Optional[Any]
            Transformed sample.

        """

        for transform in self.transforms:
            if sample is None:
                return None
            sample = transform(sample)

        return sample

    # ------------------------------------------------------------------------------------------------------------------
