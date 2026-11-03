"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import numpy as np
from collections.abc import Mapping
from typing import Any


# ======================================================================================================================
def stft(data: np.ndarray, times: np.ndarray, support_n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the short-time Fourier transform (STFT) of input data using input (sample) times.

    Parameters
    ----------
    data : np.ndarray
        Input 1D data array of shape (T,).
    times : np.ndarray
        Input 1D times array of shape (T,).
    support_n : int
        Number of support points for the STFT.
        Optional. Default: 512.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Pair (spectrum, frame_times), with spectrum being of shape (n_frames, support_n//2) and frame_times being of
        shape (n_frames, ).

    """

    xx = np.linspace(0, 2 * np.pi, support_n)
    window = (1 - np.cos(xx)) / 2

    nnp = int(len(data) / support_n)  # Number of non-overlapping windows

    if nnp == 0:
        # No full window fits -> return empty arrays
        return (
            np.zeros(shape=(0, support_n // 2), dtype=float),
            np.zeros(shape=(0,), dtype=float),
        )

    window = np.tile(window, nnp)

    data = data[: nnp * support_n] * window
    data = data.reshape(nnp, support_n)
    times = times[: nnp * support_n].reshape(nnp, support_n)

    # Simple frame time: mean time within each window
    frame_times = np.array([t[-1] for t in times])  # Shape: (n_frames,)

    spectrum = np.fft.fft(data)
    spectrum = spectrum[:, : support_n // 2]
    spectrum = np.abs(spectrum * np.conjugate(spectrum))  # Shape: (n_frames, support_n//2)

    return spectrum, frame_times


# ======================================================================================================================
class STFTTransform:
    """
    Class for the Short-time Fourier Transform (STFT) Transform.

    Attributes
    ----------
    support_n : int
        Number of support points for the STFT.

    Methods
    -------
    __call__(dict_)
        Call method for the class instances to behave like a function.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, support_n: int = 512) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        support_n : int
            Number of support points for the STFT.
            Optional. Default: 512.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors, as this is a callable class.

        """

        self.support_n = support_n

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, dict_: Mapping[str, Any]) -> dict[str, Any]:
        """
        Call method for the class instances to behave like a function.

        Parameters
        ----------
        dict_ : Mapping[str, Any]
            Dictionary with keys "time" and "values" (values shape: (C, T) or (T,))

        Returns
        -------
        dict[str, Any]
            Dictionary with "time" and STFT "values" (shape: (C, n_frames, support_n//2))

        """

        time = np.asarray(dict_["time"])
        values = np.asarray(dict_["values"])

        if values.ndim == 1:
            values = values[None, :]

        if time.shape[0] != values.shape[-1]:
            raise ValueError(f"Time length ({time.shape[0]}) does not match values length ({values.shape[-1]}).")

        spectra = []
        frame_times = None
        for ch in range(values.shape[0]):
            spectrum, frame_times = stft(data=values[ch], times=time, support_n=self.support_n)
            spectra.append(spectrum)

        if not spectra:
            spectrum_stack = np.zeros((0, 0, self.support_n // 2), dtype=float)
        else:
            spectrum_stack = np.stack(spectra, axis=0)

        return {"time": frame_times, "values": spectrum_stack.transpose(0, 2, 1)}

    # ------------------------------------------------------------------------------------------------------------------
