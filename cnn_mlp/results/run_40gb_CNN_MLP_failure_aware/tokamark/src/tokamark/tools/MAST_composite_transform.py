"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import yaml
from typing import Union

from tokamark.tools.transforms.compose_transform import ComposeTransforms
from tokamark.tools.transforms.stdscale_transform import StdScalingTransform
from tokamark.tools.transforms.reshape_lcfs_transform import ReshapeLcfsTransform
from tokamark.tools.transforms.fill_profile_with_zeros_imputer_transform import FillProfileWithZerosTransform
from tokamark.tools.transforms.stft_transform import STFTTransform
from tokamark.tools.transforms.clip_non_physical_x_point_transform import ClipXPointTransform
from tokamark.tools.path import RANDOM_SPLIT_SIGNALS_STATS_FILE


# ----------------------------------------------------------------------------------------------------------------------
def build_common_signal_transform_map(
    source_signal_list: list[tuple],
    use_std_scaling: bool = True,
    use_nan_filling: bool = True,
    stats_metadata_file_path: str = RANDOM_SPLIT_SIGNALS_STATS_FILE,
) -> dict[str, ComposeTransforms]:
    """
    Build the signal transform map for each variable.

    Parameters
    ----------
    source_signal_list : list[tuple]
        List of source-signal tuples.
    use_std_scaling: bool
        If True, use STD scaling.
        Optional. Default: True.
    use_nan_filling: bool
        If True, use NaN filling.
        Optional. Default: True.
    stats_metadata_file_path : str
        Target path to the dict_stats_metadata YAML file.
        Optional. Default: `tokamark.tools.path.RANDOM_SPLIT_SIGNALS_STATS_FILE`.

    Returns
    -------
    dict[str, ComposeTransforms]
        Signal transform map for each variable.

    """

    # ..................................................................................................................
    def maybe_std(
        var: str,
    ) -> Union[list, list[StdScalingTransform]]:
        """
        Return [StdScalingTransform] if enabled, else empty list.

        Parameters
        ----------
        var : str
            Input variable.

        Returns
        -------
        Union[list, list[StdScalingTransform]]
            [<StdScalingTransform instance>] if enabled, else empty list.

        """

        if use_std_scaling:
            with open(stats_metadata_file_path, "r") as f:
                dict_stats_metadata = yaml.safe_load(f)

            return [StdScalingTransform(mean=dict_stats_metadata[var]["mean"], std=dict_stats_metadata[var]["std"])]

        return []

    # ..................................................................................................................
    def maybe_nan_filling() -> Union[list, list[FillProfileWithZerosTransform]]:
        """
        Return [FillProfileWithZerosTransform] if enabled, else empty list.

        Returns
        -------
        Union[list, list[FillProfileWithZerosTransform]]
            [<FillProfileWithZerosTransform instance>] if enabled, else empty list.

        """

        if use_nan_filling:
            return [FillProfileWithZerosTransform()]

        return []

    # ..................................................................................................................

    # Define base signal_transform_map

    signal_transform_map = {
        var: ComposeTransforms(maybe_std(var=var))
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }

    # Specific case of profiles with NaNs in full channel
    for var in [
        "magnetics-flux_loop_flux",
        "magnetics-b_field_pol_probe_ccbv_field",
        "magnetics-b_field_pol_probe_obr_field",
        "magnetics-b_field_pol_probe_obv_field",
        "magnetics-b_field_tor_probe_saddle_voltage",
        "thomson_scattering-t_e",
        "thomson_scattering-n_e",
    ]:
        signal_transform_map[var] = ComposeTransforms(transforms=maybe_std(var=var) + maybe_nan_filling())

    # Specific case of reformating LCFS
    for var in ["equilibrium-lcfs_r", "equilibrium-lcfs_z"]:
        signal_transform_map[var] = ComposeTransforms(
            transforms=maybe_std(var=var)
            + [
                ReshapeLcfsTransform(),
            ]
        )

    # Specific case of x point clipping of non-physical values
    for var in ["equilibrium-x_point_r", "equilibrium-x_point_z"]:
        signal_transform_map[var] = ComposeTransforms(
            transforms=[
                ClipXPointTransform(),
            ]
            + maybe_std(var=var)
        )

    # Specific case where we apply the FFT transform
    for var in ["magnetics-b_field_tor_probe_cc_field", "magnetics-b_field_pol_probe_omv_voltage"]:
        # Standardization must be done after the Fourier transform:
        # Mean and STD in the metadata have been computed for the FTT space
        signal_transform_map[var] = ComposeTransforms(
            transforms=[
                STFTTransform(support_n=512),
            ]
            + maybe_std(var=var)
        )

    return signal_transform_map

    # ..................................................................................................................
