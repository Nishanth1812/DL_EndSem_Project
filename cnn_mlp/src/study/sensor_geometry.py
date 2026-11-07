#!/usr/bin/env python
"""Real MAST sensor geometry for the 115 TokaMark task_1-3 inputs (data: geometry/mast_sensor_geometry.json).

Provenance is stored per sensor in the data file. Source: FAIR-MAST level-2 zarr (shot 30421) geometry arrays, ingested from
the MAST EFIT++ XML configs; PF coil centroids cross-checked against FreeGS `MAST()` (see cross_check_coils).
Sensor order == sensor_attribution.default_inventory() order. Ip has no position (R, Z = NaN). Saddle loops have a position but
measure a voltage (dPhi/dt), so they have no static axisymmetric Green's response.
The 65 x 65 grid axes are PARAMETERS (defaults = raw zarr equilibrium/major_radius 0.06..1.98 m and equilibrium/z -2..2 m,
psi indexed [Z, R]); pass the re-derived axes when they exist.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEOMETRY_JSON = os.path.join(HERE, "geometry", "mast_sensor_geometry.json")
DEFAULT_R_AXIS = np.linspace(0.06, 1.98, 65)
DEFAULT_Z_AXIS = np.linspace(-2.0, 2.0, 65)
# FLUX_LOOP_COORDS in cnn_mlp/src/pipeline.py (r_idx, z_idx), copied here for the comparison only
PIPELINE_FLUX_LOOP_COORDS = ((2, 13), (2, 22), (2, 32), (2, 42), (2, 51), (60, 6), (60, 13), (60, 19), (60, 26), (60, 32),
                             (60, 38), (60, 45), (60, 51), (60, 58), (60, 62))


@dataclass
class SensorGeometry:
    names: list
    group: list
    kind: list
    quantity: list
    R: np.ndarray
    Z: np.ndarray
    raw: dict

    def idx(self, name):
        return self.names.index(name)

    def group_indices(self, g):
        return [i for i, x in enumerate(self.group) if x == g]

    def pixel_positions(self, R_axis=DEFAULT_R_AXIS, Z_axis=DEFAULT_Z_AXIS):
        """Fractional (r_idx, z_idx) of each sensor on the grid (NaN where no position)."""
        r = np.interp(self.R, R_axis, np.arange(len(R_axis)), left=np.nan, right=np.nan)
        z = np.interp(self.Z, Z_axis, np.arange(len(Z_axis)), left=np.nan, right=np.nan)
        return r, z

    @property
    def limiter(self):
        return np.array(self.raw["limiter"]["R"]), np.array(self.raw["limiter"]["Z"])

    def coil_elements(self, coil_name):
        """(R, Z) arrays of the rectangular turn elements of a PF coil ('P2IL'.. 'P5U') or 'SOL'."""
        e = self.raw["solenoid_elements"] if coil_name == "SOL" else self.raw["pf_coil_elements"][coil_name]["elements"]
        return np.array([x["R"] for x in e]), np.array([x["Z"] for x in e])


def load_geometry(path: str = GEOMETRY_JSON) -> SensorGeometry:
    d = json.load(open(path, encoding="utf8"))
    s = d["sensors"]
    nan = lambda v: np.nan if v is None else float(v)  # noqa: E731
    return SensorGeometry([x["name"] for x in s], [x["group"] for x in s], [x["kind"] for x in s], [x["quantity"] for x in s],
                          np.array([nan(x["R"]) for x in s]), np.array([nan(x["Z"]) for x in s]), d)


def cross_check_coils(geo: SensorGeometry) -> dict:
    """Distance (m) between the FAIR-MAST element centroid of each PF coil and the FreeGS MAST() coil position (P2 inner/outer
    both compared to the single FreeGS P2 position, so those carry a physical offset)."""
    fg = geo.raw["cross_check_FreeGS"]["coils"]
    out = {}
    for ch in ("P2IL", "P2IU", "P2OL", "P2OU", "P3L", "P3U", "P4L", "P4U", "P5L", "P5U"):
        c = geo.raw["pf_coil_elements"][ch]
        ref = fg[ch.replace("I", "").replace("O", "")] if ch.startswith("P2") else fg[ch]
        out[ch] = float(np.hypot(c["R_centroid"] - ref[0], c["Z_centroid"] - ref[1]))
    return out


def compare_pipeline_flux_loops(geo: SensorGeometry, R_axis=DEFAULT_R_AXIS, Z_axis=DEFAULT_Z_AXIS,
                                coords=PIPELINE_FLUX_LOOP_COORDS) -> dict:
    """For each hard-coded pipeline flux-loop pixel: its (R, Z) in metres and the distance to the NEAREST real flux loop
    (and to the same-index real loop from the dataset channel list, the pairing the PINN loss implicitly assumes)."""
    fl = geo.group_indices("flux_loop")
    RR, ZZ = geo.R[fl], geo.Z[fl]
    rows = []
    for k, (ri, zi) in enumerate(coords):
        R, Z = float(R_axis[ri]), float(Z_axis[zi])
        d = np.hypot(RR - R, ZZ - Z)
        rows.append(dict(index=k, R=R, Z=Z, nearest=geo.names[fl[int(d.argmin())]], d_nearest_m=float(d.min()),
                         d_same_index_m=float(d[k])))
    return dict(rows=rows, median_d_nearest_m=float(np.median([r["d_nearest_m"] for r in rows])),
                max_d_nearest_m=float(max(r["d_nearest_m"] for r in rows)),
                median_d_same_index_m=float(np.median([r["d_same_index_m"] for r in rows])),
                grid_dR_m=float(np.diff(R_axis).mean()), grid_dZ_m=float(np.diff(Z_axis).mean()))
