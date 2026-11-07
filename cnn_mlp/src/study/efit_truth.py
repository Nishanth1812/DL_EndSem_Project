#!/usr/bin/env python
"""EFIT ground truth for TokaMark task_1-3 windows (Module 1 v2).

TokaMark ships the EFIT reconstruction behind the psi target: magnetic axis, LCFS polygon, X-points, elongation and
triangularity (all in metres, on the zarr `equilibrium` group). This module loads them and aligns them with the
(shot_id, window_index) pairs that `TokaMarkDataset` emits, so a predicted psi can be scored against EFIT itself and not
against a flux-fraction heuristic.

Alignment is a re-implementation of TokaMarkDataset._process_shot/_build_window for task_1-3 (input 5 ms, output 5 ms,
delta -5 ms, stride 5 ms as set in pipeline.py, test_mode=True sparse filter, MastDataset ip_rating mask). It is checked
against the real dataset in `xpoint_diagnostic.py --prove-alignment` and in test_physics_evaluation.py.

Only numpy and zarr are needed (zarr 2 or 3). Arrays are [n_t, ...]; psi is [n_t, z_row, r_col].
"""
import os

import numpy as np

# the 8 input groups of task_1-3 with the nominal dt that TokaMark's metadata uses for padding (random_split_signals_stats)
INPUT_SIGNALS = (
    ("magnetics", "flux_loop_flux", 0.0002),
    ("magnetics", "b_field_pol_probe_ccbv_field", 0.0002),
    ("magnetics", "b_field_pol_probe_obr_field", 0.0002),
    ("magnetics", "b_field_pol_probe_obv_field", 0.0002),
    ("magnetics", "b_field_tor_probe_saddle_voltage", 0.00002),
    ("pf_active", "coil_current", 0.00025),
    ("pf_active", "solenoid_current", 0.00025),
    ("summary", "ip", 0.00025),
)
PSI_DT = 0.005
INPUT_LENGTH, OUTPUT_LENGTH, DELTA, STRIDE = 0.005, 0.005, -0.005, 0.005  # task_1-3 yaml + pipeline.py stride override


def open_shot(zarr_path):
    import zarr
    try:
        return zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    except TypeError:  # zarr 2
        return zarr.open_group(str(zarr_path), mode="r")


def _arr(g, path):
    try:
        return np.asarray(g[path][...])
    except KeyError:
        return None


def _time_of(g, group, name):
    """(time, values) of a signal; time is the dimension starting with 'time' (last axis), as TokaMark loads it."""
    try:
        a = g[f"{group}/{name}"]
    except KeyError:
        return np.array([]), np.array([])
    dims = a.attrs.get("_ARRAY_DIMENSIONS", [])
    td = [d for d in dims if str(d).startswith("time")]
    if not td:
        return np.array([]), np.array([])
    v = np.asarray(a[...])
    if v.ndim == 1:
        v = v[None, :]
    return np.asarray(g[f"{group}/{td[0]}"][...], dtype=np.float64), v


def geometry(g):
    """Grid of the psi map from the zarr: dict(r0, dr, z0, dz, n_r, n_z) in metres. Raises when the grid is not uniform."""
    R = np.asarray(g["equilibrium/major_radius"][...], dtype=np.float64)
    Z = np.asarray(g["equilibrium/z"][...], dtype=np.float64)
    dr, dz = np.diff(R), np.diff(Z)
    if not (np.allclose(dr, dr[0], rtol=1e-4) and np.allclose(dz, dz[0], rtol=1e-4)):
        raise ValueError("EFIT grid is not uniform")
    return dict(r0=float(R[0]), dr=float(dr[0]), z0=float(Z[0]), dz=float(dz[0]), n_r=len(R), n_z=len(Z))


def load_efit_truth(zarr_path):
    """All EFIT slices of one shot (metres, native EFIT times), or None when the shot has no equilibrium/psi.

    keys: shot_id, grid, t [n_t], psi [n_t, n_z, n_r] (raw Wb/rad, NaN where EFIT has no solution), axis_rz [n_t, 2],
    lcfs [list of n_t arrays (n, 2) in (R, Z) or None], xpoints [n_t, 2, 2] (slot, (R, Z); NaN = absent),
    kappa, kappa_axis, delta_upper, delta_lower, minor_radius, ip_rating [n_t].
    """
    g = open_shot(zarr_path)
    try:
        psi = np.asarray(g["equilibrium/psi"][...], dtype=np.float64)
    except KeyError:
        return None
    t = np.asarray(g["equilibrium/time"][...], dtype=np.float64)
    n_t = len(t)
    psi = np.moveaxis(psi, -1, 0)  # zarr dims are (z, major_radius, time); TokaMark moves time to the front
    lr, lz = _arr(g, "equilibrium/lcfs_r"), _arr(g, "equilibrium/lcfs_z")
    lcfs = []
    for i in range(n_t):
        if lr is None:
            lcfs.append(None)
            continue
        ok = np.isfinite(lr[:, i]) & np.isfinite(lz[:, i])
        lcfs.append(np.column_stack([lr[ok, i], lz[ok, i]]) if ok.sum() >= 8 else None)
    xr, xz = _arr(g, "equilibrium/x_point_r"), _arr(g, "equilibrium/x_point_z")
    xp = np.full((n_t, 2, 2), np.nan)
    if xr is not None:
        xp[:, :, 0], xp[:, :, 1] = xr.T[:, :2], xz.T[:, :2]

    def col(name):
        a = _arr(g, "equilibrium/" + name)
        return np.full(n_t, np.nan) if a is None else a.astype(np.float64)

    sid = os.path.basename(os.path.normpath(str(zarr_path)))
    sid = int(sid.split(".")[0]) if sid.split(".")[0].isdigit() else -1
    return dict(
        shot_id=sid, grid=geometry(g), t=t, psi=psi,
        axis_rz=np.column_stack([col("magnetic_axis_r"), col("magnetic_axis_z")]),
        lcfs=lcfs, xpoints=xp, kappa=col("elongation"), kappa_axis=col("elongation_axis"),
        delta_upper=col("triangularity_upper"), delta_lower=col("triangularity_lower"),
        minor_radius=col("minor_radius"), ip_rating=col("ip_rating"),
    )


def _pad_counts(t0, t_last, dt, t_start, t_end):
    """(n_left, n_right) pad samples of TokaMarkDataset._pad_time_series_to_interval."""
    n_l = int(np.ceil((t0 - t_start) / dt)) + 1 if t0 > t_start else 0
    n_r = int(np.ceil((t_end - t_last) / dt)) + 1 if t_last < t_end else 0
    return n_l, n_r


def enumerate_windows(zarr_path, outliers=(), truth=None):
    """List of dicts, one per window that TokaMarkDataset(test_mode=True) emits for this shot, in dataset order.

    Each has window_index (the dataset's idx_t, counting dropped windows too), t_cut, t_psi, slice_index (row of the EFIT
    arrays that is the psi target). Windows the dataset drops (psi slice all NaN, or every input group all-NaN) are
    absent. Returns [] for a shot without equilibrium, like the dataset (which cannot find an output).
    """
    truth = truth or load_efit_truth(zarr_path)
    if truth is None or "equilibrium-psi" in outliers:
        return []
    g = open_shot(zarr_path)
    # MastDataset: remove_bad_efit_rating masks psi where ip_rating == 0 (and drops all equilibrium signals if no ip_rating)
    ipr = truth["ip_rating"]
    if np.all(np.isnan(ipr)) and _arr(g, "equilibrium/ip_rating") is None:
        return []
    psi = truth["psi"].copy()
    psi[ipr == 0] = np.nan
    t_psi = truth["t"]
    psi_ok = ~np.all(np.isnan(psi), axis=(1, 2))  # per-slice "has data"
    if not psi_ok.any():
        return []
    sig = []  # (times, has_data[T], dt) per input group; empty groups stay empty (all-NaN windows)
    for grp, name, dt in INPUT_SIGNALS:
        if f"{grp}-{name}" in outliers:
            sig.append((np.array([]), np.array([], bool), dt))
            continue
        tt, vv = _time_of(g, grp, name)
        if tt.size == 0 or vv.size == 0:
            sig.append((np.array([]), np.array([], bool), dt))
            continue
        sig.append((tt, ~np.all(np.isnan(vv), axis=tuple(range(vv.ndim - 1))), dt))
    starts = [tt[ok][0] for tt, ok, _ in sig if tt.size and ok.any()]
    if not starts:
        return []
    t_start = float(np.min(starts))
    t_end = float(t_psi[psi_ok][-1])
    t_cut_start = t_start + INPUT_LENGTH
    t_cut_stop = t_end - DELTA - OUTPUT_LENGTH
    if t_cut_start >= t_cut_stop:
        return []

    def padded(tt, ok, dt):
        if tt.size == 0:
            return tt, ok, 0
        n_l, n_r = _pad_counts(tt[0], tt[-1], dt, t_start, t_end)
        t_all = tt
        if n_l:
            t_all = np.concatenate([tt[0] - dt * np.arange(n_l, 0, -1), t_all])
        if n_r:
            t_all = np.concatenate([t_all, tt[-1] + dt * np.arange(1, n_r + 1)])
        return t_all, np.concatenate([np.zeros(n_l, bool), ok, np.zeros(n_r, bool)]), n_l

    sig = [padded(*s) + (s[2],) for s in sig]  # (t_pad, ok_pad, n_left, dt)
    t_psi_pad, ok_pad, n_left = padded(t_psi, psi_ok, PSI_DT)
    out = []
    for idx_t, t_cut in enumerate(np.arange(t_cut_start, stop=t_cut_stop, step=STRIDE)):
        s0 = int(round((t_cut + DELTA - t_psi_pad[0]) / PSI_DT))
        s0 = int(np.clip(s0, 0, len(t_psi_pad) - 1))
        if not ok_pad[s0]:
            continue  # output all-NaN -> dropped
        any_input = False
        for tt, ok, _, dt in sig:
            if tt.size == 0:
                continue
            end = int(round((t_cut - tt[0]) / dt))
            n_in = int(round(INPUT_LENGTH / dt))
            ids = np.clip(np.arange(end - n_in, end), 0, len(tt) - 1)
            if ok[ids].any():
                any_input = True
                break
        if not any_input:
            continue
        j = s0 - n_left
        out.append(dict(window_index=idx_t, t_cut=float(t_cut), t_psi=float(t_psi[j]), slice_index=j))
    return out


def window_truth(truth, w):
    """Truth dict of one window: the EFIT slice `w['slice_index']`, plus window bookkeeping. Units: metres, raw psi."""
    j = w["slice_index"]
    xp = truth["xpoints"][j]
    xp = xp[np.all(np.isfinite(xp), axis=1)]
    return dict(
        shot_id=truth["shot_id"], window_index=w["window_index"], t_cut=w["t_cut"], t_psi=w["t_psi"], slice_index=j,
        grid=truth["grid"], psi_raw=truth["psi"][j], axis_rz_m=truth["axis_rz"][j], lcfs_rz_m=truth["lcfs"][j],
        xpoints_rz_m=xp, n_xpoints=int(len(xp)), kappa=float(truth["kappa"][j]), kappa_axis=float(truth["kappa_axis"][j]),
        delta_upper=float(truth["delta_upper"][j]), delta_lower=float(truth["delta_lower"][j]),
        delta=float(np.nanmean([truth["delta_upper"][j], truth["delta_lower"][j]])) if np.isfinite(
            [truth["delta_upper"][j], truth["delta_lower"][j]]).any() else float("nan"),
        minor_radius_m=float(truth["minor_radius"][j]), ip_rating=float(truth["ip_rating"][j]),
    )


class TruthIndex:
    """EFIT truth for every window of a directory of `<shot>.zarr` stores.

        ti = TruthIndex(data_dir)                    # data_dir/<shot_id>.zarr
        tr = ti.get(shot_id, window_index)           # dict (see window_truth) or None if the dataset never emits it
        tr = ti.by_tcut(shot_id, t_cut)              # same, keyed by the window's t_cut
        evaluate_window_physics(yt, yp, truth=tr)

    `outlier_yaml` (MAST_tools/metadata/<split>_split_outlier_metadata.yaml, or env TM_OUTLIER_YAML) replaces flagged
    signals by empty ones exactly like MastDataset(remove_outliers=True); without it no signal is treated as an outlier.
    """

    def __init__(self, data_dir, outlier_yaml=None):
        self.data_dir = str(data_dir)
        self._outliers = {}
        path = outlier_yaml or os.environ.get("TM_OUTLIER_YAML")
        if path and os.path.exists(path):
            import yaml
            self._outliers = {int(k): set(v or []) for k, v in (yaml.safe_load(open(path)) or {}).items()}
        self._cache = {}

    def shot_path(self, shot_id):
        return os.path.join(self.data_dir, f"{int(shot_id)}.zarr")

    def shot_ids(self):
        return sorted(int(f[:-5]) for f in os.listdir(self.data_dir) if f.endswith(".zarr") and f[:-5].isdigit())

    def windows(self, shot_id):
        """{window_index: truth dict} of one shot ({} if the shot has no equilibrium or the dataset drops it)."""
        sid = int(shot_id)
        if sid not in self._cache:
            p = self.shot_path(sid)
            tr = load_efit_truth(p) if os.path.isdir(p) else None
            ws = enumerate_windows(p, outliers=self._outliers.get(sid, ()), truth=tr) if tr is not None else []
            self._cache[sid] = {w["window_index"]: window_truth(tr, w) for w in ws}
        return self._cache[sid]

    def get(self, shot_id, window_index, default=None):
        return self.windows(shot_id).get(int(window_index), default)

    def by_tcut(self, shot_id, t_cut, tol=1e-6, default=None):
        for w in self.windows(shot_id).values():
            if abs(w["t_cut"] - float(t_cut)) <= tol:
                return w
        return default


# MastDataset standardises psi per signal with the split's global mean / std (tokamark/metadata/<split>_split_signals_stats.yaml)
PSI_STATS = {"random": (-0.026102678542290144, 0.054930705723548384), "temporal": (-0.02704797707091641, 0.06492857922672353)}


def check_alignment(rows, data_dir, split="random", outlier_yaml=None, atol=1e-9):
    """Compare real TokaMarkDataset windows with this module's index.

    rows: [(shot_id, window_index, y)] where y is the 65x65 target the dataset emitted (standardised psi). For every shot in
    rows the set of window indices must equal TruthIndex's, and y must equal (psi_raw - mean) / std of the slice it names.
    Returns (n_windows_matching, problems) with problems a list of strings (empty when aligned).
    """
    mean, std = PSI_STATS[split]
    ti = TruthIndex(data_dir, outlier_yaml=outlier_yaml)
    by_shot = {}
    for sid, wi, y in rows:
        by_shot.setdefault(int(sid), {})[int(wi)] = np.asarray(y)
    ok, problems = 0, []
    for sid, ds in by_shot.items():
        mine = ti.windows(sid)
        if set(ds) != set(mine):
            problems.append(f"shot {sid}: dataset windows {sorted(set(ds) - set(mine))} not indexed, indexed {sorted(set(mine) - set(ds))} not emitted")
        for wi in sorted(set(ds) & set(mine)):
            ref = (mine[wi]["psi_raw"] - mean) / std
            if np.allclose(ds[wi], ref, atol=atol, rtol=0, equal_nan=True):
                ok += 1
            else:
                problems.append(f"shot {sid} window {wi}: max|dy| = {np.nanmax(np.abs(ds[wi] - ref)):.3g}")
    return ok, problems
