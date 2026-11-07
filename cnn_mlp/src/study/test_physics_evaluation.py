#!/usr/bin/env python
"""Self-checks for Module 1 v2. Synthetic known-answer tests always run; real-shot tests need TokaMark zarr stores and are
skipped (printed as SKIP) when none is found. Point TM_ZARR_DIR at a folder of <shot>.zarr stores that include the
equilibrium group. Run: python test_physics_evaluation.py   (or pytest)."""
import glob
import os
import sys
from unittest import SkipTest  # pytest reports it as a skip

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(os.environ.get("LOCALAPPDATA", ""), "Temp", "claude", "94cae6ef419a3dd0", "9b3dc88f-3a2c-4815-bbad-6ee91700a5da",
                             "scratchpad", "pylib"))  # dev machine only: where zarr lives; harmless elsewhere
import physics_evaluation as pe
import efit_truth as et

R_CM = pe.GEOM.r0 + np.arange(pe.GRID_SIZE) * pe.GEOM.dr
Z_CM = pe.GEOM.z0 + np.arange(pe.GRID_SIZE) * pe.GEOM.dz
RR, ZZ = np.meshgrid(R_CM, Z_CM)  # arrays are indexed [z_row, r_col], like the model output
GRID_M = dict(r0=0.06, dr=0.03, z0=-2.0, dz=0.0625)


def ellipse(kappa, a=30.0, r0=110.0, z0=0.0, bend=0.0):
    """Peaked field whose contours are ellipses of elongation `kappa`; `bend` shifts R with Z squared (D shape)."""
    rc = r0 + bend * (ZZ - z0) ** 2
    return np.exp(-(((RR - rc) / a) ** 2 + ((ZZ - z0) / (kappa * a)) ** 2))


def ellipse_truth(kappa, a=30.0, r0=110.0, z0=0.0, level=0.4):
    """EFIT-like truth dict for ellipse(): LCFS = the exact contour psi = level (semi-axes A and kappa * A)."""
    A = a * np.sqrt(-np.log(level))
    t = np.linspace(0, 2 * np.pi, 147)
    lcfs = np.column_stack([(r0 + A * np.cos(t)), (z0 + kappa * A * np.sin(t))]) / 100.0
    return dict(grid=GRID_M, axis_rz_m=np.array([r0, z0]) / 100.0, lcfs_rz_m=lcfs, xpoints_rz_m=np.zeros((0, 2)),
                kappa=float(kappa), delta=0.0, delta_upper=0.0, delta_lower=0.0)


# --------------------------------------------------------------------------- synthetic
def test_geometry_constants():
    assert pe.DR_CM == 3.0 and pe.DZ_CM == 6.25, (pe.DR_CM, pe.DZ_CM)
    assert abs(pe.GEOM.r0 + 64 * pe.GEOM.dr - 198.0) < 1e-9 and abs(pe.GEOM.z0 + 64 * pe.GEOM.dz - 200.0) < 1e-9


def test_identical_fields_give_zero_axis_shift():
    # peak exactly on the half boundary row (32) used to be distorted by the masked sub-pixel step (2.3 cm for identical fields)
    for z0 in (0.0, 0.5 * pe.GEOM.dz, -3.0, 40.0):
        f = ellipse(1.6, z0=z0)
        for mode in ("heuristic", "legacy"):
            assert pe.compute_axis_shift(f, f)[0] < 1e-9, (z0, mode)
    f = ellipse(1.6)
    assert pe.evaluate_window_physics(f, f)["axis_shift_cm"] < 1e-9
    # a real half-pixel shift is still measured: R centre moved by +3 cm = one column
    assert abs(pe.compute_axis_shift(ellipse(1.6), ellipse(1.6, r0=113.0))[0] - 3.0) < 0.1


def test_elongation_and_triangularity():
    for kappa in (1.0, 1.6, 2.0):
        k, d, kp, dp = pe.compute_shape_error(ellipse(kappa), ellipse(kappa))
        assert abs(k - kappa) < 0.08, (kappa, k)
        assert abs(d) < 0.05, d
        assert kp == k and dp == d
    k, d, kp, dp = pe.compute_shape_error(ellipse(1.5), ellipse(1.8))
    assert 0.2 < kp - k < 0.4 and abs(dp - d) < 0.05, (k, kp, d, dp)
    k0, d0, _, _ = pe.compute_shape_error(ellipse(1.6), ellipse(1.6))
    _, db, _, _ = pe.compute_shape_error(ellipse(1.6, bend=2e-3), ellipse(1.6, bend=2e-3))
    assert db < d0 - 0.05, (d0, db)
    # EFIT definitions on a hand-made shape: R in 80..140, Z in -90..+90, top at R = 125, bottom at R = 95
    poly = np.array([[125, 90], [140, 0], [95, -90], [80, 0], [125, 90.0]])
    s = pe.shape_parameters(poly)
    assert abs(s["kappa"] - 180 / 60) < 1e-9 and abs(s["a_cm"] - 30) < 1e-9 and abs(s["r_geo_cm"] - 110) < 1e-9
    assert abs(s["delta_upper"] - (110 - 125) / 30) < 1e-9 and abs(s["delta_lower"] - (110 - 95) / 30) < 1e-9


def two_peaks():
    g = lambda z0, h: h * np.exp(-(((RR - 110.0) / 35.0) ** 2 + ((ZZ - z0) / 45.0) ** 2))
    return g(30.0, 1.0) + g(-90.0, 0.55)


def test_xpoint_found_and_not_invented():
    f = two_peaks()
    col = np.argmin(np.abs(R_CM - 110.0))
    sub = (Z_CM > -85) & (Z_CM < 25)
    z_true = Z_CM[sub][np.argmin(f[sub, col])]
    psi_x = f[sub, col].min()
    xs = pe.find_xpoints(f, level=psi_x, side="lower")
    assert xs, "saddle missed"
    r_px, z_px = xs[0][:2]
    assert abs(pe.GEOM.r0 + r_px * pe.GEOM.dr - 110.0) < pe.GEOM.dr, r_px
    assert abs(pe.GEOM.z0 + z_px * pe.GEOM.dz - z_true) < 1.5 * pe.GEOM.dz, (z_px, z_true)
    single = ellipse(1.6)
    assert pe.find_xpoints(single, level=pe.lcfs_level(single), side="lower") == []
    assert pe.find_xpoints(single, level=0.4, psi_tol=pe.XPOINT_PSI_TOL_EFIT, side="both") == []
    assert pe.find_divertor_xpoint(single, level=pe.lcfs_level(single)) is None


def test_efit_mode_known_answers():
    kappa = 1.7
    f, tr = ellipse(kappa), ellipse_truth(kappa)
    r = pe.evaluate_window_physics(f, f, truth=tr)
    assert r["mode"] == "efit" and r["efit_lcfs_available"] and r["boundary_closed"]
    assert abs(r["psi_b"] - 0.4) < 0.01 and r["psi_b_spread_pct"] < 0.5, (r["psi_b"], r["psi_b_spread_pct"])
    assert r["boundary_mean_cm"] < 0.5 and r["axis_shift_cm"] < 0.5, r
    assert abs(r["kappa_pred"] - kappa) < 0.05 and abs(r["delta_pred"]) < 0.05, r
    assert not r["xpoint_true_found"] and not r["xpoint_pred_found"]
    # prediction displaced by exactly one column (3 cm): axis shift 3 cm, Hausdorff boundary error 3 cm
    r = pe.evaluate_window_physics(f, ellipse(kappa, r0=113.0), truth=tr)
    assert abs(r["axis_shift_cm"] - 3.0) < 0.3, r["axis_shift_cm"]
    assert abs(r["boundary_hausdorff_cm"] - 3.0) < 0.5 and 1.5 < r["boundary_mean_cm"] < 3.0, r
    # a prediction that is 20% too tall shows in kappa
    r = pe.evaluate_window_physics(f, ellipse(kappa * 1.2), truth=tr)
    assert 0.25 < r["kappa_pred"] - kappa < 0.45 and r["kappa_err"] > 0.25, r
    # a prediction without any contour at psi_b is penalised and counted, not dropped from the mean
    r = pe.evaluate_window_physics(f, np.zeros_like(f), truth=tr)
    assert r["boundary_failed"] and r["boundary_mean_cm"] == pe.BOUNDARY_FAIL_CM and not r["boundary_closed"]
    agg = pe.aggregate_physics_metrics([r, pe.evaluate_window_physics(f, f, truth=tr)])
    assert agg["boundary_fail_rate_pct"] == 50.0 and agg["boundary_mean_cm_valid_only"] < 0.5 and agg["boundary_mean_cm"] > 20
    # truth without an EFIT LCFS: no boundary score (nan), no invented number, but the axis is still scored
    tr2 = dict(tr, lcfs_rz_m=None)
    r = pe.evaluate_window_physics(f, f, truth=tr2)
    assert not r["efit_lcfs_available"] and np.isnan(r["boundary_mean_cm"]) and r["axis_shift_cm"] < 0.5
    agg = pe.aggregate_physics_metrics([r, pe.evaluate_window_physics(f, f, truth=tr)])
    assert agg["n_windows"] == 2 and agg["n_boundary_valid"] == 1


def test_efit_mode_xpoint():
    f = two_peaks()
    col = np.argmin(np.abs(R_CM - 110.0))
    sub = (Z_CM > -85) & (Z_CM < 25)
    z_x = Z_CM[sub][np.argmin(f[sub, col])]
    psi_x = float(f[sub, col].min())
    rng = f.max() - f.min()
    axis_px = ((110.0 - pe.GEOM.r0) / pe.GEOM.dr, (30.0 - pe.GEOM.z0) / pe.GEOM.dz)
    loop = pe.loop_around(f, psi_x + 0.02 * rng, axis_px)
    assert loop is not None
    lcfs_m = pe._to_cm(loop, pe.GEOM) / 100.0
    tr = dict(grid=GRID_M, axis_rz_m=np.array([1.10, 0.30]), lcfs_rz_m=lcfs_m, xpoints_rz_m=np.array([[1.10, z_x / 100.0]]),
              kappa=1.5, delta=0.0, delta_upper=0.0, delta_lower=0.0)
    r = pe.evaluate_window_physics(f, f, truth=tr)
    assert r["xpoint_true_found"] and r["xpoint_pred_found"] and r["xpoint_err_cm"] < 1.0, r
    # a prediction with no second peak has no saddle: missed
    r = pe.evaluate_window_physics(f, ellipse(1.5, z0=30.0, a=35.0), truth=tr)
    assert r["xpoint_true_found"] and not r["xpoint_pred_found"] and np.isnan(r["xpoint_err_cm"]), r
    agg = pe.aggregate_physics_metrics([pe.evaluate_window_physics(f, f, truth=tr), r])
    assert agg["xpoint_detect_rate_pct"] == 50.0


def test_modes_and_legacy_scale():
    f, p = ellipse(1.6), ellipse(1.6, r0=113.0)
    h = pe.evaluate_window_physics(f, p)
    l = pe.evaluate_window_physics(f, p, mode="legacy")
    assert h["mode"] == "heuristic" and l["mode"] == "legacy"
    assert abs(h["axis_shift_cm"] - 3.0) < 0.1
    # legacy converts the same pixel shift (R 3 cm = 1 px) with the old 2.8125 cm/px
    assert abs(l["axis_shift_cm"] / h["axis_shift_cm"] - pe.LEGACY_DR_CM / pe.DR_CM) < 0.05, (l["axis_shift_cm"], h["axis_shift_cm"])
    try:
        pe.evaluate_window_physics(f, p, mode="efit")
        raise AssertionError("mode='efit' without truth must raise")
    except ValueError:
        pass


def test_window_record_and_aggregate():
    f = ellipse(1.6)
    rec = pe.evaluate_window_physics(f, f)
    assert rec["boundary_mean_cm"] < 0.1 and rec["axis_shift_cm"] < 0.1
    assert not rec["xpoint_true_found"] and np.isnan(rec["xpoint_err_cm"])
    agg = pe.aggregate_physics_metrics([rec, rec])
    assert agg["xpoint_true_count"] == 0 and np.isnan(agg["xpoint_detect_rate_pct"])
    assert agg["xpoint_false_rate_pct"] == 0.0 and agg["shape_valid_n"] == 2 and agg["kappa_err"] < 1e-6
    base = dict(rec)
    hit = dict(base, xpoint_true_found=True, xpoint_pred_found=True, xpoint_err_cm=3.0)
    miss = dict(base, xpoint_true_found=True, xpoint_pred_found=False, xpoint_err_cm=float("nan"))
    fp = dict(base, xpoint_true_found=False, xpoint_pred_found=True, xpoint_err_cm=float("nan"))
    agg = pe.aggregate_physics_metrics([hit, miss, fp])
    assert agg["xpoint_detect_rate_pct"] == 50.0 and agg["xpoint_false_rate_pct"] == 100.0, agg
    assert agg["xpoint_err_cm"] == 3.0 and agg["xpoint_err_n"] == 1


def test_wall_proxy():
    box = pe.WALL_BOX_CM
    c = np.array([[box[0] + 4.0, 0.0], [100.0, 10.0]])
    assert abs(pe.compute_wall_proximity(c) - 4.0) < 1e-9
    assert pe.compute_wall_proximity(None) == 50.0


# --------------------------------------------------------------------------- real shots (skipped without data)
def _zarr_dir():
    cands = [os.environ.get("TM_ZARR_DIR", ""), os.path.join(HERE, "..", "..", "data", "random"), os.path.join(HERE, "..", "..", "..", "data", "random"),
             os.path.join(os.environ.get("LOCALAPPDATA", ""), "Temp", "claude", "94cae6ef419a3dd0", "9b3dc88f-3a2c-4815-bbad-6ee91700a5da",
                          "scratchpad", "data", "random")]
    for d in cands:
        if d and os.path.isdir(d):
            try:
                import zarr  # noqa: F401
            except Exception:
                return None
            if glob.glob(os.path.join(d, "*.zarr", "equilibrium", "psi")):
                return d
    return None


def _real_shots(limit=4):
    d = _zarr_dir()
    if d is None:
        return None, []
    return d, sorted(int(os.path.basename(os.path.dirname(os.path.dirname(p)))[:-5])
                     for p in glob.glob(os.path.join(d, "*.zarr", "equilibrium", "psi")))[:limit]


def test_geometry_matches_zarr():
    d, shots = _real_shots()
    if not shots:
        raise SkipTest("no TokaMark zarr data with an equilibrium group (set TM_ZARR_DIR)")
    for s in shots:
        g = et.geometry(et.open_shot(os.path.join(d, f"{s}.zarr")))
        assert abs(g["r0"] - pe.R_MIN_M) < 1e-6 and abs(g["dr"] * 100 - pe.DR_CM) < 1e-6, g
        assert abs(g["z0"] - pe.Z_MIN_M) < 1e-6 and abs(g["dz"] * 100 - pe.DZ_CM) < 1e-6, g
        assert g["n_r"] == g["n_z"] == pe.GRID_SIZE


def test_real_orientation_and_efit_consistency():
    """psi is [z_row, r_col]: along the EFIT LCFS the flux is constant only in that orientation; shape function = EFIT's."""
    d, shots = _real_shots()
    if not shots:
        raise SkipTest("no TokaMark zarr data with an equilibrium group (set TM_ZARR_DIR)")
    n = 0
    for s in shots:
        ti = et.TruthIndex(d)
        for wi, tr in ti.windows(s).items():
            if tr["lcfs_rz_m"] is None:
                continue
            g = tr["grid"]
            lc_px = pe._px_of(tr["lcfs_rz_m"], g)
            _, sp_ok = pe.efit_psi_b(tr["psi_raw"], lc_px)
            _, sp_t = pe.efit_psi_b(tr["psi_raw"].T, lc_px)
            assert sp_ok < 0.2 and sp_t > 5 * sp_ok + 1.0, (s, wi, sp_ok, sp_t)
            sh = pe.shape_parameters(tr["lcfs_rz_m"] * 100.0)
            assert abs(sh["kappa"] - tr["kappa"]) < 1e-3 and abs(sh["a_cm"] - tr["minor_radius_m"] * 100) < 0.05, (s, wi)
            assert abs(sh["delta_upper"] - tr["delta_upper"]) < 1e-3 and abs(sh["delta_lower"] - tr["delta_lower"]) < 1e-3, (s, wi)
            rec = pe.evaluate_window_physics(tr["psi_raw"], tr["psi_raw"], truth=tr)
            assert rec["boundary_mean_cm"] < 1.0 and rec["axis_shift_cm"] < 1.0 and rec["boundary_closed"], (s, wi, rec)
            n += 1
            if n >= 12:
                return
    assert n > 0, "no window with an EFIT LCFS"


def test_real_window_enumeration():
    d, shots = _real_shots()
    if not shots:
        raise SkipTest("no TokaMark zarr data with an equilibrium group (set TM_ZARR_DIR)")
    ti = et.TruthIndex(d)
    for s in shots:
        ws = ti.windows(s)
        assert ws, s
        idx = sorted(ws)
        assert idx == list(idx) and len(set(idx)) == len(idx)
        for wi, w in ws.items():
            assert abs(w["t_psi"] - (w["t_cut"] + et.DELTA)) <= 0.5 * et.PSI_DT + 1e-9, (s, wi, w["t_psi"], w["t_cut"])
            assert np.isfinite(w["psi_raw"]).any()
    assert ti.get(shots[0], 10 ** 6) is None


def test_alignment_with_real_tokamark_dataset():
    """Only when TM_ALIGN_PKL points at a pickle of [(shot_id, window_index, y)] made by xpoint_diagnostic.py --dump-real-windows."""
    pkl = os.environ.get("TM_ALIGN_PKL")
    d, _ = _real_shots()
    if not pkl or not os.path.exists(pkl) or d is None:
        raise SkipTest("set TM_ALIGN_PKL (xpoint_diagnostic.py --prove-alignment --dump-real-windows) and TM_ZARR_DIR")
    import pickle
    rows = pickle.load(open(pkl, "rb"))
    ok, missing = et.check_alignment(rows, d)
    assert ok > 0 and not missing, (ok, missing[:5])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except SkipTest as e:
                print("SKIP", name, "-", e)
    print("all Module 1 checks passed")
