#!/usr/bin/env python
"""Module 1 v2 diagnostic: how well does the physics-evaluation method agree with EFIT on ERROR-FREE psi, and how biased was v1?

Run over a folder of TokaMark `<shot>.zarr` stores (needs numpy, scipy, contourpy, matplotlib, zarr; no torch, no GPU):

    python xpoint_diagnostic.py --data-dir <dir> [--json out.json] [--max-shots N]
    python xpoint_diagnostic.py --data-dir <dir> --prove-alignment --tokamark-src <.../tokamark/src> --baseline-src <.../tokamark_baseline>
                                [--shots 12397 13222] [--dump-real-windows real.pkl]

The population is the set of windows TokaMarkDataset(test_mode=True) emits (efit_truth.TruthIndex), i.e. what the model is
scored on. Sections: geometry / orientation, psi_b self-consistency, boundary, magnetic axis, X-points, kappa / delta, and the
bias of the v1 heuristic (35% level, argmax axis, lower-only saddle) against EFIT.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contourpy  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import efit_truth as et  # noqa: E402
import physics_evaluation as pe  # noqa: E402


def q(a, ps=(50, 90, 99)):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not a.size:
        return dict(n=0)
    return dict(n=int(a.size), mean=float(a.mean()), **{f"p{p}": float(np.percentile(a, p)) for p in ps}, max=float(a.max()))


def fmt(d):
    return "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def heuristic_segments_cm(f, level, geom):
    """All contour pieces of f at `level` (>= 8 points), each densified separately (cm)."""
    out = []
    for sg in contourpy.contour_generator(z=f).lines(level):
        if len(sg) >= 8:
            out.append(pe._densify(pe._to_cm(sg, geom)))
    return out


def one_way(a, b):
    return cKDTree(b).query(a)[0]


def window_stats(tr):
    """Everything the report needs for one window with an EFIT LCFS, from the error-free psi. None if no LCFS."""
    lc = tr["lcfs_rz_m"]
    if lc is None:
        return None
    f, g = tr["psi_raw"], tr["grid"]
    geom = pe._geom("efit", tr)
    lc_px = pe._px_of(lc, g)
    lcfs_cm = pe._to_cm(lc_px, geom)
    lcfs_d = pe._densify(lcfs_cm)
    psi_b, spread = pe.efit_psi_b(f, lc_px)
    _, spread_T = pe.efit_psi_b(f.T, lc_px)
    rng = float(f.max() - f.min())
    s = dict(spread=spread, spread_T=spread_T, psi_b_frac=(psi_b - f.min()) / rng, lcfs_rmin=lcfs_cm[:, 0].min(), lcfs_rmax=lcfs_cm[:, 0].max(),
             lcfs_zmin=lcfs_cm[:, 1].min(), lcfs_zmax=lcfs_cm[:, 1].max())
    axis_ok = bool(np.all(np.isfinite(tr["axis_rz_m"])))
    axis_px = pe._px_of(tr["axis_rz_m"], g)[0] if axis_ok else None
    sign = 1.0
    if axis_ok:
        sign = 1.0 if pe._sample(f, axis_px[None])[0] >= psi_b else -1.0
    s["axis_is_max"] = sign > 0
    # --- v2 method on true psi vs EFIT
    r = pe.evaluate_window_physics(f, f, truth=tr)
    for k in ("boundary_mean_cm", "boundary_hausdorff_cm", "axis_shift_cm", "xpoint_oracle_err_cm", "kappa_err", "delta_err"):
        s["v2_" + k] = r[k]
    s["v2_closed"] = r["boundary_closed"]
    s["v2_xtrue"], s["v2_xfound"] = r["xpoint_true_found"], r["xpoint_true_field_found"]
    s["v2_xhit"] = r["xpoint_n_hit"] > 0
    # shape: EFIT polygon -> EFIT numbers (definition check) and psi contour -> EFIT numbers (measurement floor)
    sh = pe.shape_parameters(lcfs_cm)
    s["poly_kappa_diff"] = sh["kappa"] - tr["kappa"]
    s["poly_du_diff"] = sh["delta_upper"] - tr["delta_upper"]
    s["poly_dl_diff"] = sh["delta_lower"] - tr["delta_lower"]
    s["psi_kappa_diff"] = r["kappa_oracle"] - tr["kappa"]
    s["psi_delta_diff"] = r["delta_oracle"] - tr["delta"]
    # --- v1 heuristic on true psi vs EFIT, true geometry
    lv = pe.lcfs_level(f)
    segs = heuristic_segments_cm(f, lv, geom)
    if segs:
        ct = np.concatenate(segs)
        s["v1_boundary_mean_cm"] = 0.5 * (one_way(ct, lcfs_d).mean() + one_way(lcfs_d, ct).mean())
        s["v1_boundary_haus_cm"] = max(one_way(ct, lcfs_d).max(), one_way(lcfs_d, ct).max())
    if axis_ok:
        rp, zp = pe.get_subpixel_peak(f)  # v1: global maximum
        s["v1_axis_err_cm"] = float(np.hypot((rp - axis_px[0]) * geom.dr, (zp - axis_px[1]) * geom.dz))
        s["v1_axis_in_lcfs"] = bool(pe.Path(lc_px).contains_point((rp, zp)))
        s["v2_axis_pix_gap"] = float(np.hypot(rp - axis_px[0], zp - axis_px[1]))
    xs_low = [x for x in tr["xpoints_rz_m"] if x[1] < 0]
    xt = pe.find_divertor_xpoint(f, level=lv)
    s["v1_x_lower_true"] = bool(xs_low)
    s["v1_x_found"] = xt is not None
    if xs_low and xt is not None:
        xp = pe._px_of(np.array(xs_low), g)
        s["v1_x_err_cm"] = float(min(np.hypot((xt[0] - p[0]) * geom.dr, (xt[1] - p[1]) * geom.dz) for p in xp))
    s["n_x"] = tr["n_xpoints"]
    return s


def pct(a):
    a = np.asarray(a, dtype=bool)
    return float(100.0 * a.mean()) if a.size else float("nan")


def perturbation_experiment(wins, epsilons, n_max, seed=0):
    """Same synthetic prediction error scored by v1 (legacy geometry), v1 with true geometry, and v2 against EFIT.

    prediction = true psi + smooth random field (Gaussian-filtered, ~4 px correlation) with std = eps * psi range. This shows what a
    given flux error means in cm under each protocol; it is NOT a model result.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    wins = [w for w in wins if w["lcfs_rz_m"] is not None]
    if len(wins) > n_max:
        wins = [wins[i] for i in np.linspace(0, len(wins) - 1, n_max).astype(int)]
    out = {}
    for eps in epsilons:
        recs = {"v1_legacy": [], "v1_true_geometry": [], "v2_efit": []}
        for w in wins:
            f = w["psi_raw"]
            noise = gaussian_filter(rng.standard_normal(f.shape), 4.0)
            yp = f + eps * (f.max() - f.min()) * noise / noise.std()
            recs["v1_legacy"].append(pe.evaluate_window_physics(f, yp, mode="legacy"))
            recs["v1_true_geometry"].append(pe.evaluate_window_physics(f, yp, mode="heuristic"))
            recs["v2_efit"].append(pe.evaluate_window_physics(f, yp, truth=w))
        out[f"{eps:g}"] = {k: {m: v for m, v in pe.aggregate_physics_metrics(r).items() if m in (
            "n_windows", "boundary_mean_cm", "boundary_hausdorff_cm", "axis_shift_cm", "xpoint_err_cm", "xpoint_detect_rate_pct",
            "xpoint_false_rate_pct", "kappa_err", "delta_err", "false_alarm_rate_pct", "missed_alarm_rate_pct")} for k, r in recs.items()}
    return out


def report(data_dir, max_shots=None, perturb=(), perturb_windows=150):
    ti = et.TruthIndex(data_dir)
    shots = ti.shot_ids()
    rows, per_shot, no_eq, n_win, n_nolcfs, wins = [], {}, [], 0, 0, []
    geoms = set()
    for sid in shots:
        p = ti.shot_path(sid)
        try:
            ws = ti.windows(sid)
        except ImportError:
            raise
        except Exception as e:  # corrupt / partial store
            print(f"  skip shot {sid}: {e}", file=sys.stderr)
            continue
        if not os.path.isdir(os.path.join(p, "equilibrium", "psi")):
            no_eq.append(sid)
            continue
        if max_shots and len(per_shot) >= max_shots:
            break
        per_shot[sid] = len(ws)
        for tr in ws.values():
            n_win += 1
            geoms.add(tuple(round(tr["grid"][k], 6) for k in ("r0", "dr", "z0", "dz")))
            wins.append(tr)
            st = window_stats(tr)
            if st is None:
                n_nolcfs += 1
            else:
                st["shot"] = sid
                rows.append(st)
    out = dict(shots_with_equilibrium=len(per_shot), shots_without_equilibrium=no_eq, windows=n_win, windows_without_efit_lcfs=n_nolcfs,
               windows_scored=len(rows), grids=sorted(geoms), windows_per_shot=per_shot)
    if not rows:
        return out
    c = lambda k: np.array([r.get(k, np.nan) for r in rows], dtype=np.float64)  # noqa: E731
    out["orientation"] = dict(
        psi_b_spread_pct_z_r=q(c("spread")), psi_b_spread_pct_transposed=q(c("spread_T")),
        share_orientation_z_r_better_pct=pct(c("spread") < c("spread_T")), axis_is_psi_max_pct=pct([r["axis_is_max"] for r in rows]))
    out["psi_b_as_fraction_of_range"] = q(c("psi_b_frac"), (5, 50, 95))
    out["v2_oracle"] = dict(
        boundary_mean_cm=q(c("v2_boundary_mean_cm")), boundary_hausdorff_cm=q(c("v2_boundary_hausdorff_cm")), closed_loop_pct=pct(c("v2_closed") > 0),
        axis_err_cm=q(c("v2_axis_shift_cm")), kappa_err=q(c("v2_kappa_err")), delta_err=q(c("v2_delta_err")))
    xt, xf, xh = c("v2_xtrue") > 0, c("v2_xfound") > 0, c("v2_xhit") > 0
    out["v2_xpoint_error_free_psi"] = dict(
        windows_with_efit_xpoint=int(xt.sum()), detect_within_10cm_pct=pct(xh[xt]), windows_without=int((~xt).sum()),
        false_pct=pct(xf[~xt]), position_err_cm=q(c("v2_xpoint_oracle_err_cm")), psi_tol=pe.XPOINT_PSI_TOL_EFIT)
    out["shape_definition_check"] = dict(
        efit_polygon_minus_efit_number=dict(kappa=q(np.abs(c("poly_kappa_diff"))), delta_upper=q(np.abs(c("poly_du_diff"))), delta_lower=q(np.abs(c("poly_dl_diff")))),
        psi_contour_minus_efit_number=dict(kappa_signed=q(c("psi_kappa_diff")), kappa_abs=q(np.abs(c("psi_kappa_diff"))), delta_signed=q(c("psi_delta_diff")),
                                           delta_abs=q(np.abs(c("psi_delta_diff")))))
    out["v1_heuristic_vs_efit_on_error_free_psi"] = dict(
        note="35% level, global argmax axis, lower-only saddle at the 35% level; true-psi inputs, true geometry, EFIT as reference",
        boundary_mean_cm=q(c("v1_boundary_mean_cm")), boundary_hausdorff_cm=q(c("v1_boundary_haus_cm")),
        axis_err_cm=q(c("v1_axis_err_cm")), axis_outside_efit_lcfs_pct=pct(~(c("v1_axis_in_lcfs") > 0)), axis_gap_over_1px_pct=pct(c("v2_axis_pix_gap") > 1.0),
        lower_xpoint_windows=int((c("v1_x_lower_true") > 0).sum()),
        lower_xpoint_found_pct=pct((c("v1_x_found") > 0)[c("v1_x_lower_true") > 0]),
        lower_xpoint_err_cm=q(c("v1_x_err_cm")), spurious_lower_xpoint_pct=pct((c("v1_x_found") > 0)[~(c("v1_x_lower_true") > 0)]))
    out["scale_error_v1"] = dict(
        old_cm_per_px=[pe.LEGACY_DR_CM, pe.LEGACY_DZ_CM], true_cm_per_px=[pe.DR_CM, pe.DZ_CM],
        old_over_true=[pe.LEGACY_DR_CM / pe.DR_CM, pe.LEGACY_DZ_CM / pe.DZ_CM],
        note="v1 reported a pure radial pixel error 6.3% too small and a pure vertical one 25% too small")
    out["efit_lcfs_envelope_cm"] = dict(Rmin=float(np.nanmin(c("lcfs_rmin"))), Rmax=float(np.nanmax(c("lcfs_rmax"))),
                                        Zmin=float(np.nanmin(c("lcfs_zmin"))), Zmax=float(np.nanmax(c("lcfs_zmax"))),
                                        note="outermost EFIT LCFS over all scored windows: a data-derived lower bound on the vessel, not the wall")
    if perturb:
        out["perturbation_experiment"] = perturbation_experiment(wins, perturb, perturb_windows)
    return out


def print_report(o):
    print(f"shots with equilibrium: {o['shots_with_equilibrium']}  without: {o['shots_without_equilibrium']}")
    print(f"dataset windows: {o['windows']}  (no EFIT LCFS: {o['windows_without_efit_lcfs']}, scored: {o['windows_scored']})")
    print("grids (r0, dr, z0, dz in m):", o["grids"])
    if "orientation" not in o:
        return
    for sec in ("orientation", "psi_b_as_fraction_of_range", "v2_oracle", "v2_xpoint_error_free_psi", "shape_definition_check",
                "v1_heuristic_vs_efit_on_error_free_psi", "scale_error_v1", "efit_lcfs_envelope_cm", "perturbation_experiment"):
        if sec not in o:
            continue
        print(f"\n== {sec}")
        for k, v in o[sec].items():
            if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
                for k2, v2 in v.items():
                    if v2 and all(isinstance(x, dict) for x in v2.values()):
                        for k3, v3 in v2.items():
                            print(f"  {k}.{k2}.{k3}: {fmt(v3)}")
                    else:
                        print(f"  {k}.{k2}: {fmt(v2)}")
            else:
                print(f"  {k}: {fmt(v) if isinstance(v, dict) else v}")


# ---------------------------------------------------------------------------------------------- alignment proof
def real_windows(data_dir, tokamark_src, baseline_src, shots, split="random", task="task_1-3"):
    """Run the REAL TokaMarkDataset exactly like pipeline.build_data (test_mode=True, stride 5 ms, official filters) on `shots`."""
    import types
    sys.path[:0] = [tokamark_src, baseline_src]
    try:
        import s3fs  # noqa: F401
    except ImportError:  # only remote access needs it; the store manager just constructs the object
        m = types.ModuleType("s3fs")
        m.__version__ = "2099.1"
        m.S3FileSystem = type("S3FileSystem", (), {"async_impl": False, "mirror_sync_methods": False, "protocol": "s3", "__init__": lambda self, *a, **k: None})
        sys.modules["s3fs"] = m
    from tokamark.tasks import get_task_config, get_task_metadata
    from tokamark.tools.transforms.compose_transform import ComposeTransforms
    from tokamark.data import initialize_MAST_dataset, initialize_TokaMark_dataset
    from src.model_transform import ModelTransform_1, ModelTransform_2
    md = os.path.join(tokamark_src, "tokamark", "metadata")
    stats = os.path.join(md, f"{split}_split_signals_stats.yaml")
    outl = os.path.join(tokamark_src, "MAST_tools", "metadata", f"{split}_split_outlier_metadata.yaml")
    cfgt = get_task_config(task_name=task)
    cfgt["stride_window"] = 0.005  # the override in pipeline.py / run_training.py
    meta = get_task_metadata(config_task=cfgt, verbose=False)
    tf = ComposeTransforms([ModelTransform_1(meta | cfgt), ModelTransform_2(meta | cfgt)])
    mast = initialize_MAST_dataset(config_task=cfgt, shots_list=list(shots), local_flag=True, use_std_scaling=True, stats_metadata_file_path=stats,
                                   use_nan_filling=False, remove_outliers=True, outlier_metadata_file=outl, remove_bad_efit_rating=True,
                                   store_manager_settings={"base_local_zarr_path": data_dir}, verbose=False)
    ds = initialize_TokaMark_dataset(dataset=mast, task_metadata=meta, config_metadata=cfgt, custom_transform=tf, test_mode=True)
    if ds is None:
        return []
    return [(int(w["shot_id"]), int(w["window_index"]), np.asarray(w["y"][0])[0].astype(np.float64)) for w in ds]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--json")
    ap.add_argument("--max-shots", type=int)
    ap.add_argument("--prove-alignment", action="store_true")
    ap.add_argument("--tokamark-src")
    ap.add_argument("--baseline-src")
    ap.add_argument("--shots", type=int, nargs="*")
    ap.add_argument("--perturb", type=float, nargs="*", default=[0.01, 0.03], help="flux-error levels (fraction of range) for the v1-vs-v2 experiment; empty to skip")
    ap.add_argument("--perturb-windows", type=int, default=150)
    ap.add_argument("--split", default="random")
    ap.add_argument("--dump-real-windows")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    if a.prove_alignment:
        shots = a.shots or et.TruthIndex(a.data_dir).shot_ids()
        rows = real_windows(a.data_dir, a.tokamark_src, a.baseline_src, shots, split=a.split)
        if a.dump_real_windows:
            import pickle
            pickle.dump(rows, open(a.dump_real_windows, "wb"))
        ok, problems = et.check_alignment(rows, a.data_dir, split=a.split, atol=1e-7)
        print(f"real TokaMarkDataset windows: {len(rows)} from {len({r[0] for r in rows})} shots; matching EFIT-index psi: {ok}")
        print("ALIGNED" if not problems and ok else "MISALIGNED:\n  " + "\n  ".join(problems[:20]))
        return 0 if not problems else 1
    o = report(a.data_dir, max_shots=a.max_shots, perturb=a.perturb, perturb_windows=a.perturb_windows)
    print_report(o)
    if a.json:
        json.dump(o, open(a.json, "w"), indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else float(x) if isinstance(x, np.floating) else str(x))
    return 0


if __name__ == "__main__":
    sys.exit(main())
