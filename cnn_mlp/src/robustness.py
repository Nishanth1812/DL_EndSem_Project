#!/usr/bin/env python
"""TokaMark task_1-3 sensor-robustness harness: corrupt | eval-robust | report.

Stress-tests a *trained, frozen* task_1-3 model under the six sensor-failure
scenarios of Gupta (arXiv 2607.11915), on the TEST split only, with optional
imputation. Reuses the exact pipeline helpers (build_data / make_model) and the
official un-standardise + hierarchical-metric protocol so the CLEAN baseline
reproduces the eval-time NRMSE before any corruption is trusted.

Commands:
  python robustness.py corrupt     --split random     # build corrupted test loaders + manifest
  python robustness.py eval-robust --split random     # run the sweep, save metrics
  python robustness.py report      --split random     # figures + robustness.md

All settings come from TM_* env vars (same CFG contract as pipeline.py). Robustness
knobs:
  TM_ROB_SCENARIOS  comma list; default: random_dropout,channel_ablation,category_abl,
                    temporal_gap,group_failure,disruption_proximate  (or 'all')
  TM_ROB_IMPUTE     zero,mean,ffill (comma list); default: zero,mean,ffill
  TM_ROB_SEED       RNG seed for corruption (default TM_SEED)
  TM_ROB_CLEAN_ONLY 1 -> only the clean pass (sanity check), skip corruption
  TM_ROB_BATCH      forward batch size (default TM_BATCH)
"""
import os, sys, json, time, math, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline as P
from pipeline import CFG, T0, log, setup_imports, build_data, make_model

TASK = CFG["task"]

SIGNALS = [
    "magnetics/flux_loop_flux",
    "magnetics/b_field_pol_probe_ccbv_field",
    "magnetics/b_field_pol_probe_obr_field",
    "magnetics/b_field_pol_probe_obv_field",
    "magnetics/b_field_tor_probe_saddle_voltage",
    "pf_active/coil_current",
    "pf_active/solenoid_current",
    "summary/ip",
]
SIGNAL_SHORT = ["flux_loop", "ccbv", "obr", "obv", "saddle_v", "coil_i", "solenoid_i", "ip"]

CATEGORY_GROUPS = {
    "flux_loop": [0],
    "pickup_fields": [1, 2, 3],
    "saddle_voltage": [4],
    "active_coils": [5],
    "solenoid": [6],
    "plasma_current": [7],
}
CORRELATED_GROUPS = {
    "magnetic_probes": [1, 2, 3],
    "power_supply": [5, 6],
    "magnetics_all": [0, 1, 2, 3, 4],
}

SCENARIOS = ["random_dropout", "channel_ablation", "category_abl",
             "temporal_gap", "group_failure", "disruption_proximate"]
SEVERITIES = {
    "random_dropout": [0.10, 0.25, 0.50],
    "channel_ablation": [1, 3, 6],
    "category_abl": [1],
    "temporal_gap": [0.20, 0.40, 0.60],
    "group_failure": [1],
    "disruption_proximate": [0.10, 0.25, 0.50],
}
IMPUTE_MODES = ("zero", "mean", "ffill")


def envf(k, d):
    return type(d)(os.environ.get(k, d))


def rob_cfg():
    scen = os.environ.get("TM_ROB_SCENARIOS", "all")
    if scen.strip() == "all":
        scen = SCENARIOS
    else:
        scen = [s.strip() for s in scen.split(",") if s.strip()]
    imp = [m.strip() for m in os.environ.get("TM_ROB_IMPUTE", "zero,mean,ffill").split(",") if m.strip()]
    bad = [s for s in scen if s not in SCENARIOS]
    bad += [m for m in imp if m not in IMPUTE_MODES]
    if bad:
        raise SystemExit(f"FATAL: unknown robustness options: {bad}")
    return dict(
        scenarios=scen,
        imputes=imp,
        seed=envf("TM_ROB_SEED", CFG["seed"]),
        clean_only=os.environ.get("TM_ROB_CLEAN_ONLY", "0") == "1",
        batch=envf("TM_ROB_BATCH", CFG["batch_size"]),
    )


def rob_paths(split):
    out = os.path.join(CFG["work"], "out", split)
    rdir = os.path.join(out, "robustness")
    return dict(
        dir=rdir,
        manifest=os.path.join(rdir, "robust_manifest.json"),
        metrics=os.path.join(rdir, "robust_metrics.json"),
        windows=os.path.join(rdir, "robust_windows.csv"),
        figures=os.path.join(rdir, "figures"),
        report=os.path.join(rdir, "robustness.md"),
        model_dir=os.path.join(out, "model") + "/",
    )


# ======================================================================================
# CORRUPTION PRIMITIVES  (operate on a per-channel torch tensor x: (B, W, C, L))
# ======================================================================================
def _as_np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def _mask_random_dropout(arr, sev, rng):
    """Zero a fraction `sev` of the *finite, non-zero* values (Gupta's definition).

    Returns (masked_array, hole_mask) where hole_mask marks exactly the values we zeroed.
    """
    m = np.isfinite(arr) & (arr != 0)
    idx = np.flatnonzero(m.ravel())
    out = arr.copy()
    holes = np.zeros_like(arr, dtype=bool)
    if idx.size == 0:
        return out, holes
    k = int(round(sev * idx.size))
    drop = rng.choice(idx, size=min(k, idx.size), replace=False)
    out.ravel()[drop] = 0.0
    holes.ravel()[drop] = True
    return out, holes


def _mask_temporal_gap(arr, frac, rng, where="random"):
    """Zero a contiguous fraction `frac` of the time axis (last axis)."""
    out = arr.copy()
    holes = np.zeros_like(arr, dtype=bool)
    L = arr.shape[-1]
    g = max(1, int(round(frac * L)))
    if where == "front":
        s = 0
    elif where == "end":
        s = L - g
    else:
        s = int(rng.integers(0, max(1, L - g + 1)))
    out[..., s:s + g] = 0.0
    holes[..., s:s + g] = True
    return out, holes


def _zero_channels(arr, chans):
    """Zero whole channels; return (masked, holes) with holes marking the zeroed area."""
    out = arr.copy()
    holes = np.zeros_like(arr, dtype=bool)
    for c in chans:
        out[..., c, :] = 0.0
        holes[..., c, :] = True
    return out, holes


def _impute(masked, holes, mode, channel_mean=0.0):
    """Fill ONLY the positions flagged in `holes`. mode in {zero,mean,ffill}.

    'zero'   -> leave the zeros (TokaMark default; no filling).
    'mean'   -> put `channel_mean` (TRAIN-split signal mean) in every hole.
    'ffill'  -> carry the last valid (non-hole) value forward along time, per channel.
    """
    if mode == "zero":
        return masked
    out = np.array(masked, dtype=masked.dtype, copy=True)
    if not holes.any():
        return out
    if mode == "mean":
        out[holes] = float(channel_mean)
        return out
    if mode == "ffill":
        ow = out.reshape(-1, out.shape[-2], out.shape[-1])
        hw = holes.reshape(-1, holes.shape[-2], holes.shape[-1])
        for i in range(ow.shape[0]):
            for c in range(ow.shape[1]):
                row, hole = ow[i, c], hw[i, c]
                last = 0.0
                for t in range(row.shape[0]):
                    if hole[t]:
                        row[t] = last
                    else:
                        last = row[t]
        return out
    raise ValueError(mode)


# ======================================================================================
# SCENARIO APPLICATION  (returns a new inputs list, corruption bookkeeping dict)
# ======================================================================================
def apply_scenario(inputs, scenario, sev, impute, train_means, rng):
    """inputs: list of 8 torch tensors (B, W, C, L). Returns (corrupted_inputs, info).

    Corruption is applied on CPU in numpy, then re-wrapped to torch tensors matching
    each input's original dtype. Imputation 'mean' uses `train_means` (per signal, a
    scalar computed from the TRAIN split) so nothing about the test split leaks.
    """
    import torch
    info = dict(scenario=scenario, severity=sev, impute=impute, details={})
    # channel-set scenarios are deterministic per task (draw once, not per channel)
    ablate_chans = None
    if scenario == "channel_ablation":
        n = min(int(sev), len(SIGNALS))
        ablate_chans = sorted(rng.choice(len(SIGNALS), size=n, replace=False).tolist())
        info["details"]["channels_zeroed"] = [SIGNAL_SHORT[c] for c in ablate_chans]
    out = []
    for i, t in enumerate(inputs):
        arr = _as_np(t).astype(np.float64)
        masked, holes = arr, np.zeros_like(arr, dtype=bool)
        if scenario == "random_dropout":
            masked, holes = _mask_random_dropout(arr, float(sev), rng)
        elif scenario == "channel_ablation":
            if i in ablate_chans:
                masked, holes = _zero_channels(arr, [0])
        elif scenario == "temporal_gap":
            masked, holes = _mask_temporal_gap(arr, float(sev), rng, where="random")
        elif scenario == "disruption_proximate":
            masked, holes = _mask_temporal_gap(arr, float(sev), rng, where="end")
        else:
            raise ValueError(scenario)
        masked = _impute(masked, holes, impute, channel_mean=train_means.get(i, 0.0))
        out.append(torch.as_tensor(masked, dtype=getattr(t, "dtype", None)))
    return out, info


def apply_group(inputs, scenario, sev, impute, train_means, rng, group_key, group_chans):
    """Like apply_scenario but zeroes a fixed set of channels (category / group runs)."""
    import torch
    group_chans = [c for c in group_chans if 0 <= c < len(inputs)]
    info = dict(scenario=scenario, severity=sev, impute=impute,
                details={"group": group_key, "channels": [SIGNAL_SHORT[c] for c in group_chans]})
    out = []
    for i, t in enumerate(inputs):
        arr = _as_np(t).astype(np.float64)
        if i in group_chans:
            masked, holes = _zero_channels(arr, [0])
        else:
            masked, holes = arr, np.zeros_like(arr, dtype=bool)
        masked = _impute(masked, holes, impute, channel_mean=train_means.get(i, 0.0))
        out.append(torch.as_tensor(masked, dtype=getattr(t, "dtype", None)))
    return out, info


# ======================================================================================
# METRICS  (reuse the official hierarchy: per-window -> per-shot -> across shots)
# ======================================================================================
def _psi_mean_std(meta):
    m = meta["output"]["equilibrium-psi"]
    return float(m["mean"]), float(m["std"])


def window_metrics(y_true, y_pred):
    """y_true/y_pred: (N, 65, 65) numpy, real units. Returns per-window RMSE, MAE arrays."""
    diff = y_pred - y_true
    rmse = np.sqrt(np.nanmean(diff ** 2, axis=(1, 2)))
    mae = np.nanmean(np.abs(diff), axis=(1, 2))
    return rmse, mae


def aggregate(shot_ids, rmse, mae, std):
    """Per-window -> per-shot mean -> across-shot mean/std; plus window-level stats.

    Mirrors tokamark.evaluator: RMSE per window, then averaged per shot, then NRMSE =
    shot_RMSE / output_std, then mean/std across shots (population sd, ddof=0).
    """
    shot_ids = np.asarray(shot_ids, dtype=np.int64)
    rmse = np.asarray(rmse, dtype=np.float64)
    mae = np.asarray(mae, dtype=np.float64)
    uniq = np.unique(shot_ids)
    shot_rmse, shot_mae = [], []
    for s in uniq:
        m = shot_ids == s
        # Official TokaMark: shot RMSE = sqrt(mean(window_RMSE ** 2))
        shot_rmse.append(float(np.sqrt(np.nanmean(rmse[m] ** 2))))
        shot_mae.append(float(np.nanmean(mae[m])))
    shot_rmse = np.asarray(shot_rmse)
    shot_mae = np.asarray(shot_mae)
    return dict(
        n_shots=int(len(uniq)),
        n_windows=int(len(rmse)),
        NRMSE=float(np.nanmean(shot_rmse / std)),
        NRMSE_std=float(np.nanstd(shot_rmse / std, ddof=0)),
        NMAE=float(np.nanmean(shot_mae / std)),
        NMAE_std=float(np.nanstd(shot_mae / std, ddof=0)),
        RMSE=float(np.nanmean(shot_rmse)),
        RMSE_std=float(np.nanstd(shot_rmse, ddof=0)),
        MAE=float(np.nanmean(shot_mae)),
        MAE_std=float(np.nanstd(shot_mae, ddof=0)),
        window_RMSE_mean=float(np.nanmean(rmse)),
        window_RMSE_median=float(np.nanmedian(rmse)),
        window_RMSE_p90=float(np.nanpercentile(rmse, 90)),
        window_RMSE_max=float(np.nanmax(rmse)),
    )


# ======================================================================================
# INFERENCE PASS
# ======================================================================================
def load_test_loader_and_model(split, batch=None):
    setup_imports()
    import torch
    loaders, meta = build_data(split, ["test"])
    if batch is not None:
        # rebuild the loader with the requested batch size (build_data uses CFG batch)
        from torch.utils.data import DataLoader
        from src.trainer import model_collate_fn
        from utils import seed_worker
        ds = loaders["test"].dataset
        nw = CFG["dl_workers"]
        if nw < 0:
            nw = max(2, min(8, os.cpu_count() or 2))
        g = torch.Generator(); g.manual_seed(CFG["seed"])
        loaders["test"] = DataLoader(ds, batch_size=batch, num_workers=nw, collate_fn=model_collate_fn,
                                     worker_init_fn=seed_worker, generator=g,
                                     pin_memory=torch.cuda.is_available(),
                                     prefetch_factor=2 if nw > 0 else None)
    model = make_model(loaders["test"], meta)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(os.path.join(rob_paths(split)["model_dir"], "best_model.pt"),
                                     map_location=dev, weights_only=False))
    model.to(dev).eval()
    return loaders["test"], model, meta, dev


def train_channel_means(split):
    """Per-signal mean computed on the TRAIN subset (for mean-imputation, no leakage)."""
    setup_imports()
    try:
        P_ = P.paths(split)
        sel = json.load(open(P_["sel"]))
        train_shots = [s for s in sel.get("shots", {}).get("train", [])
                       if os.path.isdir(os.path.join(P_["data"], f"{s}.zarr"))]
        if len(train_shots) >= 5:
            loaders, _ = build_data(split, ["train"])
            return _train_means_pass(loaders["train"])
        else:
            log(f"only {len(train_shots)} train shots on disk; using baseline 0.0 for mean imputation")
            return {i: 0.0 for i in range(len(SIGNALS))}
    except Exception as e:  # noqa
        log("train-channel-mean pass failed, falling back to zeros:", repr(e)[:200])
        return {i: 0.0 for i in range(len(SIGNALS))}


def _train_means_pass(loader, max_batches=40):
    sums = None; cnts = None
    seen = 0
    for batch in loader:
        if batch is None:
            continue
        x = batch[2]
        if sums is None:
            sums = [0.0] * len(x); cnts = [0] * len(x)
        for i, t in enumerate(x):
            a = _as_np(t).astype(np.float64)
            a = a[np.isfinite(a) & (a != 0)]
            sums[i] += float(a.sum()); cnts[i] += int(a.size)
        seen += 1
        if seen >= max_batches:
            break
    means = {}
    for i in range(len(sums)):
        means[i] = (sums[i] / cnts[i]) if cnts[i] > 0 else 0.0
    log(f"train channel means (first {seen} batches): " +
        ", ".join(f"{SIGNAL_SHORT[i]}={means[i]:.3g}" for i in sorted(means)))
    return means


def run_pass(loader, model, dev, mu, sd, corrupt_fn=None, accumulate_spatial=False, return_viz=False):
    """Forward the whole test loader; return aggregated metrics + per-window rows.

    corrupt_fn: callable(x_list) -> (x_corrupted_list, info) applied per batch, or None
                for the clean pass.
    """
    import torch
    sid_all, wid_all, rmse_all, mae_all = [], [], [], []
    spatial_sum = None; spatial_cnt = None
    viz_data = None
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            shot_id, window_id, x, y = batch
            x = [t.to(dev, dtype=torch.float32) for t in x]
            xin = x
            if corrupt_fn is not None:
                xin_cpu, info = corrupt_fn(x)
                xin = [t.to(dev, dtype=torch.float32) for t in xin_cpu]
            y_pred = model(*xin)
            if not isinstance(y_pred, (list, tuple)):
                y_pred = [y_pred]
            yt = y[0].detach().cpu().squeeze(1).numpy().astype(np.float64) * sd + mu
            yp = y_pred[0].detach().cpu().squeeze(1).numpy().astype(np.float64) * sd + mu
            yt = yt.reshape(yt.shape[0], -1, 65, 65)
            yp = yp.reshape(yp.shape[0], -1, 65, 65)
            # flatten any extra window axis into the sample axis
            n = yt.shape[0] * yt.shape[1]
            yt2 = yt.reshape(n, 65, 65)
            yp2 = yp.reshape(n, 65, 65)
            if return_viz and viz_data is None and n > 0:
                sids_raw = np.asarray(shot_id).reshape(-1)
                viz_data = dict(yt=yt2[0].tolist(), yp=yp2[0].tolist(),
                                shot_id=int(sids_raw[0]) if len(sids_raw) > 0 else 0)
            rmse, mae = window_metrics(yt2, yp2)
            rmse = rmse.reshape(yt.shape[0], yt.shape[1]).mean(axis=1)
            mae = mae.reshape(yt.shape[0], yt.shape[1]).mean(axis=1)
            sid_all.append(np.asarray(shot_id).reshape(-1))
            wid_all.append(np.asarray(window_id).reshape(-1))
            rmse_all.append(rmse); mae_all.append(mae)
            if accumulate_spatial:
                err = np.abs(yp2 - yt2)
                s = np.nansum(err, axis=0); c = np.isfinite(err).sum(axis=0)
                spatial_sum = s if spatial_sum is None else spatial_sum + s
                spatial_cnt = c if spatial_cnt is None else spatial_cnt + c
            n_batches += 1
    sid = np.concatenate(sid_all) if len(sid_all) > 0 else np.array([])
    rmse = np.concatenate(rmse_all) if len(rmse_all) > 0 else np.array([])
    mae = np.concatenate(mae_all) if len(mae_all) > 0 else np.array([])
    m = aggregate(sid, rmse, mae, sd)
    m["n_batches"] = n_batches
    if accumulate_spatial and spatial_sum is not None and spatial_cnt is not None:
        m["spatial_abs_err"] = (spatial_sum / np.maximum(spatial_cnt, 1)).tolist()
    if return_viz and viz_data is not None:
        m["viz"] = viz_data
    m["_windows"] = dict(shot_id=sid.tolist(), rmse=rmse.tolist(), mae=mae.tolist())
    return m


def build_sweep_tasks(rc):
    """Expand the sweep into a list of task dicts (id, callable-corruption spec)."""
    tasks = []
    for scen in rc["scenarios"]:
        for sev in SEVERITIES[scen]:
            for imp in rc["imputes"]:
                if scen in ("category_abl", "group_failure"):
                    groups = CATEGORY_GROUPS if scen == "category_abl" else CORRELATED_GROUPS
                    for gname, gchans in groups.items():
                        tasks.append(dict(id=f"{scen}|{gname}|{imp}", scenario=scen,
                                          severity=sev, impute=imp, group=gname, chans=gchans))
                else:
                    tasks.append(dict(id=f"{scen}|{sev}|{imp}", scenario=scen,
                                      severity=sev, impute=imp, group=None, chans=None))
    return tasks


def make_corrupt_fn(task, rng_seed, train_means):
    """Return a callable(x_list)->(x_corrupt,info) with a fresh RNG per task (seeded)."""
    rng = np.random.default_rng(rng_seed)
    if task["scenario"] in ("category_abl", "group_failure"):
        def fn(x):
            return apply_group(x, task["scenario"], task["severity"], task["impute"],
                               train_means, rng, task["group"], task["chans"])
    else:
        def fn(x):
            return apply_scenario(x, task["scenario"], task["severity"], task["impute"],
                                  train_means, rng)
    return fn


# ======================================================================================
# COMMANDS
# ======================================================================================
def cmd_corrupt(split):
    """Build + persist the sweep manifest (and verify the clean baseline gate)."""
    rc = rob_cfg()
    P_ = rob_paths(split); os.makedirs(P_["dir"], exist_ok=True)
    tasks = build_sweep_tasks(rc)
    json.dump(dict(config=rc, tasks=tasks, signals=SIGNALS, categories=CATEGORY_GROUPS,
                   correlated_groups=CORRELATED_GROUPS, severities=SEVERITIES),
              open(P_["manifest"], "w"), indent=1)
    log(f"corruption manifest: {len(tasks)} tasks -> {P_['manifest']}")
    log("sweep tasks:\n  " + "\n  ".join(t["id"] for t in tasks))


def cmd_eval_robust(split):
    rc = rob_cfg()
    P_ = rob_paths(split); os.makedirs(P_["dir"], exist_ok=True)
    loader, model, meta, dev = load_test_loader_and_model(split, batch=rc["batch"])
    mu, sd = _psi_mean_std(meta)

    # ---- CLEAN GATE: must reproduce the eval-time NRMSE before corruption is trusted
    log("CLEAN pass (sensor-healthy baseline)...")
    clean = run_pass(loader, model, dev, mu, sd, accumulate_spatial=True, return_viz=True)
    viz_samples = dict(clean=clean.pop("viz", None), tasks={})
    log(f"CLEAN: NRMSE={clean['NRMSE']:.4f} (±{clean['NRMSE_std']:.4f}) NMAE={clean['NMAE']:.4f} "
        f"RMSE={clean['RMSE']:.5f} MAE={clean['MAE']:.5f} n_shots={clean['n_shots']} n_windows={clean['n_windows']}")
    targets = None
    mp = os.path.join(P_["model_dir"], "..", "results", "task_1-3", "task_metrics.csv")
    if os.path.exists(mp):
        import csv as _csv
        with open(mp) as fh:
            for row in _csv.DictReader(fh):
                feat = row.get("feature_name", "")
                if feat in ("equilibrium-psi", "equilibrium_psi", "task_1-3") or "psi" in feat:
                    val = row.get("NRMSE_mean") or row.get("NRMSE")
                    if val:
                        try:
                            targets = float(val)
                        except Exception:
                            pass
                        break
    if targets is not None:
        d = abs(clean["NRMSE"] - targets)
        ok = d <= max(0.02, 0.15 * targets)
        log(f"clean-gate: harness NRMSE={clean['NRMSE']:.4f} vs eval CSV={targets:.4f} "
            f"(Δ={d:.4f}) -> {'PASS' if ok else 'FAIL (trust nothing downstream)'}")
        if not ok:
            log("WARNING: clean baseline mismatch; corruption numbers below are suspect")

    result = dict(clean=clean, corrupted={}, config=rc, psi_mean=mu, psi_std=sd, viz_samples=viz_samples)
    if rc["clean_only"]:
        clean.pop("_windows", None)
        json.dump(result, open(P_["metrics"], "w"), indent=1)
        log("TM_ROB_CLEAN_ONLY=1 -> wrote clean-only metrics, skipping corruption")
        return

    log("building TRAIN channel means for mean-imputation...")
    train_means = train_channel_means(split)

    VIZ_TASKS = {"random_dropout|0.5|zero", "random_dropout|0.5|ffill",
                 "category_abl|plasma_current|zero", "category_abl|plasma_current|ffill"}
    tasks = build_sweep_tasks(rc)
    log(f"corruption sweep: {len(tasks)} tasks on the TEST split...")
    rows = []
    for ti, task in enumerate(tasks):
        cf = make_corrupt_fn(task, rc["seed"] * 1000003 + ti, train_means)
        need_viz = task["id"] in VIZ_TASKS
        log(f"[{ti + 1}/{len(tasks)}] {task['id']} ...")
        try:
            m = run_pass(loader, model, dev, mu, sd, corrupt_fn=cf, return_viz=need_viz)
        except Exception as e:  # noqa
            log(f"  task {task['id']} FAILED: {repr(e)[:200]}")
            continue
        if need_viz and "viz" in m:
            viz_samples["tasks"][task["id"]] = m.pop("viz", None)
        m.pop("_windows", None)
        m["id"] = task["id"]
        rs = clean["NRMSE"] / m["NRMSE"] if m["NRMSE"] > 0 else float("nan")
        m["RS_vs_clean"] = float(rs)
        result["corrupted"][task["id"]] = m
        log(f"  {task['id']}: NRMSE={m['NRMSE']:.4f} (+{100 * (m['NRMSE'] / clean['NRMSE'] - 1):.1f}%) "
            f"NMAE={m['NMAE']:.4f} RS={rs:.3f}")
        rows.append(dict(id=task["id"], scenario=task["scenario"], severity=task["severity"],
                         impute=task["impute"], group=task["group"], **{k: m[k] for k in (
                             "NRMSE", "NMAE", "RMSE", "MAE", "RS_vs_clean", "n_shots", "n_windows")}))
        json.dump(result, open(P_["metrics"], "w"), indent=1)  # incremental save

    clean.pop("_windows", None)
    json.dump(result, open(P_["metrics"], "w"), indent=1)
    log(f"robustness metrics written -> {P_['metrics']} ({len(rows)} corrupted tasks)")


def cmd_report(split):
    rc = rob_cfg()
    P_ = rob_paths(split)
    if not os.path.exists(P_["metrics"]):
        raise SystemExit(f"FATAL: {P_['metrics']} not found - run eval-robust first")
    res = json.load(open(P_["metrics"]))
    os.makedirs(P_["figures"], exist_ok=True)
    clean = res["clean"]; corrupted = res["corrupted"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- RS aggregation: worst-case NRMSE per scenario (zero-fill), Gupta-style
    def rs_table():
        worst = {}
        for tid, m in corrupted.items():
            scen, _, imp = tid.split("|")
            if imp != "zero":
                continue
            worst[scen] = max(worst.get(scen, 0.0), m["NRMSE"])
        rs = {scen: (clean["NRMSE"] / v if v else float("nan")) for scen, v in worst.items()}
        return worst, rs

    worst, rs = rs_table()

    # ---- figure 1: degradation curves NRMSE vs severity, per scenario (zero-fill)
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        for scen in SCENARIOS:
            pts = []
            for tid, m in corrupted.items():
                parts = tid.split("|")
                if parts[0] == scen and parts[-1] == "zero" and len(parts) == 3:
                    pts.append((float(parts[1]), m["NRMSE"]))
            if pts:
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=scen)
        ax.axhline(clean["NRMSE"], ls="--", c="k", lw=1, label="clean")
        ax.set_xlabel("severity"); ax.set_ylabel("NRMSE (test, un-standardised)")
        ax.set_title(f"Robustness degradation - {TASK} ({split})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(P_["figures"], "degradation_curves.png"), dpi=120)
        plt.close(fig)
        log("saved degradation_curves.png")
    except Exception as e:  # noqa
        log("degradation curve failed:", repr(e)[:200])

    # ---- figure 2: per-signal importance (category_abl + channel_ablation n=1), zero-fill
    try:
        labels, vals = [], []
        for tid, m in corrupted.items():
            p = tid.split("|")
            if p[0] == "category_abl" and p[-1] == "zero":
                labels.append(p[1]); vals.append(100 * (m["NRMSE"] / clean["NRMSE"] - 1))
        order = np.argsort(vals)[::-1]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([labels[i] for i in order], [vals[i] for i in order], color="steelblue")
        ax.set_ylabel("% NRMSE increase vs clean"); ax.grid(alpha=0.3, axis="y")
        ax.set_title(f"Diagnostic importance (zero each category) - {TASK}")
        fig.tight_layout(); fig.savefig(os.path.join(P_["figures"], "signal_importance.png"), dpi=120)
        plt.close(fig)
        log("saved signal_importance.png")
    except Exception as e:  # noqa
        log("importance bar failed:", repr(e)[:200])

    # ---- figure 3: spatial error map (clean)
    try:
        sm = np.array(clean.get("spatial_abs_err", []))
        if sm.size:
            fig, ax = plt.subplots(figsize=(4.5, 4))
            im = ax.imshow(sm, cmap="magma")
            ax.set_title("Clean |error| map (65x65)")
            ax.axis("off"); fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout(); fig.savefig(os.path.join(P_["figures"], "spatial_error_clean.png"), dpi=120)
            plt.close(fig)
            log("saved spatial_error_clean.png")
    except Exception as e:  # noqa
        log("spatial map failed:", repr(e)[:200])

    # ---- figure 4: qualitative 2D psi under corruption & mitigation
    try:
        vs = res.get("viz_samples")
        if vs and vs.get("clean") and vs.get("tasks"):
            yt = np.array(vs["clean"]["yt"])
            yp_clean = np.array(vs["clean"]["yp"])
            sid = vs["clean"].get("shot_id", 0)

            t_drop_z = vs["tasks"].get("random_dropout|0.5|zero") or {}
            t_drop_f = vs["tasks"].get("random_dropout|0.5|ffill") or {}
            t_ip_z = vs["tasks"].get("category_abl|plasma_current|zero") or {}
            t_ip_f = vs["tasks"].get("category_abl|plasma_current|ffill") or {}

            yp_drop_z = np.array(t_drop_z.get("yp", yp_clean))
            yp_drop_f = np.array(t_drop_f.get("yp", yp_clean))
            yp_ip_z = np.array(t_ip_z.get("yp", yp_clean))
            yp_ip_f = np.array(t_ip_f.get("yp", yp_clean))

            fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), layout="constrained")
            rows = [
                ("50% Random Sensor Dropout", yp_drop_z, yp_drop_f),
                ("Complete Plasma Current (Ip) Loss", yp_ip_z, yp_ip_f),
            ]
            for r_i, (label, yp_z, yp_f) in enumerate(rows):
                err_clean = float(np.mean(np.abs(yp_clean - yt)))
                err_z = float(np.mean(np.abs(yp_z - yt)))
                err_f = float(np.mean(np.abs(yp_f - yt)))
                cols = [
                    (yt, f"Ground Truth $\\psi$ (Shot {sid})", "viridis"),
                    (yp_clean, f"Clean Pred (MAE: {err_clean:.4f} Wb)", "viridis"),
                    (yp_z, f"Fault: {label} [Zero-Fill]\n(MAE: {err_z:.4f} Wb)", "inferno"),
                    (yp_f, f"Mitigated: [Forward-Fill]\n(MAE: {err_f:.4f} Wb)", "viridis"),
                ]
                for c_i, (img, ttl, cmap) in enumerate(cols):
                    ax = axes[r_i, c_i]
                    im = ax.imshow(img, cmap=cmap)
                    ax.set_title(ttl, fontsize=9.5, fontweight="semibold")
                    ax.axis("off")
                    fig.colorbar(im, ax=ax, fraction=0.046)
            out_psi = os.path.join(P_["figures"], "robustness_psi_reconstruction.png")
            fig.savefig(out_psi, dpi=150)
            plt.close(fig)
            log("saved robustness_psi_reconstruction.png")
    except Exception as e:  # noqa
        log("robustness psi plot failed:", repr(e)[:200])

    # ---- figure 5: diagnostic criticality ranking
    try:
        cats, vals = [], []
        for tid, m in corrupted.items():
            p = tid.split("|")
            if p[0] == "category_abl" and p[-1] == "zero":
                cats.append(p[1])
                vals.append(100 * (m["NRMSE"] / clean["NRMSE"] - 1))
        if cats:
            order = np.argsort(vals)
            max_v = max(vals) if vals else 100.0
            fig, ax = plt.subplots(figsize=(7.5, 3.8), layout="constrained")
            bars = ax.barh([cats[i] for i in order], [vals[i] for i in order], color="#2b5c8f", height=0.6)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlim(0, max_v * 1.25)
            ax.set_xlabel("NRMSE Error Surge Above Clean Baseline (%)", fontsize=10, fontweight="bold")
            ax.set_title("Diagnostic Criticality: Test NRMSE Surge When Sensor Groups Fail", fontsize=11, fontweight="bold")
            for bar, idx in zip(bars, order):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f"+{vals[idx]:.1f}%", va="center", ha="left", fontsize=9, fontweight="semibold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.savefig(os.path.join(P_["figures"], "robustness_criticality.png"), dpi=200)
            plt.close(fig)
            log("saved robustness_criticality.png")
    except Exception as e:  # noqa
        log("robustness criticality failed:", repr(e)[:200])

    # ---- figure 6: online fault mitigation comparison
    try:
        drop_sevs = [0.10, 0.25, 0.50]
        z_vals = [clean["NRMSE"]]
        m_vals = [clean["NRMSE"]]
        f_vals = [clean["NRMSE"]]
        labels = ["Clean (p=0)"] + [f"p = {s:.2f}" for s in drop_sevs]
        for s in drop_sevs:
            z_vals.append(corrupted.get(f"random_dropout|{s}|zero", {}).get("NRMSE", clean["NRMSE"]))
            m_vals.append(corrupted.get(f"random_dropout|{s}|mean", {}).get("NRMSE", clean["NRMSE"]))
            f_vals.append(corrupted.get(f"random_dropout|{s}|ffill", {}).get("NRMSE", clean["NRMSE"]))
        x = np.arange(len(labels))
        width = 0.24
        fig, ax = plt.subplots(figsize=(7.5, 3.8), layout="constrained")
        ax.bar(x - width, z_vals, width, label="Zero-Fill (Default Baseline)", color="#dc2626")
        ax.bar(x, m_vals, width, label="Mean-Fill (Dataset Mean)", color="#f59e0b")
        ax.bar(x + width, f_vals, width, label="Forward-Fill (Physics-Informed)", color="#10b981")
        ax.axhline(clean["NRMSE"], color="#64748b", linestyle="--", linewidth=1.0, label="Clean Baseline")
        ax.set_ylabel("Test NRMSE (Lower is Better)", fontsize=10, fontweight="bold")
        ax.set_title("Online Fault Mitigation: Imputation Under Random Dropouts", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5, fontweight="semibold")
        ax.legend(frameon=True, fontsize=8.5, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(os.path.join(P_["figures"], "robustness_mitigation.png"), dpi=200)
        plt.close(fig)
        log("saved robustness_mitigation.png")
    except Exception as e:  # noqa
        log("robustness mitigation failed:", repr(e)[:200])

    # ---- write robustness.md
    lines = [f"# Robustness report - {TASK} ({split} split)", ""]
    lines += [
        "Sensor-failure study mirroring Gupta (arXiv 2607.11915), on the equilibrium",
        "reconstruction task (Gupta used task_4-4). Model: frozen `best_model.pt`.",
        "**Test split only. Model never retrained on corrupted data.**",
        "",
        "## Clean baseline (sensor-healthy)",
        f"- NRMSE = **{clean['NRMSE']:.4f}** (±{clean['NRMSE_std']:.4f})",
        f"- NMAE = {clean['NMAE']:.4f}, RMSE = {clean['RMSE']:.5f}, MAE = {clean['MAE']:.5f}",
        f"- n_shots = {clean['n_shots']}, n_windows = {clean['n_windows']}",
        "",
        "## Robustness Score (RS) — worst case per scenario, zero-fill",
        "RS = NRMSE_clean / NRMSE_corrupted; 1.0 = no damage, lower = worse.",
        "",
        "| scenario | worst NRMSE | RS |",
        "|---|---|---|",
    ]
    for scen in SCENARIOS:
        if scen in worst:
            lines.append(f"| {scen} | {worst[scen]:.4f} | {rs[scen]:.3f} |")
    lines += ["", "## Full sweep", "",
              "| task | NRMSE | NMAE | RS | n_shots |", "|---|---|---|---|---|"]
    for tid in sorted(corrupted):
        m = corrupted[tid]
        lines.append(f"| {tid} | {m['NRMSE']:.4f} | {m['NMAE']:.4f} | {m['RS_vs_clean']:.3f} | {m['n_shots']} |")
    lines += ["", "## Caveats", "",
              "- Subset data: absolute NRMSE is NOT the official full-set figure.",
              "- mean-imputation uses TRAIN-split channel means (no test leakage).",
              "- Corruption is inference-time only; a model retrained on damaged data",
              "  could behave differently.",
              ""]
    open(P_["report"], "w", encoding="utf-8").write("\n".join(lines))
    log(f"wrote {P_['report']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["corrupt", "eval-robust", "report"])
    ap.add_argument("--split", default="random", choices=["random", "temporal"])
    args = ap.parse_args()
    log(f"robustness {args.cmd} split={args.split} task={TASK}")
    if args.cmd == "corrupt":
        cmd_corrupt(args.split)
    elif args.cmd == "eval-robust":
        cmd_eval_robust(args.split)
    else:
        cmd_report(args.split)
    log("done")


if __name__ == "__main__":
    main()
