#!/usr/bin/env python
"""Multi-seed aggregation of robust_metrics.json files (numpy only, no torch/pipeline import).

    python aggregate_seeds.py OUT.json RUN_DIR [RUN_DIR ...]      # RUN_DIR holds robust_metrics.json (or is the json itself)
    python aggregate_seeds.py --summary OUT.md RUN_DIR [RUN_DIR ...]   # one-run-per-row headline table (RS, boundary + 95% CI)

Output: {"n_seeds", "sources", "cells": {cell: {"NRMSE": {mean,std,n}, "boundary_mean_cm": {...}, "RS": ..., ...}}}
cell is "clean" or "family|param|impute" (same ids as robust_metrics.json["corrupted"]). std is the sample std
(ddof=1) across seeds, 0.0 when n == 1. RS = clean NRMSE / cell NRMSE of the SAME run (per seed), then averaged.

Also provides bootstrap_ci(): the cluster (shot-level) bootstrap used for boundary-error CIs inside one run.
"""
import json
import os
import sys

import numpy as np

METRICS = ("NRMSE", "NMAE", "RMSE", "MAE")
PHYS_KEYS = ("boundary_mean_cm", "boundary_hausdorff_cm", "axis_shift_cm", "xpoint_detect_rate_pct", "false_alarm_rate_pct",
             "missed_alarm_rate_pct", "boundary_fail_rate_pct")


def bootstrap_ci(shots, values, n_boot=1000, alpha=0.05, seed=0):
    """Mean of `values` with a percentile CI from resampling whole shots (windows of one shot are correlated).
    NaNs are ignored. Returns dict(mean, lo, hi, n, n_shots)."""
    shots = np.asarray(shots)
    v = np.asarray(values, dtype=float)
    ok = ~np.isnan(v)
    shots, v = shots[ok], v[ok]
    if v.size == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0, n_shots=0)
    uniq, inv = np.unique(shots, return_inverse=True)
    s = np.bincount(inv, weights=v, minlength=len(uniq))  # per-shot sum and count -> resampled mean without loops over windows
    c = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(mean=float(v.mean()), lo=float(lo), hi=float(hi), n=int(v.size), n_shots=int(len(uniq)))


def _ms(xs):
    xs = np.asarray([x for x in xs if x is not None and not np.isnan(x)], dtype=float)
    if xs.size == 0:
        return None
    return dict(mean=float(xs.mean()), std=float(xs.std(ddof=1)) if xs.size > 1 else 0.0, n=int(xs.size))


def _cell(m, clean_nrmse):
    out = {k: m.get(k) for k in METRICS}
    out["RS"] = clean_nrmse / m["NRMSE"] if m.get("NRMSE") else None
    ph = m.get("PHYS") or {}
    for k in PHYS_KEYS:
        out[k] = ph.get(k)
    out["n_phys_windows"] = ph.get("n_windows")
    return out


def load_run(path):
    p = path if path.endswith(".json") else os.path.join(path, "robust_metrics.json")
    return json.load(open(p))


def aggregate(paths):
    runs = [load_run(p) for p in paths]
    cells = {}
    for r in runs:
        cn = r["clean"]["NRMSE"]
        for cid, m in [("clean", r["clean"])] + list(r["corrupted"].items()):
            cells.setdefault(cid, []).append(_cell(m, cn if cid != "clean" else cn))
    agg = {cid: {k: _ms([c[k] for c in lst]) for k in lst[0]} for cid, lst in cells.items()}
    return dict(n_seeds=len(runs), sources=[str(p) for p in paths], cells=agg)


FAMILIES = [("random_dropout", ["0.1", "0.25", "0.5"]), ("temporal_gap", ["0.2", "0.4", "0.6"]),
            ("disruption_proximate", ["0.1", "0.25", "0.5"]), ("channel_ablation", ["1", "3", "6"]),
            ("group_failure", ["magnetic_probes", "power_supply", "magnetics_all"]),
            ("category_abl", ["flux_loop", "pickup_fields", "saddle_voltage", "active_coils", "solenoid", "plasma_current"])]


def run_summary(r):
    """Headline numbers of one run: Robustness Score (mean over all 18 family params of clean/NRMSE, zero-fill and best-of-3 imputation)."""
    cn, cor = r["clean"]["NRMSE"], r["corrupted"]
    out = dict(clean_NRMSE=cn, n_shots=r["clean"]["n_shots"], n_windows=r["clean"]["n_windows"], n_passes=len(cor))
    ph = r["clean"].get("PHYS_ci95", {}).get("boundary_mean_cm")
    if ph:
        out["clean_boundary_cm"] = ph
    out["clean_PHYS_counts"] = r["clean"].get("PHYS_counts")
    for imp, imps in (("zero", ("zero",)), ("best", ("zero", "mean", "ffill"))):
        fam = {}
        for f, ps in FAMILIES:
            v = [cn / min(cor[f"{f}|{p}|{i}"]["NRMSE"] for i in imps) for p in ps if all(f"{f}|{p}|{i}" in cor for i in imps)]
            if v:
                fam[f] = float(np.mean(v))
        out[f"RS_{imp}_by_family"] = fam
        out[f"RS_{imp}"] = float(np.mean(list(fam.values()))) if len(fam) == len(FAMILIES) else None
    for key in ("group_failure|magnetics_all|zero", "temporal_gap|0.4|zero", "channel_ablation|6|zero", "disruption_proximate|0.25|zero"):
        if key in cor:
            c = cor[key]
            out[key] = dict(NRMSE=c["NRMSE"], boundary_cm=(c.get("PHYS_ci95", {}).get("boundary_mean_cm") or {}))
    return out


def summary_md(paths):
    rows = ["| run | shots/windows | passes | clean NRMSE | clean boundary cm [95% CI] | RS zero | RS best-imp | group loss b cm (NRMSE) | gap 40% b cm (NRMSE) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for p in paths:
        s = run_summary(load_run(p))
        b = s.get("clean_boundary_cm") or {}

        def cell(k):
            c = s.get(k)
            return f"{c['boundary_cm'].get('mean', float('nan')):.2f} ({c['NRMSE']:.3f})" if c else "--"
        rs = lambda v: f"{v:.3f}" if v is not None else "--"
        rows.append(f"| {os.path.basename(os.path.normpath(p if not p.endswith('.json') else os.path.dirname(p)))} | {s['n_shots']}/{s['n_windows']} | {s['n_passes']} | "
                    f"{s['clean_NRMSE']:.4f} | {b.get('mean', float('nan')):.2f} [{b.get('lo', float('nan')):.2f}, {b.get('hi', float('nan')):.2f}] | "
                    f"{rs(s['RS_zero'])} | {rs(s['RS_best'])} | {cell('group_failure|magnetics_all|zero')} | {cell('temporal_gap|0.4|zero')} |")
    return chr(10).join(rows) + chr(10)


def main(argv):
    if len(argv) >= 4 and argv[1] == "--summary":
        open(argv[2], "w", encoding="utf-8").write(summary_md(argv[3:]))
        print(open(argv[2], encoding="utf-8").read())
        return
    if len(argv) < 3:
        raise SystemExit(__doc__)
    res = aggregate(argv[2:])
    json.dump(res, open(argv[1], "w"), indent=1)
    print(f"{res['n_seeds']} run(s), {len(res['cells'])} cells -> {argv[1]}")


if __name__ == "__main__":
    main(sys.argv)
