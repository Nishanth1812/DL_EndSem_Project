#!/usr/bin/env python
"""Model-driven studies for Module 3 (sensor criticality) and Module 4 (adaptive mitigation) on a trained model.

    python physics_study.py criticality [--split random]
    python physics_study.py mitigation  [--split random]
    python physics_study.py all

Same environment contract as pipeline.py (TM_* variables; TM_MODEL / TM_FAILURE_AWARE must match the checkpoint).
Both studies score reconstructions with Module 1 (modules/physics_evaluation.py), on a fixed sample of every
TM_PHYS_STRIDE-th test window (default 100, the same windows the robustness sweep scores), and write to
<out>/<split>/robustness/physics/.

criticality  ablates each sensor branch (zero-fill, as a dead sensor), records the boundary / axis / X-point error and
             the error map, tiers the branches, then runs greedy backward elimination to the smallest branch set whose
             boundary error stays within 2.0 cm.  -> criticality.json
mitigation   calibrates Module 4's conformal radius on validation windows, then injects five failure scenarios into the
             test windows and compares the bare model with the full brain: boundary error, false / missed wall
             alarms, delta coverage and measured latency.  -> mitigation.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "modules"))

import pipeline as P
try:
    import pipeline_robustness as R
except ImportError:  # the Kaggle notebook writes pipeline_robustness.py as robustness.py
    import robustness as R
from pipeline import CFG, log
from adaptive_mitigation import ConformalGauge, RealTimeFaultDetector, SmartAdaptiveMitigationBrain
from physics_evaluation import (SAFETY_CLEARANCE_CM, aggregate_physics_metrics, evaluate_window_physics,
                                extract_lcfs_contour, compute_boundary_error, compute_wall_proximity)
from sensor_criticality import TIER1_CM, tier_of

STRIDE = int(os.environ.get("TM_PHYS_STRIDE", 100))
CHUNK = 64  # windows per forward pass
CALIB_ALPHA = 0.05


def sample_windows(loader, stride):
    """Every `stride`-th window of a loader as (inputs: list of (N, ...) CPU tensors, psi (N, 65, 65) standardised)."""
    import torch
    xs, ys, seen = None, [], 0
    for batch in loader:
        if batch is None:
            continue
        _, _, x, y = batch
        n = x[0].shape[0]
        keep = [j for j in range(n) if (seen + j) % stride == 0]
        seen += n
        if not keep:
            continue
        xs = [[] for _ in x] if xs is None else xs
        for i, t in enumerate(x):
            xs[i].append(t[keep].float())
        ys.append(y[0][keep].float().reshape(len(keep), 65, 65))
    if xs is None:
        raise SystemExit("no windows sampled: check the split / shots")
    return [torch.cat(c) for c in xs], torch.cat(ys)


class Study:
    """Loads the trained model once and scores reconstructions in Wb with Module 1."""

    def __init__(self, split):
        import torch
        self.torch, self.split = torch, split
        loader, self.model, meta, self.dev = R.load_test_loader_and_model(split, batch=CFG["batch_size"])
        self.mu, self.sd = R._psi_mean_std(meta)
        self.test_loader, self.meta = loader, meta
        self.out_dir = os.path.join(R.rob_paths(split)["dir"], "physics")
        os.makedirs(self.out_dir, exist_ok=True)
        self.x, y = sample_windows(loader, STRIDE)
        self.truth = y.numpy().astype(np.float64) * self.sd + self.mu
        self.n_branches = len(self.x)
        self.probes = [int(t.shape[-1]) if t.dim() > 4 else 1 for t in self.x]  # (N, W, C, T, P); a 1-probe branch is (N, W, C, T)
        log(f"{len(self.truth)} windows sampled (stride {STRIDE}); probes per branch {self.probes}")

    def predict(self, x, dead=(), mask=None):
        """Physical-units reconstructions (N, 65, 65). `dead` branches are zeroed; a failure-aware model is told."""
        torch, out = self.torch, []
        dead = list(dead)
        with torch.no_grad():
            for s in range(0, x[0].shape[0], CHUNK):
                xin = [t[s:s + CHUNK].clone().to(self.dev) for t in x]
                for c in dead:
                    xin[c].zero_()
                if mask is not None:
                    health = torch.as_tensor(mask[s:s + CHUNK], dtype=torch.float32, device=self.dev)
                else:
                    health = torch.ones((xin[0].shape[0], self.n_branches), device=self.dev)
                    health[:, dead] = 0.0
                y = self.model(*xin, health) if CFG["failure_aware"] else self.model(*xin)
                out.append(y[0].detach().cpu().float().reshape(-1, 65, 65).numpy())
        return np.concatenate(out).astype(np.float64) * self.sd + self.mu

    def score(self, pred):
        rows = [evaluate_window_physics(t, p) for t, p in zip(self.truth, pred)]
        return aggregate_physics_metrics(rows)


# ------------------------------------------------------------------ Module 3
def run_criticality(st):
    clean = st.score(st.predict(st.x))
    log(f"clean boundary error {clean['boundary_mean_cm']:.3f} cm")
    signals = []
    for c in range(st.n_branches):
        pred = st.predict(st.x, dead=[c])
        m = st.score(pred)
        err = np.abs(pred - st.truth).mean(axis=0)
        signals.append(dict(index=c, signal=R.SIGNALS[c], short=R.SIGNAL_SHORT[c], n_probes=st.probes[c],
                            boundary_cm=m["boundary_mean_cm"], shift_cm=m["boundary_mean_cm"] - clean["boundary_mean_cm"],
                            axis_shift_cm=m["axis_shift_cm"], xpoint_err_cm=m["xpoint_err_cm"], tier=tier_of(m["boundary_mean_cm"]),
                            spatial_abs_err=err.tolist()))
        log(f"  {R.SIGNAL_SHORT[c]} lost: boundary {m['boundary_mean_cm']:.3f} cm (tier {signals[-1]['tier']})")

    # greedy backward elimination over branches: drop the one whose loss hurts least while within tolerance
    kept, removed, steps = set(range(st.n_branches)), [], []

    def total_probes(s):
        return int(sum(st.probes[c] for c in s))

    steps.append(dict(step=0, removed=None, remaining_probes=total_probes(kept),
                      boundary_cm=clean["boundary_mean_cm"], safe=clean["boundary_mean_cm"] <= TIER1_CM))
    while len(kept) > 1:
        cand = {s: st.score(st.predict(st.x, dead=removed + [s]))["boundary_mean_cm"] for s in kept}
        best = min(cand, key=cand.get)
        if cand[best] > TIER1_CM:
            break
        kept.remove(best); removed.append(best)
        steps.append(dict(step=len(steps), removed=R.SIGNAL_SHORT[best], remaining_probes=total_probes(kept),
                          boundary_cm=cand[best], safe=True))
        log(f"  drop {R.SIGNAL_SHORT[best]} -> {total_probes(kept)} probes, boundary {cand[best]:.3f} cm")
    data = dict(model=CFG["model"], failure_aware=CFG["failure_aware"], n_windows=int(len(st.truth)), threshold_cm=TIER1_CM,
                clean=dict(boundary_cm=clean["boundary_mean_cm"], axis_cm=clean["axis_shift_cm"], xpoint_cm=clean["xpoint_err_cm"]),
                total_probes=total_probes(range(st.n_branches)), signals=signals, elimination=steps,
                minimal_set=dict(kept=[R.SIGNAL_SHORT[c] for c in sorted(kept)], dropped=[R.SIGNAL_SHORT[c] for c in removed],
                                 probes=total_probes(kept), boundary_cm=steps[-1]["boundary_cm"]))
    path = os.path.join(st.out_dir, "criticality.json")
    json.dump(data, open(path, "w"), indent=1)
    log(f"criticality -> {path}")
    return data


# ------------------------------------------------------------------ Module 4
# the last four mirror the robustness-sweep scenarios the grand ablation table is built from (zero-filled damage)
SCENARIOS = ("clean", "transient_spike", "redundant_probe_dead", "critical_branch_dead",
             "group_loss", "channel_loss_6", "temporal_gap_40", "pre_disruption_25")


def inject(scenario, window, tiers, probes, rng):
    """Damage one window (list of per-branch numpy arrays, (T, P) or (T,)); returns the damaged copy."""
    w = [a.copy() for a in window]
    order = sorted(range(len(w)), key=lambda i: (tiers.get(i, 2), i))  # most critical first
    if scenario == "transient_spike":
        b = int(rng.integers(len(w)))
        flat = w[b].reshape(-1, w[b].shape[-1]) if w[b].ndim > 1 else w[b].reshape(-1, 1)
        flat[int(rng.integers(flat.shape[0])), int(rng.integers(flat.shape[1]))] = rng.choice([-1, 1]) * 12.0
        w[b] = flat.reshape(w[b].shape)
    elif scenario == "redundant_probe_dead":
        b = max((i for i in range(len(w)) if probes[i] > 2), key=lambda i: (tiers.get(i, 2), -i),
                default=None)  # the most redundant branch that has neighbouring probes
        if b is not None:
            w[b][..., int(rng.integers(1, w[b].shape[-1] - 1))] = 0.0
    elif scenario == "critical_branch_dead":
        w[order[0]][:] = 0.0
    elif scenario == "group_loss":  # all magnetics branches (flux loops, pickup coils, saddle loops), as the sweep's group_failure
        for b in R.CORRELATED_GROUPS["magnetics_all"]:
            w[b][:] = 0.0
    elif scenario == "channel_loss_6":
        for b in rng.choice(len(w), size=6, replace=False):
            w[b][:] = 0.0
    elif scenario in ("temporal_gap_40", "pre_disruption_25"):  # a block of the time axis on every sensor at once
        frac = 0.40 if scenario == "temporal_gap_40" else 0.25
        for b, a in enumerate(w):
            ax = -1 if probes[b] == 1 else -2
            n = a.shape[ax]
            k = max(1, int(round(frac * n)))
            s = n - k if scenario == "pre_disruption_25" else int(rng.integers(0, n - k + 1))
            sl = [slice(None)] * a.ndim
            sl[ax] = slice(s, s + k)
            a[tuple(sl)] = 0.0
    return w


def run_mitigation(st, tiers):
    torch = st.torch
    rng = np.random.default_rng(CFG["seed"])

    def model_fn_for(shapes):
        def fn(window, mask):
            x = [torch.as_tensor(a.reshape(s), dtype=torch.float32).unsqueeze(0) for a, s in zip(window, shapes)]
            pred = st.predict(x, mask=mask[None, :])
            if st.dev.type == "cuda":
                torch.cuda.synchronize()
            return pred[0]
        return fn

    shapes = [tuple(t.shape[1:]) for t in st.x]
    model_fn = model_fn_for(shapes)

    def windows_of(x, n):
        return [[x[b][i].numpy().astype(np.float64) for b in range(len(x))] for i in range(n)]

    def run_scenario(brain, windows, truth, scenario, rng):
        rows = []
        for w, t in zip(windows, truth):
            damaged = inject(scenario, w, tiers, st.probes, rng)
            naive_x = [torch.as_tensor(a, dtype=torch.float32).unsqueeze(0) for a in damaged]
            naive = st.predict(naive_x, mask=np.ones((1, len(damaged)), dtype=np.float32) if CFG["failure_aware"] else None)[0]
            res = brain.process(damaged, model_fn)
            rows.append((t, naive, res))
        return rows

    # ---- calibration on the validation split (different windows from the test sample)
    log("calibrating the conformal gauge on validation windows...")
    val_loader = P.build_data(st.split, ["val"])[0]["val"]
    vx, vy = sample_windows(val_loader, max(1, STRIDE // 2))
    vtruth = vy.numpy().astype(np.float64) * st.sd + st.mu
    vwin = windows_of(vx, len(vtruth))
    # probes that are zero in most clean validation windows are absent from the data itself, not failed
    dead_counts = None
    for w in vwin:
        rep = RealTimeFaultDetector().inspect(w)
        dead_counts = [r["dead"].astype(int) for r in rep] if dead_counts is None else [d + r["dead"] for d, r in zip(dead_counts, rep)]
    known_missing = [d >= 0.5 * len(vwin) for d in dead_counts]
    log("probes treated as permanently absent: " + ", ".join(f"{R.SIGNAL_SHORT[i]}={int(k.sum())}" for i, k in enumerate(known_missing)))
    cal_brain = SmartAdaptiveMitigationBrain(tiers, ConformalGauge(CALIB_ALPHA), known_missing)
    errors = {}
    for scen in SCENARIOS:
        for t, naive, res in run_scenario(cal_brain, vwin, vtruth, scen, rng):
            errors.setdefault(res["route"], []).append(compute_boundary_error(t, res["psi"])[1])  # Hausdorff, cm
    gauge = ConformalGauge(CALIB_ALPHA)
    deltas = gauge.calibrate(errors)
    log("conformal radius per route (cm): " + json.dumps({k: round(v, 3) for k, v in deltas.items()}))

    # ---- evaluation on the test sample
    brain = SmartAdaptiveMitigationBrain(tiers, gauge, known_missing)
    twin = windows_of(st.x, len(st.truth))
    out = dict(model=CFG["model"], failure_aware=CFG["failure_aware"], alpha=CALIB_ALPHA, n_windows=int(len(st.truth)),
              n_calibration=int(len(vtruth)) * len(SCENARIOS), delta_cm=deltas,
              absent_probes={R.SIGNAL_SHORT[i]: int(k.sum()) for i, k in enumerate(known_missing)}, scenarios={})
    for scen in SCENARIOS:
        rows = run_scenario(brain, twin, st.truth, scen, rng)
        recs = dict(naive=[], brain=[])
        alarms = dict(naive=[], brain=[], conservative=[])
        covered, lat, routes = [], [], {}
        for t, naive, res in rows:
            recs["naive"].append(evaluate_window_physics(t, naive))
            recs["brain"].append(evaluate_window_physics(t, res["psi"]))
            d_true = compute_wall_proximity(extract_lcfs_contour(t))
            hazard = d_true < SAFETY_CLEARANCE_CM
            a = res["assessment"]
            alarms["naive"].append((recs["naive"][-1]["d_wall_pred_cm"] < SAFETY_CLEARANCE_CM, hazard))
            alarms["brain"].append((a["point_alarm"], hazard))
            alarms["conservative"].append((a["alarm"], hazard))
            covered.append(recs["brain"][-1]["boundary_hausdorff_cm"] <= a["delta_cm"])
            lat.append(res["latencies_ms"])
            routes[res["route"]] = routes.get(res["route"], 0) + 1

        def rates(pairs):
            safe = [al for al, hz in pairs if not hz]; haz = [al for al, hz in pairs if hz]
            return dict(false_alarm_rate_pct=100.0 * sum(safe) / max(1, len(safe)), n_safe=len(safe),
                        missed_alarm_rate_pct=100.0 * sum(not al for al in haz) / max(1, len(haz)), n_hazard=len(haz))
        keys = ("detector", "router", "model", "gauge", "total")
        out["scenarios"][scen] = dict(
            routes=routes, naive=aggregate_physics_metrics(recs["naive"]), brain=aggregate_physics_metrics(recs["brain"]),
            alarms={k: rates(v) for k, v in alarms.items()}, delta_coverage_pct=100.0 * float(np.mean(covered)),
            latency_ms={k: dict(mean=float(np.mean([l[k] for l in lat])), p95=float(np.percentile([l[k] for l in lat], 95))) for k in keys})
        log(f"  {scen}: boundary naive {out['scenarios'][scen]['naive']['boundary_mean_cm']:.2f} -> brain "
            f"{out['scenarios'][scen]['brain']['boundary_mean_cm']:.2f} cm | routes {routes}")
    path = os.path.join(st.out_dir, "mitigation.json")
    json.dump(out, open(path, "w"), indent=1)
    write_mitigation_report(out, os.path.join(st.out_dir, "mitigation_report.md"))
    log(f"mitigation -> {path}")
    return out


def write_mitigation_report(out, path):
    lines = ["# Module 4: adaptive mitigation (measured)", "",
             f"CNN+{out['model'].upper()} ({'failure-aware' if out['failure_aware'] else 'baseline'}) | {out['n_windows']} test windows "
             f"per scenario | conformal level {100 * (1 - out['alpha']):.0f}% from {out['n_calibration']} calibration windows", "",
             "Conformal radius per route (cm): " + ", ".join(f"{k} {v:.2f}" for k, v in out["delta_cm"].items()), "",
             "| Scenario | Routes | Boundary, bare model (cm) | Boundary, with brain (cm) | False alarm: bare / brain / gauge (%) | "
             "Missed alarm: bare / brain / gauge (%) | delta coverage (%) | Loop latency mean / p95 (ms) |",
             "| :-- | :-- | --: | --: | :-- | :-- | --: | :-- |"]
    for name, s in out["scenarios"].items():
        al = s["alarms"]
        lines.append(f"| {name} | {', '.join(f'{k} {v}' for k, v in s['routes'].items())} | {s['naive']['boundary_mean_cm']:.2f} | "
                     f"{s['brain']['boundary_mean_cm']:.2f} | "
                     + " / ".join(f"{al[k]['false_alarm_rate_pct']:.1f}" for k in ("naive", "brain", "conservative")) + " | "
                     + " / ".join(f"{al[k]['missed_alarm_rate_pct']:.1f}" for k in ("naive", "brain", "conservative"))
                     + f" | {s['delta_coverage_pct']:.0f} | {s['latency_ms']['total']['mean']:.1f} / {s['latency_ms']['total']['p95']:.1f} |")
    lines += ["", "bare = the model on zero-filled damaged input; brain = after detector + router, point alarm; "
              "gauge = brain with the conformal margin (clearance minus delta). Rates are over windows whose true clearance is "
              "safe (false alarm) or below 5 cm (missed alarm); with no hazard windows the missed rate reads 0."]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["criticality", "mitigation", "all"])
    ap.add_argument("--split", default="random", choices=["random", "temporal"])
    a = ap.parse_args()
    st = Study(a.split)
    crit_path = os.path.join(st.out_dir, "criticality.json")
    if a.cmd in ("criticality", "all"):
        crit = run_criticality(st)
    if a.cmd in ("mitigation", "all"):
        crit = crit if a.cmd == "all" else json.load(open(crit_path)) if os.path.exists(crit_path) else None
        if crit is None:
            raise SystemExit(f"mitigation needs the tiers from {crit_path}: run `criticality` first")
        run_mitigation(st, {s["index"]: s["tier"] for s in crit["signals"]})


if __name__ == "__main__":
    main()
