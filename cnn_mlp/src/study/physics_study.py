#!/usr/bin/env python
"""Model-driven studies for Module 3 (sensor criticality) and Module 4 (adaptive mitigation) on a REAL trained model (v2).

Library + the Module 4 study. Module 3 lives in sensor_criticality.py (built on sensor_attribution.py).
Driver: run_module34_local.py (window cache -> stages m3 / m4); see STUDY_V2.md.

Scoring is Module 1 v2: EFIT-truth physics (evaluate_window_physics(..., truth=...), efit_truth.TruthIndex), always in "efit" mode.
Every window keeps its (shot_id, window_index). Windows without an EFIT LCFS (about 13 %) are dropped from the sets up front
(and counted), so every reported boundary error is against the EFIT separatrix.
The wall alarm uses the documented PROXY wall (the EFIT grid box; the data holds no vessel geometry): alarm rates are a protocol
demonstration, not a safety result.
"""
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from physics_evaluation import BOUNDARY_FAIL_CM, SAFETY_CLEARANCE_CM, evaluate_window_physics  # noqa: E402
from fault_models import FaultSpec, WindowLayout, as_tp, group_loss_specs, inject_window, random_specs  # noqa: E402
from adaptive_mitigation import (AdaptiveConformalGauge, ConformalGauge, FaultDetector, RouterV2,  # noqa: E402
                                 SmartAdaptiveMitigationBrain, confidence_flag, detection_from_labels,
                                 difficulty_features, empirical_coverage, summarise_latencies)
from virtual_sensor import LinearFluxFallback, fit_physics_fallback, fit_virtual_sensors  # noqa: E402

REC_KEYS = ("boundary_mean_cm", "boundary_hausdorff_cm", "axis_shift_cm", "xpoint_err_cm", "d_wall_true_cm", "d_wall_pred_cm",
            "is_false_alarm", "is_missed_alarm", "boundary_failed", "xpoint_true_found", "xpoint_pred_found", "kappa_err_field")
SIGNAL_SHORT = ("flux_loop", "ccbv", "obr", "obv", "saddle_v", "coil_i", "solenoid_i", "ip")
ALPHA = 0.1                      # conformal miscoverage (90 % intervals)
CPU_NOTE = "CPU (torch %s, %s threads); a GPU number comes from Kaggle / Modal"


def t():
    return time.strftime("%H:%M:%S")


# ======================================================================================================== windows
class WindowSet:
    """Cached model inputs/targets. x[b]: (N,1,1,T,n) for the six profile branches, (N,1,1,T) for solenoid and ip (standardised
    with the TokaMark train statistics, i.e. zero = the train mean = the pipeline's zero fill). y: (N,65,65) standardised psi."""

    def __init__(self, sid, wid, x, y):
        self.sid = np.asarray(sid, np.int64)
        self.wid = np.asarray(wid, np.int64)
        self.x = [np.asarray(v, np.float32) for v in x]
        self.y = np.asarray(y, np.float32).reshape(len(self.sid), 65, 65)

    def __len__(self):
        return len(self.sid)

    def subset(self, idx):
        idx = np.asarray(idx)
        return WindowSet(self.sid[idx], self.wid[idx], [v[idx] for v in self.x], self.y[idx])

    def window(self, i):
        """List of 8 (T_b, P_b) float64 arrays (the format of fault_models / adaptive_mitigation)."""
        return [as_tp(v[i])[0] for v in self.x]

    def put(self, windows):
        """List of N windows (as returned by window()) -> list of 8 arrays shaped like self.x."""
        return [np.stack([np.asarray(w[b], np.float32).reshape(self.x[b].shape[1:]) for w in windows]) for b in range(8)]

    def save(self, path):
        tmp = path + ".tmp.npz"
        np.savez(tmp, sid=self.sid, wid=self.wid, y=self.y, **{f"x{b}": v for b, v in enumerate(self.x)})
        os.replace(tmp, path)

    @staticmethod
    def load(path):
        z = np.load(path)
        return WindowSet(z["sid"], z["wid"], [z[f"x{b}"] for b in range(8)], z["y"])


class Net:
    """Frozen network on CPU: predict(x_list, health) -> psi in Wb/rad (N,65,65) float64."""

    def __init__(self, model, mu, sd, failure_aware=True, chunk=64):
        self.model, self.mu, self.sd, self.fa, self.chunk = model, mu, sd, failure_aware, chunk
        self.n_calls = 0

    def predict(self, x, health=None):
        import torch
        N = len(x[0])
        out = []
        with torch.no_grad():
            for s in range(0, N, self.chunk):
                xin = [torch.from_numpy(np.ascontiguousarray(v[s:s + self.chunk], dtype=np.float32)) for v in x]
                n = xin[0].shape[0]
                if self.fa:
                    h = torch.ones(n, 8) if health is None else torch.from_numpy(np.asarray(health[s:s + n], np.float32))
                    y = self.model(*xin, h)
                else:
                    y = self.model(*xin)
                if not isinstance(y, (list, tuple)):
                    y = [y]
                out.append(y[0].detach().cpu().numpy().reshape(n, 65, 65))
        self.n_calls += N
        return np.concatenate(out).astype(np.float64) * self.sd + self.mu


# ======================================================================================================== scoring
def _reduce(rec):
    return {k: (bool(rec[k]) if isinstance(rec[k], (bool, np.bool_)) else float(rec[k])) for k in REC_KEYS if k in rec}


def _score_one(args):
    yt, yp, tr = args
    try:
        return _reduce(evaluate_window_physics(yt, yp, truth=tr))
    except Exception as e:  # a failed score is recorded as such, never replaced by a number
        return dict(error=repr(e)[:80])


class Scorer:
    """EFIT-truth physics for batches of windows; optional process pool (workers are stateless: truth is passed along)."""

    def __init__(self, procs=0):
        self.pool = None
        self.procs = procs
        self.n_scored = 0
        self.n_errors = 0
        if procs and procs > 1:
            import multiprocessing as mp
            self.pool = mp.get_context("spawn").Pool(procs)

    def score(self, yt, yp, truths):
        jobs = list(zip(yt, yp, truths))
        recs = self.pool.map(_score_one, jobs, chunksize=max(1, len(jobs) // (4 * self.procs))) if self.pool else [_score_one(j) for j in jobs]
        self.n_scored += len(recs)
        self.n_errors += sum("error" in r for r in recs)
        return recs

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()


def rec_arrays(recs):
    """list of reduced records -> dict of float arrays (bool -> 0/1, missing/failed -> nan)."""
    out = {}
    for k in REC_KEYS:
        out[k] = np.array([float(r.get(k, np.nan)) for r in recs], float)
    return out


def window_rmse(yt, yp):
    return np.sqrt(np.mean((np.asarray(yp) - np.asarray(yt)) ** 2, axis=(1, 2)))


def nrmse_official(shots, rmse, sd):
    """TokaMark: per-window RMSE -> per-shot sqrt(mean(rmse^2)) -> / output std -> mean over shots."""
    shots, rmse = np.asarray(shots), np.asarray(rmse, float)
    u = np.unique(shots)
    return float(np.mean([np.sqrt(np.mean(rmse[shots == s] ** 2)) / sd for s in u]))


def shot_bootstrap(shots, boundary, rmse, sd, n_boot=1000, seed=0, extra=None):
    """Point estimate + 95 % shot-level bootstrap CI of the pooled-window mean boundary error and the official NRMSE.
    `extra` optionally maps name -> per-window array whose pooled mean (ignoring nan) is bootstrapped too."""
    shots, boundary, rmse = np.asarray(shots), np.asarray(boundary, float), np.asarray(rmse, float)
    u = np.unique(shots)
    ok = np.isfinite(boundary)
    sb = np.array([boundary[(shots == s) & ok].sum() for s in u])
    cb = np.array([((shots == s) & ok).sum() for s in u], float)
    sr = np.array([np.sqrt(np.mean(rmse[shots == s] ** 2)) for s in u]) / sd
    ex = {k: (np.array([np.nansum(np.asarray(v, float)[shots == s]) for s in u]),
              np.array([np.isfinite(np.asarray(v, float)[shots == s]).sum() for s in u], float)) for k, v in (extra or {}).items()}
    rng = np.random.default_rng(seed)
    bs, ns, exs = [], [], {k: [] for k in ex}
    for _ in range(n_boot):
        i = rng.integers(0, len(u), len(u))
        bs.append(sb[i].sum() / max(cb[i].sum(), 1.0))
        ns.append(sr[i].mean())
        for k, (a, c) in ex.items():
            exs[k].append(a[i].sum() / max(c[i].sum(), 1.0))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    out = dict(boundary_cm=float(sb.sum() / max(cb.sum(), 1.0)), boundary_ci95=q(bs), nrmse=float(sr.mean()), nrmse_ci95=q(ns),
               n_windows=int(len(shots)), n_shots=int(len(u)), n_boundary_valid=int(ok.sum()))
    for k, (a, c) in ex.items():
        out[k] = float(a.sum() / max(c.sum(), 1.0))
        out[k + "_ci95"] = q(exs[k])
    return out


# ======================================================================================================== context
class Ctx:
    pass


def make_context(P, ckpt_dir, data_dir, cache_dir, tag, procs="0", **kw):
    """Load the frozen checkpoint and the cached window sets (train / val / test); keep only windows with an EFIT LCFS."""
    import torch
    from efit_truth import TruthIndex
    c = Ctx()
    c.tag, c.data_dir, c.cache_dir = tag, data_dir, cache_dir
    torch.set_num_threads(int(kw.get("threads", 4)))
    raw = {s: WindowSet.load(os.path.join(cache_dir, f"{s}.npz")) for s in ("train", "val", "test")
           if os.path.exists(os.path.join(cache_dir, f"{s}.npz"))}
    first = next(iter(raw))
    loaders, meta = P.build_data("random", [first if first in ("val", "test", "train") else "test"], shuffle_windows=False)
    m = meta["output"]["equilibrium-psi"]
    c.mu, c.sd = float(m["mean"]), float(m["std"])
    shapes = ([tuple(v.shape[1:]) for v in raw[first].x], [], [(1, 65, 65)])
    model = P.make_model(None, meta, shapes=shapes)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "model", "best_model.pt"), map_location="cpu", weights_only=False))
    model.eval()
    c.net = Net(model, c.mu, c.sd, failure_aware=P.CFG["failure_aware"])
    c.backbone = P.CFG["model"]
    c.sensor_masks = c.backbone == "transformer"     # the Transformer takes a per-sensor (115) mask; the CNNs only a per-branch (8) one
    c.n_params = int(sum(p.numel() for p in model.parameters()))
    c.truth = TruthIndex(data_dir, outlier_yaml=P.split_files("random")[2])
    c.sets, c.truths, c.dropped = {}, {}, {}
    for name, ws in raw.items():
        keep, trs = [], []
        for i in range(len(ws)):
            tr = c.truth.get(ws.sid[i], ws.wid[i])
            if tr is not None and tr.get("lcfs_rz_m") is not None:
                keep.append(i)
                trs.append(tr)
        c.dropped[name] = dict(n_cached=len(ws), n_efit_valid=len(keep), n_dropped_no_efit_lcfs=len(ws) - len(keep))
        c.sets[name] = ws.subset(keep)
        c.truths[name] = trs
    c.scorer = Scorer(int(procs))
    c.torch_threads = torch.get_num_threads()
    c.torch_version = torch.__version__
    return c


def yt_wb(ctx, ws):
    return ws.y.astype(np.float64) * ctx.sd + ctx.mu


def score_pred(ctx, ws, truths, yp):
    """Physics records + official-NRMSE ingredients for predictions yp (N,65,65) in Wb against the window set's targets."""
    yt = yt_wb(ctx, ws)
    recs = ctx.scorer.score(yt, yp, truths)
    a = rec_arrays(recs)
    a["rmse"] = window_rmse(yt, yp)
    return a


# ======================================================================================================== Module 3
def run_criticality(ctx, out, **kw):
    import sensor_criticality as SC
    return SC.run_module3(ctx, out, **kw)


# ======================================================================================================== Module 4
SCENARIOS = ("clean", "channel_loss", "channel_loss_6", "group_loss", "temporal_gap", "pre_disruption", "sensor_loss",
             "stuck", "drift", "spike")
TABLE_COLUMNS = {"Clean": "clean", "Channel loss": "channel_loss_6", "Group loss": "group_loss", "Temporal gap": "temporal_gap",
                 "Pre-disruption": "pre_disruption"}
FAULT_SCENARIOS = tuple(s for s in SCENARIOS if s != "clean")
MAGNETICS_ALL = (0, 1, 2, 3, 4)   # the harness group_failure "magnetics_all"


def make_faults(scen, L, rng, usable, channel_k=3):
    """Fault specs of one scenario for one window. Definitions follow the robustness harness (pipeline_robustness.py):
    channel_loss      channel_k whole input signals (branches) zeroed for the whole window  (Gupta channel_ablation)
    group_loss        magnetics_all = flux loops, CCBV, OBR, OBV, saddle loops zeroed       (group_failure)
    temporal_gap      a contiguous 40 % of the window zeroed on every probe, random position (temporal_gap 0.4)
    pre_disruption    the LAST 25 % of the window zeroed on every probe                      (disruption_proximate 0.25)
    sensor_loss       6 random individual probes zeroed for the whole window                  (sensor level, extra)
    stuck / drift / spike  3 random probes, random onset (fault_models); stuck is half rails, half frozen values."""
    ms = 5.0
    if scen == "clean":
        return []
    if scen.startswith("channel_loss"):
        k = 6 if scen.endswith("_6") else channel_k
        return group_loss_specs(L, sorted(rng.choice(8, size=k, replace=False).tolist()))
    if scen == "group_loss":
        return group_loss_specs(L, MAGNETICS_ALL)
    if scen in ("temporal_gap", "pre_disruption"):
        if scen == "temporal_gap":
            on = float(rng.uniform(0, ms * 0.6))
            return [FaultSpec(int(c), "dropout", on, ms * 0.4) for c in range(L.n_channels)]
        return [FaultSpec(int(c), "dropout", ms * 0.75, None) for c in range(L.n_channels)]
    if scen == "sensor_loss":
        return [FaultSpec(int(c), "dropout", 0.0, None) for c in rng.choice(usable, size=6, replace=False)]
    if scen == "stuck":
        return [s for kind in (rng.choice(["stuck_rail", "stuck_value"]) for _ in range(3)) for s in random_specs(rng, 1, str(kind), 1, channels=usable)]
    if scen in ("drift", "spike"):
        return random_specs(rng, 1, scen, 3, channels=usable)
    raise ValueError(scen)


def impute_window(win, L, labels, mode, mean_c):
    """Gupta-style fixed imputation of the KNOWN holes (labels.mask): zero | mean (train channel mean) | ffill (last good sample)."""
    M = labels.mask
    if not M.any():
        return win
    X = L.flatten(win)
    Y = X.copy()
    if mode == "zero":
        Y[M] = 0.0
    elif mode == "mean":
        Y = np.where(M, mean_c[None, :], X)
    elif mode == "ffill":
        for c in np.flatnonzero(M.any(axis=0)):
            last = 0.0
            for i in range(X.shape[0]):
                if M[i, c]:
                    Y[i, c] = last
                else:
                    last = X[i, c]
    else:
        raise ValueError(mode)
    return L.write_back(win, Y, M)


class Mitigators:
    """Everything fitted on TRAIN windows (detector statistics, virtual sensors, physics fallback) + val_A (detector thresholds)."""

    def __init__(self, ctx, train, cal, tiers=None, router_kw=None):
        self.ctx = ctx
        w0 = train.window(0)
        self.L = L = WindowLayout(w0)
        t0 = time.time()
        self.Xtr = np.stack([L.flatten(train.window(i)) for i in range(len(train))])
        Xcal = np.stack([L.flatten(cal.window(i)) for i in range(len(cal))])
        self.det = FaultDetector().fit(self.Xtr, Xcal)
        self.bank = fit_virtual_sensors(self.Xtr)
        feats = np.stack([LinearFluxFallback.features(x) for x in self.Xtr])
        self.fb = fit_physics_fallback(feats, yt_wb(ctx, train))
        self.mean_c = np.nanmean(self.Xtr.reshape(-1, L.n_channels), axis=0)
        self.usable = np.flatnonzero(~self.det.absent)
        self.tiers = tiers
        self.router_kw = router_kw or {}
        self.router = RouterV2(tiers, L, self.bank, self.fb, **self.router_kw)
        self.sensor_level = bool(getattr(ctx, "sensor_masks", False))   # Transformer: announce faults per sensor, not per branch
        self.long_fault = "router"
        self.fit_seconds = time.time() - t0

    def set_policy(self, thr_scale=1.0, short_ms=1.0, max_masked=5, critical_policy="mask", long_fault="router"):
        """Operating point: detector residual thresholds x thr_scale; forward-fill up to short_ms; physics fallback beyond max_masked
        masked branches. These are the router's 'policy placeholders' (MODULE4_V2.md), chosen on validation shots in run_mitigation."""
        if not hasattr(self, "thr0"):
            self.thr0 = np.array(self.det.thr, float)
        self.det.thr = self.thr0 * float(thr_scale)
        self.router_kw = dict(short_ms=float(short_ms), max_masked_branches=int(max_masked), critical_policy=critical_policy)
        self.router = RouterV2(self.tiers, self.L, self.bank, self.fb, **self.router_kw)
        self.long_fault = long_fault     # sensor-level models only: "router" = virtual sensors where the router decides, "mask" = mask every long fault
        self.policy = dict(thr_scale=float(thr_scale), short_ms=float(short_ms), max_masked_branches=int(max_masked), critical_policy=critical_policy)
        if self.sensor_level:
            self.policy["long_fault"] = long_fault

    def brain(self, gauge, oracle=False):
        return SmartAdaptiveMitigationBrain(self.tiers, gauge, detector=None if oracle else self.det, bank=self.bank,
                                            fallback=self.fb, router_kw=self.router_kw, oracle=oracle)

    def prepare(self, win, labels, oracle):
        """detect (or oracle) -> decide -> repair. Returns (repaired window, health (8,), route, Detection, Decision, psi_fallback|None)."""
        det = detection_from_labels(labels) if oracle else self.det.detect_window(win, self.L)
        if oracle and (det.absent is None or det.absent.size != self.L.n_channels):
            det.absent = np.zeros(self.L.n_channels, bool)
        dec = self.router.decide(det, self.L)
        if self.sensor_level:
            return self._prepare_sensor_level(win, det, dec)
        rep = self.router.apply(win, det, dec, self.L)
        psi = None
        if dec.use_fallback:
            avail = (~det.flagged | dec.transient | dec.virtual) & ~dec.lost & ~det.absent
            psi = self.fb.predict(self.fb.features(self.L.flatten(rep)), avail)
            dec.route = "physics_fallback"
        return rep, dec.branch_mask, dec.route, det, dec, psi


def _prepare_sensor_level(self, win, det, dec):
    """Per-sensor variant of the router for a model that accepts a per-sensor health mask: forward-fill the short transients (router
    decision), repair with virtual sensors where the router decides (long_fault='router'), and MASK every other flagged channel
    individually instead of the whole branch. Physics fallback when more than max_masked_branches branches are fully masked."""
    L = self.L
    # labels (Oracle) do not list probes that are dead for the whole shot; the fitted dead-probe set is known to both variants
    absent = det.absent | self.det.absent
    bad = det.flagged & ~absent
    transient = dec.transient & bad
    virtual = (dec.virtual & bad) if self.long_fault == "router" else np.zeros_like(bad)
    masked = bad & ~transient & ~virtual
    n_full = 0
    bm = np.ones(L.n_branches, np.float32)
    for b in range(L.n_branches):
        ch = L.channels_of(b)
        use = ch[~absent[ch]]
        if use.size and masked[use].all():
            n_full += 1
            bm[b] = 0.0
    dec2 = dataclasses.replace(dec, branch_mask=np.ones(L.n_branches, np.float32), virtual=virtual, lost=np.zeros_like(dec.lost))
    rep = self.router.apply(win, det, dec2, L)
    for c in np.flatnonzero(masked):                       # masked tokens are ignored by the model; blank them anyway
        rep[L.branch_of[c]][:, L.probe_of[c]] = 0.0
    route = ("healthy" if not bad.any() else "module2" if masked.any() else "virtual_sensor" if virtual.any() else "transient_ffill")
    psi = None
    if n_full > self.router.max_masked_branches and self.fb is not None:
        avail = ~masked & ~absent
        psi = self.fb.predict(self.fb.features(L.flatten(rep)), avail)
        route = "physics_fallback"
    dec2 = dataclasses.replace(dec2, route=route, branch_mask=bm, lost=masked, virtual=virtual, use_fallback=psi is not None)
    return rep, (~masked).astype(np.float32), route, det, dec2, psi


Mitigators._prepare_sensor_level = _prepare_sensor_level


def detection_stats(det, labels, base=None):
    """tp / fn / fp against the INJECTED labels; fp_new excludes channels the detector already flags on the un-injected window
    (`base`): the real data contain probes that are dead / zero for a whole shot, which are genuine faults, not false alarms."""
    f, g = det.flagged & ~det.absent, labels.faulty
    b = np.zeros_like(f) if base is None else (base.flagged & ~base.absent)
    return dict(tp=int((f & g).sum()), fn=int((~f & g).sum()), fp=int((f & ~g).sum()), fp_new=int((f & ~g & ~b).sum()),
                n_preexisting=int(b.sum()), tp_new=int((f & g & ~b).sum()), fn_new=int((~f & g & ~b).sum()),
                iso_top1_new=bool(g.any() and g[int(np.argmax(np.where(b, -1.0, det.score)))]),
                iso_top1=bool(g.any() and g[int(np.argmax(det.score))]), has_fault=bool(g.any()), any_flag=bool(f.any()))


def run_variants(ctx, mit, ws, truths, scen, seed, variants, channel_k=3, tag=""):
    """Damage every window of `ws` with scenario `scen` and score the requested variants.
    variants subset of: none known_mask imp_zero imp_mean imp_ffill adaptive oracle.
    Returns (per-variant arrays, per-window meta: route/difficulty/detection stats for adaptive & oracle)."""
    L = mit.L
    rng = np.random.default_rng(seed)
    N = len(ws)
    wins_by_v = {v: [] for v in variants}
    H = 115 if getattr(ctx, "sensor_masks", False) else 8
    health = {v: np.ones((N, H), np.float32) for v in variants}
    fb_psi = {v: {} for v in ("adaptive", "oracle") if v in variants}
    meta = {v: dict(route=[], feats=[], det=[]) for v in ("adaptive", "oracle") if v in variants}
    for i in range(N):
        win = ws.window(i)
        specs = make_faults(scen, L, rng, mit.usable, channel_k)
        dwin, lab = inject_window(win, L, specs) if specs else (win, inject_window(win, L, [])[1])
        for v in variants:
            if v == "none":
                wins_by_v[v].append(dwin)
            elif v == "known_mask":
                wins_by_v[v].append(dwin)
                health[v][i] = (~lab.faulty).astype(np.float32) if H == 115 else L.branch_mask(lab.faulty, 0.5)
            elif v.startswith("imp_"):
                wins_by_v[v].append(impute_window(dwin, L, lab, v[4:], mit.mean_c))
            else:
                rep, h, route, det, dec, psi = mit.prepare(dwin, lab, oracle=(v == "oracle"))
                wins_by_v[v].append(rep)
                health[v][i] = h
                if psi is not None:
                    fb_psi[v][i] = psi
                meta[v]["route"].append(route)
                meta[v]["feats"].append(difficulty_features(det, dec))
                if v == "adaptive" and scen != "clean":
                    base = mit.det.detect_window(win, L)
                else:
                    base = det if (v == "adaptive") else None
                meta[v]["det"].append(detection_stats(det, lab, base))
    res = {}
    for v in variants:
        yp = ctx.net.predict(ws.put(wins_by_v[v]), health[v])
        for i, psi in fb_psi.get(v, {}).items():
            yp[i] = psi
        res[v] = score_pred(ctx, ws, truths, yp)
        res[v]["shot"] = ws.sid
    return res, meta


def pick_set(ctx, name, n, seed=0):
    """n windows spread evenly over the shots of a window set (round-robin by shot, fixed order)."""
    ws, trs = ctx.sets[name], ctx.truths[name]
    rng = np.random.default_rng(seed)
    by = {}
    for i, s in enumerate(ws.sid):
        by.setdefault(int(s), []).append(i)
    for s in by:
        rng.shuffle(by[s])
    idx, k = [], 0
    while len(idx) < min(n, len(ws)):
        for s in sorted(by):
            if k < len(by[s]) and len(idx) < n:
                idx.append(by[s][k])
        k += 1
    idx = np.array(sorted(idx))
    return ws.subset(idx), [trs[i] for i in idx], idx


def split_by_shot(ws, trs):
    """Two window sets on disjoint SHOTS (alternate shots)."""
    shots = np.unique(ws.sid)
    A = set(shots[::2].tolist())
    ia = [i for i, s in enumerate(ws.sid) if int(s) in A]
    ib = [i for i, s in enumerate(ws.sid) if int(s) not in A]
    return (ws.subset(ia), [trs[i] for i in ia]), (ws.subset(ib), [trs[i] for i in ib])


def split_val_by_shot(ctx):
    """val_A (detector thresholds, router / imputation choice) and val_B (conformal calibration): disjoint SHOTS."""
    return split_by_shot(ctx.sets["val"], ctx.truths["val"])


def tiers_from_criticality(path, cuts=(1.0, 0.25)):
    """Module 3 -> Module 4: tier 1 if losing the branch (group drop-one, failure-aware model told) costs >= cuts[0] cm of boundary
    error, tier 2 if >= cuts[1], else tier 3. None (-> DEFAULT_TIERS) when criticality.json is absent."""
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    dl = d["group"]["drop_one"]
    return {b: (1 if v is not None and v >= cuts[0] else 2 if v is not None and v >= cuts[1] else 3) for b, v in enumerate(dl)}


def clearance_alarm(a, delta=None):
    """Alarm decisions from the EFIT-mode clearances: true = EFIT clearance < 5 cm (PROXY wall); predicted = clearance - delta < 5 cm."""
    d_true, d_pred = a["d_wall_true_cm"], a["d_wall_pred_cm"]
    dl = 0.0 if delta is None else np.asarray(delta, float)
    pred = (d_pred - dl) < SAFETY_CLEARANCE_CM
    true = d_true < SAFETY_CLEARANCE_CM
    ok = np.isfinite(d_true) & np.isfinite(d_pred)
    return dict(n=int(ok.sum()), n_true_alarm=int((true & ok).sum()), false_alarm_rate=float((pred & ~true & ok).sum() / max(1, (~true & ok).sum())),
                missed_alarm_rate=float((~pred & true & ok).sum() / max(1, (true & ok).sum())),
                false_alarms=int((pred & ~true & ok).sum()), missed=int((~pred & true & ok).sum()))


def _jf(o):
    if isinstance(o, dict):
        return {str(k): _jf(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jf(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jf(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_mitigation(ctx, out, n_val_a=150, n_val_b=300, n_test=300, n_train=800, n_lat=120, channel_k=3, seed=0, tiers_json="",
                   n_boot=1000, **kw):
    """Module 4 study on the real network. Fit on TRAIN shots, choose the imputation / calibrate the detector on val_A shots,
    calibrate the conformal gauge on val_B shots, report on TEST shots (disjoint shots at every step)."""
    n_val_a, n_val_b, n_test, n_train, n_lat, channel_k, seed, n_boot = (int(v) for v in (n_val_a, n_val_b, n_test, n_train, n_lat, channel_k, seed, n_boot))
    t0 = time.time()
    log = lambda *a: print(t(), *a, flush=True)  # noqa: E731
    tr_n = min(n_train, len(ctx.sets["train"]))
    train = ctx.sets["train"].subset(np.linspace(0, len(ctx.sets["train"]) - 1, tr_n).astype(int))
    (vaA, trA), (vaB, trB) = split_val_by_shot(ctx)
    (vaA1, trA1), (vaA2, trA2) = split_by_shot(vaA, trA)          # A1: detector thresholds; A2: policy + imputation choice
    cal1, cal1_t, _ = _sub(vaA1, trA1, n_val_a, seed)
    cal, trA, _ = _sub(vaA2, trA2, n_val_a, seed)
    vb, vb_t, _ = _sub(vaB, trB, n_val_b, seed + 1)
    te, te_t, _ = pick_set(ctx, "test", n_test, seed + 2)
    sets_info = dict(train=dict(n=len(train), shots=int(len(np.unique(train.sid)))), val_A=dict(n=len(cal), shots=int(len(np.unique(cal.sid)))),
                     val_B=dict(n=len(vb), shots=int(len(np.unique(vb.sid)))), test=dict(n=len(te), shots=int(len(np.unique(te.sid)))))
    assert not (set(np.unique(train.sid)) & set(np.unique(te.sid))) and not (set(np.unique(cal.sid)) & set(np.unique(vb.sid)))
    assert not (set(np.unique(cal1.sid)) & set(np.unique(cal.sid)))
    log("sets", sets_info)
    tiers = tiers_from_criticality(tiers_json) if tiers_json else None
    mit = Mitigators(ctx, train, cal1, tiers=({int(k): v for k, v in tiers.items()} if tiers else None))
    log(f"mitigators fitted in {mit.fit_seconds:.1f}s; tiers={tiers or 'DEFAULT_TIERS'}; usable channels {len(mit.usable)}/{mit.L.n_channels}")
    imp = ["imp_zero", "imp_mean", "imp_ffill"]

    # ---- 0. operating point (detector threshold scale, forward-fill horizon, fallback crossover) chosen on val_A2 shots
    if mit.sensor_level:      # per-sensor model: virtual-sensor repair vs masking every long fault; critical-tier rule not used
        grid = [(4.0, b, c, "mask", lf) for b in (1.0, 2.5) for c in (5, 8) for lf in ("router", "mask")]
    elif str(kw.get("grid", "full")) == "small":   # thr_scale had no effect in the first study: fixed at 4
        grid = [(4.0, b, c, d, "router") for b in (1.0, 2.5) for c in (5, 8) for d in ("mask", "virtual")]
    else:
        grid = [(a, b, c, d, "router") for a in (1.0, 4.0) for b in (1.0, 2.5) for c in (5, 8) for d in ("mask", "virtual")]
    n_sel = min(len(cal), int(kw.get("n_sel", 80)))
    csel = cal.subset(np.linspace(0, len(cal) - 1, n_sel).astype(int))
    tsel = [trA[i] for i in np.linspace(0, len(cal) - 1, n_sel).astype(int)]
    pol_rows = []
    for gi, (a, b, c, d, lf) in enumerate(grid):
        mit.set_policy(a, b, c, d, lf)
        per = {}
        for si, scen in enumerate(SCENARIOS):
            r, _ = run_variants(ctx, mit, csel, tsel, scen, seed + 50 + si, ["adaptive"], channel_k)
            per[scen] = float(np.nanmean(r["adaptive"]["boundary_mean_cm"]))
        pol_rows.append(dict(thr_scale=a, short_ms=b, max_masked_branches=c, critical_policy=d, long_fault=lf, mean_over_scenarios=float(np.mean(list(per.values()))), per_scenario=per))
        log(f"policy {gi + 1}/{len(grid)} thr x{a} short {b} ms max_masked {c} critical {d} long {lf}: mean boundary {pol_rows[-1]['mean_over_scenarios']:.2f} cm")
    best_pol = min(pol_rows, key=lambda r: r["mean_over_scenarios"])
    mit.set_policy(best_pol["thr_scale"], best_pol["short_ms"], best_pol["max_masked_branches"], best_pol["critical_policy"], best_pol["long_fault"])
    log("chosen policy", mit.policy)

    # ---- 1. imputation choice on val_A (the "best fixed imputation" row never sees the test shots)
    best_imp, val_imp = {}, {}
    for si, scen in enumerate(SCENARIOS):
        r, _ = run_variants(ctx, mit, cal, trA, scen, seed + 100 + si, imp, channel_k)
        val_imp[scen] = {v: float(np.nanmean(r[v]["boundary_mean_cm"])) for v in imp}
        best_imp[scen] = min(imp, key=lambda v: val_imp[scen][v])
        log(f"val_A {scen}: imputation boundary cm {val_imp[scen]} -> {best_imp[scen]}")

    # ---- 2. conformal calibration on val_B: adaptive pipeline, all scenarios
    cal_err, cal_err_m, cal_feat, cal_route, cal_scen = [], [], [], [], []
    for si, scen in enumerate(SCENARIOS):
        r, m = run_variants(ctx, mit, vb, trB, scen, seed + 200 + si, ["adaptive"], channel_k)
        e = r["adaptive"]["boundary_hausdorff_cm"]
        ok = np.isfinite(e)
        cal_err += e[ok].tolist()
        cal_err_m += r["adaptive"]["boundary_mean_cm"][ok].tolist()
        cal_feat += [f for f, o in zip(m["adaptive"]["feats"], ok) if o]
        cal_route += [x for x, o in zip(m["adaptive"]["route"], ok) if o]
        cal_scen += [scen] * int(ok.sum())
        log(f"val_B {scen}: n={int(ok.sum())} mean hausdorff {np.nanmean(e):.2f} cm, routes {dict(zip(*np.unique(m['adaptive']['route'], return_counts=True)))}")
    g_global = ConformalGauge(alpha=ALPHA)
    by_route = {}
    for e, r_ in zip(cal_err, cal_route):
        by_route.setdefault(r_, []).append(e)
    g_global.calibrate(by_route)
    g_global.global_delta_all = float(np.sort(cal_err)[min(len(cal_err), math.ceil((len(cal_err) + 1) * (1 - ALPHA))) - 1])
    g_adapt = AdaptiveConformalGauge(alpha=ALPHA)
    g_adapt.calibrate(by_route)
    g_adapt.calibrate_adaptive(np.array(cal_err), np.array(cal_feat))
    # second gauge on the MEAN boundary displacement (typical error); the Hausdorff gauge above bounds the clearance error
    g_global_m = ConformalGauge(alpha=ALPHA)
    by_route_m = {}
    for e, r_ in zip(cal_err_m, cal_route):
        by_route_m.setdefault(r_, []).append(e)
    g_global_m.calibrate(by_route_m)
    g_adapt_m = AdaptiveConformalGauge(alpha=ALPHA)
    g_adapt_m.calibrate(by_route_m)
    g_adapt_m.calibrate_adaptive(np.array(cal_err_m), np.array(cal_feat))
    log("conformal: per-route delta (cm)", {k: round(v, 2) for k, v in g_global.delta.items()}, "| marginal", round(g_global.global_delta_all, 2),
        "| adaptive q", round(g_adapt.q, 2))

    # ---- 3. test: every scenario x variant
    variants = ["none", "known_mask"] + imp + ["adaptive", "oracle"]
    cells, det_rows, route_use, conf_rows, flag_rows = {}, {}, {}, {}, {}
    test_arrays = {}
    for si, scen in enumerate(SCENARIOS):
        r, m = run_variants(ctx, mit, te, te_t, scen, seed + 300 + si, variants, channel_k)
        r["best_imp"] = r[best_imp[scen]]
        cells[scen] = {}
        for v in ["none", "known_mask", "best_imp", "adaptive", "oracle"] + imp:
            a = r[v]
            cells[scen][v] = shot_bootstrap(te.sid, a["boundary_mean_cm"], a["rmse"], ctx.sd, n_boot, seed,
                                            extra=dict(hausdorff_cm=a["boundary_hausdorff_cm"], axis_shift_cm=a["axis_shift_cm"],
                                                       xpoint_err_cm=a["xpoint_err_cm"]))
            cells[scen][v]["alarms_point"] = clearance_alarm(a)
        cells[scen]["best_imp"]["chosen"] = best_imp[scen]
        for v in ("adaptive", "oracle"):
            ds = m[v]["det"]
            tp, fn, fp, fpn, tpn, fnn = (sum(d[k] for d in ds) for k in ("tp", "fn", "fp", "fp_new", "tp_new", "fn_new"))
            hf = [d for d in ds if d["has_fault"]]
            det_rows.setdefault(scen, {})[v] = dict(
                recall=tp / max(1, tp + fn), precision=tp / max(1, tp + fp), tp=tp, fn=fn, fp=fp,
                precision_excl_preexisting=tp / max(1, tp + fpn), recall_excl_preexisting=tpn / max(1, tpn + fnn), fp_new=fpn,
                preexisting_flagged_channels_per_window=float(np.mean([d["n_preexisting"] for d in ds])) if v == "adaptive" else None,
                isolation_top1_excl_preexisting=float(np.mean([d["iso_top1_new"] for d in hf])) if hf else None,
                isolation_top1=float(np.mean([d["iso_top1"] for d in hf])) if hf else None,
                window_flag_rate=float(np.mean([d["any_flag"] for d in ds])))
            route_use.setdefault(scen, {})[v] = {k: int(c) for k, c in zip(*np.unique(m[v]["route"], return_counts=True))}
        # confidence for the adaptive output
        a = r["adaptive"]
        feats = np.array(m["adaptive"]["feats"])
        routes = m["adaptive"]["route"]
        d_glob = np.array([g_global.delta_for(x) for x in routes])
        d_marg = np.full(len(routes), g_global.global_delta_all)
        d_adap = np.array([g_adapt.delta_for(x, f) for x, f in zip(routes, feats)])
        err = a["boundary_hausdorff_cm"]
        ok = np.isfinite(err)
        flags = [confidence_flag(c, d, valid=bool(o)) for c, d, o in zip(a["d_wall_pred_cm"], d_adap, ok)]
        conf_rows[scen] = dict(
            coverage_marginal=empirical_coverage(d_marg[ok], err[ok]), coverage_route_global=empirical_coverage(d_glob[ok], err[ok]),
            coverage_adaptive=empirical_coverage(d_adap[ok], err[ok]), mean_delta_marginal=float(d_marg.mean()),
            mean_delta_route_global=float(d_glob.mean()), mean_delta_adaptive=float(d_adap.mean()),
            n=int(ok.sum()), alarms_with_delta=clearance_alarm(a, d_adap), alarms_point=clearance_alarm(a))
        em = a["boundary_mean_cm"]
        okm = np.isfinite(em)
        dm_g = np.array([g_global_m.delta_for(x) for x in routes])
        dm_a = np.array([g_adapt_m.delta_for(x, f) for x, f in zip(routes, feats)])
        conf_rows[scen]["mean_error_gauge"] = dict(coverage_route_global=empirical_coverage(dm_g[okm], em[okm]), coverage_adaptive=empirical_coverage(dm_a[okm], em[okm]),
                                                   mean_delta_route_global=float(dm_g.mean()), mean_delta_adaptive=float(dm_a.mean()))
        flag_rows[scen] = {k: int(sum(f == k for f in flags)) for k in ("HIGH", "LOW", "ABSTAIN")}
        test_arrays[scen] = dict(err=err, d_glob=d_glob, d_adap=d_adap, d_marg=d_marg, routes=np.array(routes), err_m=em, dm_g=dm_g, dm_a=dm_a)
        log(f"test {scen}: " + " | ".join(f"{v} {cells[scen][v]['boundary_cm']:.2f}cm/{cells[scen][v]['nrmse']:.3f}" for v in
                                         ("none", "best_imp", "adaptive", "oracle")) + f" | cov(adapt) {conf_rows[scen]['coverage_adaptive']:.2f}")
    allerr = np.concatenate([x["err"] for x in test_arrays.values()])
    alld = {k: np.concatenate([x[k] for x in test_arrays.values()]) for k in ("d_glob", "d_adap", "d_marg")}
    ok = np.isfinite(allerr)
    cov_all = {k: empirical_coverage(v[ok], allerr[ok]) for k, v in alld.items()}
    em_all = np.concatenate([x["err_m"] for x in test_arrays.values()])
    okm = np.isfinite(em_all)
    cov_all_mean = {k: empirical_coverage(np.concatenate([x[k] for x in test_arrays.values()])[okm], em_all[okm]) for k in ("dm_g", "dm_a")}
    # coverage by route / by difficulty tercile (the adaptive gauge should narrow easy windows)
    routes_all = np.concatenate([x["routes"] for x in test_arrays.values()])
    cov_route = {r_: dict(n=int(((routes_all == r_) & ok).sum()), global_=empirical_coverage(alld["d_glob"][(routes_all == r_) & ok], allerr[(routes_all == r_) & ok]),
                          adaptive=empirical_coverage(alld["d_adap"][(routes_all == r_) & ok], allerr[(routes_all == r_) & ok]))
                 for r_ in np.unique(routes_all) if ((routes_all == r_) & ok).sum() > 0}

    # ---- 4. latency with the real network (CPU, batch 1), per-window brain.process
    lat = latency_study(ctx, mit, g_adapt, te, n_lat, seed)
    log("latency", {k: round(v["p50"], 2) for k, v in lat["summary"].items()})

    res = dict(model=ctx.tag, device=CPU_NOTE % (ctx.torch_version, ctx.torch_threads), sets=sets_info, dropped_no_efit_lcfs=ctx.dropped,
               alpha=ALPHA, scenarios=list(SCENARIOS), table_columns=TABLE_COLUMNS, channel_k=channel_k,
               scenario_definitions=make_faults.__doc__, policy_selection_val_A2=dict(grid=pol_rows, chosen=mit.policy, n_windows=n_sel),
               imputation_choice_val_A=dict(val_boundary_cm=val_imp, chosen=best_imp),
               tiers=tiers or "DEFAULT_TIERS", cells=cells, detection=det_rows, routes=route_use, confidence=conf_rows,
               confidence_flags=flag_rows, coverage_all_scenarios=cov_all, coverage_all_scenarios_mean_gauge=cov_all_mean,
               conformal_mean_gauge=dict(per_route_delta_cm=g_global_m.delta, adaptive_q=g_adapt_m.q), coverage_by_route=cov_route,
               conformal=dict(per_route_delta_cm=g_global.delta, marginal_delta_cm=g_global.global_delta_all, adaptive_q=g_adapt.q,
                              n_calibration=len(cal_err), calibration_error="boundary_hausdorff_cm (EFIT mode), adaptive pipeline outputs",
                              calibration_scenarios={s: cal_scen.count(s) for s in SCENARIOS}),
               latency=lat, wall_note="wall = PROXY (EFIT grid box, no vessel data): alarm rates are a protocol demonstration",
               runtime_s=time.time() - t0, fit_seconds=mit.fit_seconds)
    path = os.path.join(out, "mitigation.json")
    json.dump(_jf(res), open(path, "w"), indent=1)
    json.dump(_jf(table_json(res)), open(os.path.join(out, "mitigation_table.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(ctx.cache_dir, "m4_test_arrays.npz"), **{f"{s}__{k}": v for s, d in test_arrays.items() for k, v in d.items()})
    plot_mitigation(res, os.path.join(out, "module4_boundary_error.png"))
    open(os.path.join(out, "mitigation_table.md"), "w", encoding="utf8").write(table_md(res))
    log("wrote", path, f"({res['runtime_s']:.0f}s)")
    return res


def table_json(res):
    """HARNESS_V2.md section 4 schema (consumed by build_ablation_table.py --mitigation): adaptive / oracle rows over the table columns.
    boundary_mean_cm = EFIT-truth Module 1 v2 boundary error; latency_ms = p50 total of detector+router+mitigation+REAL network (CPU, batch 1)+gauge."""
    key = {"clean": "clean", "channel_loss_6": "channel_loss_6", "group_loss": "group_loss", "temporal_gap": "temporal_gap_40",
           "pre_disruption": "pre_disruption_25"}
    out = {}
    for row in ("adaptive", "oracle"):
        out[row] = {k: dict(boundary_mean_cm=res["cells"][s][row]["boundary_cm"], nrmse=res["cells"][s][row]["nrmse"]) for s, k in key.items()}
    tot = res["latency"]["summary"]["total"]
    out["adaptive"]["latency_ms"] = tot["p50"]
    out["_meta"] = dict(model=res["model"], device=res["device"], latency="p50 of total per-window cycle; p95=%.1f ms, p99=%.1f ms, mean=%.1f ms" % (tot["p95"], tot["p99"], tot["mean"]), note="single checkpoint/seed; windows = n_test EFIT-valid test windows of the shots on disk, "
                        "NOT the harness hash-selected phys_stride protocol; the failure-aware network is the backbone of both rows")
    return out


def _sub(ws, trs, n, seed):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.permutation(len(ws))[:n]) if n < len(ws) else np.arange(len(ws))
    return ws.subset(idx), [trs[i] for i in idx], idx


def latency_study(ctx, mit, gauge, ws, n, seed):
    """Per-window brain.process (detector, router, mitigation, REAL network forward at batch 1 on CPU, gauge) over a scenario mix."""
    rng = np.random.default_rng(seed + 7)
    brain = mit.brain(gauge)
    L = mit.L
    rows, net_only, per_scen = [], [], {}

    def model_fn(window, mask):
        x = ws.put([window])
        t0 = time.perf_counter()
        y = ctx.net.predict(x, np.asarray(mask, np.float32)[None])[0]
        net_only.append((time.perf_counter() - t0) * 1e3)
        return y
    scens = list(SCENARIOS)
    for k in range(min(n, len(ws))):
        scen = scens[k % len(scens)]
        win = ws.window(k)
        specs = make_faults(scen, L, rng, mit.usable)
        dwin, lab = inject_window(win, L, specs) if specs else (win, inject_window(win, L, [])[1])
        r = brain.process(dwin, model_fn)
        if k < 3:
            continue   # first calls warm the allocator
        rows.append(r["latencies_ms"])
        per_scen.setdefault(scen, []).append(r["latencies_ms"])
    return dict(summary=summarise_latencies(rows), per_scenario={s: summarise_latencies(v) for s, v in per_scen.items()},
                network_only_batch1=dict(mean=float(np.mean(net_only)), p50=float(np.percentile(net_only, 50)),
                                         p95=float(np.percentile(net_only, 95)), p99=float(np.percentile(net_only, 99)), n=len(net_only)),
                device=CPU_NOTE % (ctx.torch_version, ctx.torch_threads),
                note="'model' = real network forward on one window (batch 1) including numpy->torch conversion; the gauge stage uses the "
                     "35 % heuristic contour of physics_evaluation (cheap, NOT the EFIT-mode score used for the accuracy tables); "
                     "the physics fallback route replaces the network forward by a linear solve.")


def table_md(res):
    """Markdown results table: boundary cm [95 % CI] (NRMSE) per scenario x mitigation level, plus latency and conformal lines."""
    rows = [("none", "no mitigation (zero-fill)"), ("known_mask", "known-health mask only"), ("best_imp", "best fixed imputation (val-chosen)"),
            ("adaptive", "adaptive mitigation"), ("oracle", "Oracle mitigation")]
    sc = res["scenarios"]
    L = [f"Model {res['model']} | {res['sets']['test']['n']} EFIT-valid test windows from {res['sets']['test']['shots']} shots | {res['device']}", "",
         "| row | " + " | ".join(sc) + " |", "|---|" + "---|" * len(sc)]
    for v, lab in rows:
        L.append(f"| {lab} | " + " | ".join(f"{res['cells'][s][v]['boundary_cm']:.2f} [{res['cells'][s][v]['boundary_ci95'][0]:.1f}, "
                                            f"{res['cells'][s][v]['boundary_ci95'][1]:.1f}] ({res['cells'][s][v]['nrmse']:.3f})" for s in sc) + " |")
    lat = res["latency"]["summary"]
    L += ["", "Latency (ms, p50 / p95 / p99), per window: " + "; ".join(f"{k} {lat[k]['p50']:.1f}/{lat[k]['p95']:.1f}/{lat[k]['p99']:.1f}" for k in lat),
          "Coverage on test (target %.0f %%): Hausdorff gauge global/route/adaptive %s; mean-error gauge %s" % (
              100 * (1 - res["alpha"]), {k: round(v, 3) for k, v in res["coverage_all_scenarios"].items()},
              {k: round(v, 3) for k, v in res["coverage_all_scenarios_mean_gauge"].items()})]
    return chr(10).join(L) + chr(10)


def plot_mitigation(res, path):
    """Boundary error (cm, EFIT truth) per scenario for the four mitigation levels with 95 % shot-bootstrap CIs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("none", "no mitigation (zero-fill)"), ("best_imp", "best fixed imputation"), ("adaptive", "adaptive mitigation"), ("oracle", "Oracle mitigation")]
    sc = [x for x in res["scenarios"]]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    w = 0.2
    for k, (v, lab) in enumerate(rows):
        y = np.array([res["cells"][s][v]["boundary_cm"] for s in sc])
        ci = np.array([res["cells"][s][v]["boundary_ci95"] for s in sc])
        ax.bar(np.arange(len(sc)) + (k - 1.5) * w, y, w, yerr=[y - ci[:, 0], ci[:, 1] - y], label=lab, capsize=2)
    ax.set_xticks(np.arange(len(sc))); ax.set_xticklabels(sc, rotation=25, ha="right"); ax.set_ylabel("boundary error (cm)")
    ax.set_title(f"{res['model']}: mitigation levels, {res['sets']['test']['n']} test windows / {res['sets']['test']['shots']} shots")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y"); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def run_latency_only(ctx, out, n_lat=150, seed=0, n_train=2000, threads_sweep="1,2,4,8", **kw):
    """Re-measure the latency on a quiet machine: refit the mitigators exactly as run_mitigation did (same policy, from mitigation.json),
    time brain.process (real network, CPU, batch 1) for each torch thread count and write the result back into mitigation.json /
    mitigation_table.json (latency_by_threads keeps all of them; `latency` is the thread count with the lowest median total)."""
    import torch
    n_lat, seed, n_train = int(n_lat), int(seed), int(n_train)
    path = os.path.join(out, "mitigation.json")
    res = json.load(open(path))
    train = ctx.sets["train"].subset(np.linspace(0, len(ctx.sets["train"]) - 1, min(n_train, len(ctx.sets["train"]))).astype(int))
    (vaA, trA), _ = split_val_by_shot(ctx)
    (vaA1, trA1), _ = split_by_shot(vaA, trA)
    cal1, _, _ = _sub(vaA1, trA1, int(kw.get("n_val_a", 150)), seed)
    tiers = res["tiers"] if isinstance(res["tiers"], dict) else None
    mit = Mitigators(ctx, train, cal1, tiers=({int(k): v for k, v in tiers.items()} if tiers else None))
    pol = res["policy_selection_val_A2"]["chosen"]
    mit.set_policy(pol["thr_scale"], pol["short_ms"], pol["max_masked_branches"], pol["critical_policy"], pol.get("long_fault", "router"))
    gauge = ConformalGauge(alpha=ALPHA)
    gauge.delta = dict(res["conformal"]["per_route_delta_cm"])
    te, _, _ = pick_set(ctx, "test", int(kw.get("n_test", 300)), seed + 2)
    by_th = {}
    for th in [int(x) for x in str(threads_sweep).replace("-", ",").split(",")]:
        torch.set_num_threads(th)
        ctx.torch_threads = th
        by_th[th] = latency_study(ctx, mit, gauge, te, n_lat, seed)
        print(t(), "threads", th, {k: round(v["p50"], 2) for k, v in by_th[th]["summary"].items()}, flush=True)
    best = min(by_th, key=lambda k: by_th[k]["summary"]["total"]["p50"])
    res["latency"] = by_th[best]
    res["latency_by_threads"] = {str(k): dict(summary=v["summary"], network_only_batch1=v["network_only_batch1"]) for k, v in by_th.items()}
    res["latency"]["measured_on_quiet_machine"] = True
    json.dump(_jf(res), open(path, "w"), indent=1)
    json.dump(_jf(table_json(res)), open(os.path.join(out, "mitigation_table.json"), "w"), indent=1)
    return res


def main():
    print(__doc__)
    print("Run: python run_module34_local.py --model mlp_fa_v3 --data-root <dir with random/*.zarr> --work-dir <scratch> "
          "--stage all --pylib <numpy<2 + zarr2 dirs>")


if __name__ == "__main__":
    main()
