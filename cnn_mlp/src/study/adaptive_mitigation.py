#!/usr/bin/env python
"""Module 4: Smart Adaptive Mitigation Brain with Calibrated Confidence.

  Step 1  RealTimeFaultDetector    scans every probe of every sensor branch in an input window for dead / stuck
                                   probes, rail saturation, impulse spikes (> 6.5 sigma) and temporal gaps / NaNs.
  Step 2  AdaptiveRouter           transient glitch        -> temporal forward-fill
                                   redundant (Tier 3) probe -> spatial neighbour interpolation along the probe axis
                                   safety-critical (Tier 1/2) probe, a whole branch, or several branches
                                                           -> Module 2 failure-aware network with that branch masked
  Step 3  ConformalGauge           split-conformal boundary uncertainty +- delta (cm) per route, calibrated on the
                                   validation split; flags a wall breach when (clearance - delta) < 5 cm.

Everything here is plain NumPy on one input window. The reconstruction network is injected as `model_fn`, so the
brain can run on a trained model (mlp/physics_study.py mitigation) or on a stub in tests. Nothing in this file is
tuned to a target number: latencies are measured, delta comes from calibration residuals.

Window format: a list with one array per sensor branch, each (T, P) = time x probes (a 1-D branch has P = 1; any
leading singleton axes are fine). Inputs are standardised with the train statistics, so sigma = 1 and mean = 0.
"""
import math
import time

import numpy as np

from physics_evaluation import SAFETY_CLEARANCE_CM, compute_wall_proximity, extract_lcfs_contour

SPIKE_SIGMA = 6.5       # |x| above this (in standardised units) is an impulse
SATURATION_SIGMA = 4.5  # a flat probe stuck beyond this is a saturated amplifier, not a dead one
ROUTES = ("healthy", "transient_ffill", "redundant_interp", "module2")  # in increasing severity


def _as_tp(a):
    """(T, P) view of a branch array (a 1-D branch becomes P = 1), plus a function restoring the original shape."""
    a = np.asarray(a, dtype=np.float64)
    tp = a.squeeze()
    if tp.ndim == 1:
        tp = tp.reshape(-1, 1)
    if tp.ndim != 2:
        raise ValueError(f"expected one window per branch, got shape {a.shape}")
    return tp, (lambda x: x.reshape(a.shape))


class RealTimeFaultDetector:
    """Step 1. Per probe: STUCK_ZERO | RAIL_SATURATION | IMPULSE_SPIKE | TEMPORAL_GAP | HEALTHY.

    known_missing: optional list (one (P,) bool array per branch) of probes that are permanently absent from the
    dataset (always zero-filled, so the model was trained without them). They are not faults."""

    def __init__(self, known_missing=None):
        self.known_missing = known_missing

    def inspect(self, window):
        """Returns one dict per branch: dead (P,) bool, transient (T, P) bool samples, diagnoses {probe: label}."""
        report = []
        for i, branch in enumerate(window):
            tp, _ = _as_tp(branch)
            finite = np.isfinite(tp)
            clean = np.where(finite, tp, 0.0)
            dead = (~finite).all(axis=0) | (clean.var(axis=0) < 1e-9)           # flat for the whole window
            saturated = dead & (np.abs(clean.mean(axis=0)) > SATURATION_SIGMA)   # flat and far from the mean
            spikes = np.abs(clean) > SPIKE_SIGMA
            missing = (clean == 0.0) | ~finite
            pair = missing[:-1] & missing[1:]                                    # >= 2 consecutive missing samples
            gap = np.zeros_like(missing)
            gap[:-1] |= pair
            gap[1:] |= pair
            absent = np.asarray(self.known_missing[i], dtype=bool) if self.known_missing is not None else np.zeros_like(dead)
            dead = dead & ~absent
            transient = (spikes | gap | ~finite) & ~(dead | absent)[None, :]
            diagnoses = {int(p): "RAIL_SATURATION" if saturated[p] else "STUCK_ZERO" for p in np.flatnonzero(dead)}
            for p in np.flatnonzero(transient.any(axis=0)):
                diagnoses[int(p)] = "IMPULSE_SPIKE" if spikes[:, p].any() else "TEMPORAL_GAP"
            report.append(dict(dead=dead, transient=transient, diagnoses=diagnoses))
        return report


class AdaptiveRouter:
    """Step 2. `tiers[branch]` = 1 (safety-critical), 2 (shaping) or 3 (redundant); unknown branches count as 2."""

    def __init__(self, tiers=None):
        self.tiers = tiers or {}

    @staticmethod
    def _ffill(tp, bad):
        out = tp.copy()
        for p in np.flatnonzero(bad.any(axis=0)):
            good = np.flatnonzero(~bad[:, p])
            if good.size == 0:
                continue
            last = tp[good[0], p]  # a leading gap takes the first healthy sample
            for t in range(tp.shape[0]):
                if bad[t, p]:
                    out[t, p] = last
                else:
                    last = tp[t, p]
        return out

    @staticmethod
    def _neighbour_interp(tp, dead):
        """Replace each dead probe by the linear interpolation of its nearest healthy probes along the probe axis."""
        out = tp.copy()
        good = np.flatnonzero(~dead)
        for p in np.flatnonzero(dead):
            lo, hi = good[good < p], good[good > p]
            if lo.size and hi.size:
                a, b = lo.max(), hi.min()
                w = (p - a) / (b - a)
                out[:, p] = (1 - w) * tp[:, a] + w * tp[:, b]
            else:
                out[:, p] = tp[:, (lo.max() if lo.size else hi.min())]
        return out

    def route(self, window, report):
        """Returns (repaired window, health mask (n_branches,), route label, per-branch action log)."""
        n = len(window)
        mask = np.ones(n, dtype=np.float32)
        repaired, actions, level = [], {}, 0
        for i, (branch, rep) in enumerate(zip(window, report)):
            tp, restore = _as_tp(branch)
            tp = tp.copy()
            dead, tier = rep["dead"], self.tiers.get(i, 2)
            if rep["transient"].any():
                tp = self._ffill(tp, rep["transient"] & ~dead[None, :])
                actions.setdefault(i, []).append("forward-fill")
                level = max(level, 1)
            if dead.any():
                healthy_left = (~dead).any()
                if tier == 3 and healthy_left:
                    tp = self._neighbour_interp(tp, dead)
                    actions.setdefault(i, []).append(f"neighbour-interp x{int(dead.sum())}")
                    level = max(level, 2)
                else:  # Tier 1/2 probe or a whole branch lost: hand over to the failure-aware network
                    mask[i] = 0.0
                    tp[:] = 0.0
                    actions.setdefault(i, []).append("module2-mask")
                    level = 3
            repaired.append(restore(tp))
        return repaired, mask, ROUTES[level], actions


class ConformalGauge:
    """Step 3. Split-conformal radius delta (cm) per route, from boundary errors on a calibration set."""

    def __init__(self, alpha=0.05, wall_threshold_cm=SAFETY_CLEARANCE_CM):
        self.alpha, self.wall = alpha, wall_threshold_cm
        self.delta = {}

    def calibrate(self, errors_by_route, min_samples=20):
        """errors_by_route: {route: [boundary error in cm per calibration window]}. Routes with fewer than
        `min_samples` windows fall back to the largest delta of the routes that have enough."""
        for route, errs in errors_by_route.items():
            errs = np.sort(np.asarray(errs, dtype=np.float64))
            if errs.size >= min_samples:
                k = min(errs.size, math.ceil((errs.size + 1) * (1 - self.alpha)))
                self.delta[route] = float(errs[k - 1])
        fallback = max(self.delta.values()) if self.delta else float("inf")
        for route in errors_by_route:
            self.delta.setdefault(route, fallback)
        return self.delta

    def assess(self, psi_pred, route):
        delta = self.delta.get(route, max(self.delta.values()) if self.delta else float("inf"))
        clearance = compute_wall_proximity(extract_lcfs_contour(psi_pred))
        conservative = max(0.0, clearance - delta)
        return dict(clearance_cm=clearance, delta_cm=delta, conservative_clearance_cm=conservative,
                    alarm=bool(conservative < self.wall), point_alarm=bool(clearance < self.wall))


class SmartAdaptiveMitigationBrain:
    def __init__(self, tiers=None, gauge=None, known_missing=None):
        self.detector, self.router = RealTimeFaultDetector(known_missing), AdaptiveRouter(tiers)
        self.gauge = gauge or ConformalGauge()

    def process(self, window, model_fn):
        """window -> dict(route, mask, psi, assessment, actions, latencies_ms). model_fn(window, mask) -> (65, 65)."""
        t0 = time.perf_counter()
        report = self.detector.inspect(window)
        t1 = time.perf_counter()
        repaired, mask, route, actions = self.router.route(window, report)
        t2 = time.perf_counter()
        psi = model_fn(repaired, mask)
        t3 = time.perf_counter()
        assessment = self.gauge.assess(psi, route)
        t4 = time.perf_counter()
        return dict(route=route, mask=mask, psi=psi, assessment=assessment, actions=actions,
                    latencies_ms=dict(detector=(t1 - t0) * 1e3, router=(t2 - t1) * 1e3, model=(t3 - t2) * 1e3,
                                      gauge=(t4 - t3) * 1e3, total=(t4 - t0) * 1e3))


if __name__ == "__main__":
    # self-check on a synthetic window: a spike, a dead redundant probe, and a dead critical branch
    rng = np.random.default_rng(0)
    win = [rng.normal(size=(25, 15)), rng.normal(size=(25, 8)), rng.normal(size=(20,))]
    win[0][10, 3] = 30.0       # impulse on branch 0 -> forward-fill
    win[1][:, 4] = 0.0         # dead probe on branch 1 (tier 3 below) -> neighbour interpolation
    det = RealTimeFaultDetector().inspect(win)
    assert det[0]["diagnoses"] == {3: "IMPULSE_SPIKE"} and det[1]["diagnoses"] == {4: "STUCK_ZERO"}
    fixed, mask, route, _ = AdaptiveRouter({0: 2, 1: 3, 2: 1}).route(win, det)
    assert route == "redundant_interp" and mask.all() and abs(fixed[0][10, 3]) < 30
    assert np.allclose(fixed[1][:, 4], 0.5 * (win[1][:, 3] + win[1][:, 5]))
    win[2][:] = 0.0            # tier-1 branch dead -> module 2 with that branch masked
    fixed, mask, route, _ = AdaptiveRouter({0: 2, 1: 3, 2: 1}).route(win, RealTimeFaultDetector().inspect(win))
    assert route == "module2" and mask.tolist() == [1, 1, 0]
    known = [np.zeros(15, bool), np.zeros(8, bool), np.zeros(1, bool)]
    known[1][4] = True  # a probe that is always absent is not a fault
    assert RealTimeFaultDetector(known).inspect(win)[1]["diagnoses"] == {}
    g = ConformalGauge(alpha=0.1)
    g.calibrate({"healthy": list(np.linspace(0.1, 1.0, 99))})
    assert abs(g.delta["healthy"] - 0.91) < 0.02  # 90% split-conformal quantile of 99 evenly spaced errors
    print("adaptive_mitigation self-check ok")
