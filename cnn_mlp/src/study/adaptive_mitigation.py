#!/usr/bin/env python
"""Module 4 (v2): Adaptive failure mitigation with calibrated confidence.  See MODULE4_V2.md.

  Step 1  detect     FaultDetector (v2): local rules (NaN / zero gaps, flat-line runs and flat variance, spikes) +
                     cross-sensor residuals (every channel predicted from all the others by a ridge/Gaussian
                     conditional fitted on clean windows; stuck / drift / offset show up as standardised residuals)
                     + flux-loop residual against the model's psi (FluxLoopCheck). Faults are isolated one at a time
                     (largest standardised residual first, then removed from the predictor set and re-scored), so a
                     fault does not smear onto its neighbours. Legacy per-branch RealTimeFaultDetector is kept.
  Step 2  route      RouterV2: by fault duration and sensor criticality tier
                         short transient              -> forward-fill
                         long fault, redundant sensor -> virtual sensor (ridge from healthy neighbours)
                         critical sensor / multi-sensor outage / whole branch -> failure-aware model with the mask
                         beyond what that model was trained for (or no model) -> PHYSICS FALLBACK (linear flux
                                                         reconstruction from the surviving measurements, not EFIT)
                     Oracle mode takes the true fault labels instead of the detector (upper bound of detector+router).
  Step 3  gauge      ConformalGauge (split conformal, per route, kept for physics_study.py) and AdaptiveConformalGauge
                     (difficulty-normalised, locally adaptive), plus a HIGH / LOW / ABSTAIN flag for the controller.

Window format: list with one array per sensor branch, each (T_b, P_b); standardised with the train statistics. The
legacy classes (RealTimeFaultDetector, AdaptiveRouter, ConformalGauge, SmartAdaptiveMitigationBrain(tiers, gauge,
known_missing).process(window, model_fn)) behave exactly as before when no v2 component is passed.
Nothing here is tuned to a target number; thresholds come from clean calibration windows, latencies are measured.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np

try:
    from physics_evaluation import SAFETY_CLEARANCE_CM, compute_wall_proximity, extract_lcfs_contour
except Exception:  # synthetic tests of detection / routing do not need the physics layer
    SAFETY_CLEARANCE_CM, compute_wall_proximity, extract_lcfs_contour = 5.0, None, None
from fault_models import WINDOW_MS, FaultLabels, FaultSpec, WindowLayout, as_tp
from virtual_sensor import LinearFluxFallback, VirtualSensorBank

SPIKE_SIGMA = 6.5       # |x| above this (in standardised units) is an impulse
SATURATION_SIGMA = 4.5  # a flat probe stuck beyond this is a saturated amplifier, not a dead one
ROUTES = ("healthy", "transient_ffill", "redundant_interp", "module2")  # legacy router, in increasing severity
ROUTES_V2 = ("healthy", "transient_ffill", "virtual_sensor", "module2", "physics_fallback")
# Tier per branch [flux_loop, ccbv, obr, obv, saddle_v, coil_i, solenoid_i, ip] from module 3's last published
# result (AGENTS.md); pass `tiers=` (from criticality.json) to override once module 3 is rebuilt.
DEFAULT_TIERS = {0: 3, 1: 1, 2: 2, 3: 2, 4: 3, 5: 2, 6: 1, 7: 2}


def _as_tp(a):
    tp, restore = as_tp(a)
    return tp, restore


# ====================================================================== legacy (unchanged behaviour)
class RealTimeFaultDetector:
    """Legacy step 1. Per probe: STUCK_ZERO | RAIL_SATURATION | IMPULSE_SPIKE | TEMPORAL_GAP | HEALTHY.

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
    """Legacy step 2. `tiers[branch]` = 1 (safety-critical), 2 (shaping) or 3 (redundant); unknown branches count as 2."""

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


# ====================================================================== v2 detection
@dataclass
class Detection:
    """Per-channel detector output on one window (canonical channel matrix, T samples)."""
    flagged: np.ndarray            # (C,) bool
    score: np.ndarray              # (C,) float, higher = more suspicious; flagged channels rank above all others
    kind: np.ndarray               # (C,) object: dropout | flatline | rail | spike | drift | offset | flux_residual | ''
    onset_ms: np.ndarray           # (C,) first altered sample (nan if healthy)
    duration_ms: np.ndarray        # (C,) altered span (nan if healthy)
    sample_mask: np.ndarray        # (T, C) bool: samples to repair
    order: list = field(default_factory=list)   # flagged channels in isolation order (most certain first)
    resid_score: float = 0.0       # largest un-flagged-normalised residual ratio (difficulty proxy for the gauge)
    absent: np.ndarray = None      # (C,) known permanently absent channels (never faults)


def detection_from_labels(labels: FaultLabels, window_ms=WINDOW_MS):
    """Oracle: build a Detection from the ground-truth labels of the injector (no detector involved)."""
    C = len(labels.kind)
    fl = labels.faulty
    score = np.where(fl, 1e3, 0.0)
    return Detection(fl, score, labels.kind.copy(), labels.onset_ms.copy(), labels.duration_ms.copy(),
                     labels.mask.copy(), list(np.flatnonzero(fl)), 0.0, np.zeros(C, bool))


def _longest_run(B):
    """B (T, C) bool -> (length (C,), end index (C,)) of the longest run of True per column."""
    T, C = B.shape
    cur = np.zeros(C, int)
    best = np.zeros(C, int)
    end = np.zeros(C, int)
    for t in range(T):
        cur = np.where(B[t], cur + 1, 0)
        up = cur > best
        best = np.where(up, cur, best)
        end = np.where(up, t, end)
    return best, end


def _pot_threshold(v, p, q0=0.95):
    """Threshold exceeded with probability ~p: peaks-over-threshold with an exponential tail above the q0 quantile."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.inf
    u = np.quantile(v, q0)
    ex = v[v > u] - u
    if ex.size < 10:
        return float(np.quantile(v, 1 - min(p, 1.0))) if v.size * p >= 1 else float(v.max() * 1.2)
    return float(u + ex.mean() * math.log((1 - q0) / p))


def _stats(R):
    """R (..., T, C) standardised residuals -> peak |r|, rms, |total change across the window| (each (..., C))."""
    T = R.shape[-2]
    t = np.arange(T) - (T - 1) / 2.0
    slope = np.tensordot(R, t, axes=([-2], [0])) / max(np.sum(t * t), 1e-12)
    return np.abs(R).max(axis=-2), np.sqrt((R ** 2).mean(axis=-2)), np.abs(slope) * (T - 1)


def _remove(P, c):
    """Marginalise variable c out of a precision matrix (rank-1 Schur update); row/col c become 0."""
    if P[c, c] > 1e-12:
        P = P - np.outer(P[:, c], P[c, :]) / P[c, c]
    P[c, :] = 0.0
    P[:, c] = 0.0
    return P


class FluxLoopCheck:
    """Flux-loop residual: measured loop flux vs the reconstruction's psi at the loop pixels.

    The two live in different units / offsets (loop flux vs standardised psi), so each loop gets a fitted linear
    map psi_pixel -> flux and a robust residual sigma from clean windows; a loop is faulty when its standardised
    residual exceeds a pooled peaks-over-threshold quantile of clean residuals. Its value is a second opinion that
    does not depend on the cross-sensor model. It sees what the reconstruction contradicts, so it is weaker when
    the same faulty loop fed the reconstruction (the fix is to run it on a model that had the fault masked).

    loop_pixels: list of (row, col) indices into psi. For pipeline.FLUX_LOOP_COORDS (r_idx, z_idx) use
    `loop_pixels_from_coords`. loop_channels: the matching channel indices in the canonical matrix."""

    def __init__(self, loop_pixels, loop_channels, fp_rate=1e-3):
        self.pix = np.asarray(loop_pixels, int)
        self.channels = np.asarray(loop_channels, int)
        self.fp_rate = fp_rate
        self.a = self.b = self.s = None
        self.thr = np.inf

    def psi_at_loops(self, psi):
        return np.asarray(psi)[self.pix[:, 0], self.pix[:, 1]]

    def fit(self, flux_means, psi_maps, calib=None):
        """flux_means (N, L) measured loop flux per window; psi_maps (N, H, W) matching clean reconstructions."""
        Y = np.asarray(flux_means, float)
        Xp = np.stack([self.psi_at_loops(p) for p in psi_maps])
        self.a, self.b, self.s = np.zeros(Y.shape[1]), np.zeros(Y.shape[1]), np.ones(Y.shape[1])
        for l in range(Y.shape[1]):
            ok = np.isfinite(Y[:, l])
            A = np.stack([Xp[ok, l], np.ones(ok.sum())], 1)
            self.a[l], self.b[l] = np.linalg.lstsq(A, Y[ok, l], rcond=None)[0]
            res = Y[ok, l] - A @ np.array([self.a[l], self.b[l]])
            self.s[l] = max(1.4826 * np.median(np.abs(res - np.median(res))), 1e-9)
        cf, cp = (Y, Xp) if calib is None else (np.asarray(calib[0], float), np.stack([self.psi_at_loops(p) for p in calib[1]]))
        z = np.abs((cf - (self.a * cp + self.b)) / self.s)
        self.thr = _pot_threshold(z, self.fp_rate)
        return self

    def residuals(self, flux_mean, psi):
        return (np.asarray(flux_mean, float) - (self.a * self.psi_at_loops(psi) + self.b)) / self.s


def loop_pixels_from_coords(coords):
    """pipeline.FLUX_LOOP_COORDS is (r_idx, z_idx) and psi is indexed [z, r]; returns (row, col) = (z_idx, r_idx)."""
    return [(int(z), int(r)) for r, z in coords]


class FaultDetector:
    """v2 detector on the canonical channel matrix X (T, C) in standardised units (see module doc).

    fit(train, calib) -- train (N, T, C) clean windows fit the statistics and the cross-sensor model; calib (N2, T, C)
    separate clean windows set the thresholds (so the false-positive rate is measured out of sample); if omitted the
    training windows are reused and the thresholds are optimistic.
        fp_rate   target false positive probability per channel per residual statistic
        thr_scale multiplies every residual threshold (operating point)
        demean    cross-check only intra-window dynamics (robust to shot-to-shot level shifts; blind to pure offsets)"""

    def __init__(self, spike_sigma=SPIKE_SIGMA, rail_sigma=SATURATION_SIGMA, fp_rate=1e-4, ridge="auto", thr_scale=1.0,
                 window_ms=WINDOW_MS, max_faults=None, absent=None, flux_check=None, demean=False):
        self.spike_sigma, self.rail_sigma, self.fp_rate, self.ridge = spike_sigma, rail_sigma, fp_rate, ridge
        self.thr_scale, self.window_ms, self.max_faults = thr_scale, window_ms, max_faults
        self.absent_given, self.flux_check, self.demean = absent, flux_check, demean
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, train, calib=None):
        tr = np.asarray(train, float)
        N, T, C = tr.shape
        rows = tr.reshape(-1, C)
        self.mu = np.nanmean(rows, axis=0)
        sd = np.nanstd(rows, axis=0)
        zero_frac = np.nanmean(rows == 0.0, axis=0)
        self.absent = (zero_frac > 0.5) | ~(sd > 1e-12)               # always zero / constant: not a fault
        if self.absent_given is not None:
            self.absent |= np.asarray(self.absent_given, bool)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        Z = self._zr(tr)
        S = Z.reshape(-1, C).T @ Z.reshape(-1, C) / (N * T)
        cal = tr if calib is None else np.asarray(calib, float)
        Zc = self._zr(cal)
        # ridge: pick (out of sample, on the calibration windows) the value that predicts channels best
        grid = [self.ridge] if self.ridge != "auto" else [1e-3, 1e-2, 1e-1, 3e-1, 1.0]
        best = None
        for lam in grid:
            self.Sreg = S + lam * np.eye(C)
            P = self._precision(self.absent)
            e2 = np.mean(((Zc @ P) / np.maximum(np.diag(P), 1e-12)) ** 2, axis=(0, 1))   # raw residual variance per channel
            score = float(np.mean(np.log(e2[~self.absent] + 1e-12)))
            if best is None or score < best[0]:
                best = (score, lam, P)
        _, self.ridge_used, self.P_base = best
        self.Sreg = S + self.ridge_used * np.eye(C)
        diag = np.sqrt(np.maximum(np.diag(self.P_base), 1e-12))
        R = (Zc @ self.P_base) / diag
        std = R.reshape(-1, C).std(axis=0)                            # out-of-sample scale (in-sample would be optimistic)
        self.kappa = np.where(std > 1e-12, 1.0 / np.where(std > 1e-12, std, 1.0), 1.0)
        # local-rule statistics from the clean windows
        allw = np.concatenate([tr, cal], axis=0)
        eq = np.isfinite(allw[:, 1:]) & (allw[:, 1:] == allw[:, :-1])
        runs = np.array([_longest_run(e)[0] for e in eq])             # (N, C) longest equal-step run
        self.min_run = np.maximum(runs.max(axis=0) + 3, 5)            # >= 5 equal-step steps = 6 identical samples
        var = np.nanvar(self._z(allw), axis=1)
        self.flat_var = 0.5 * np.nanmin(var, axis=0)           # below every clean window's variance (0 disables it)
        # per-channel spike level: never below the clean extremes seen in training / calibration
        self.spike_thr = np.maximum(self.spike_sigma, 1.25 * np.abs(self._z(allw)).max(axis=(0, 1)))
        # residual thresholds from clean calibration windows
        Rc = (Zc @ self.P_base) / diag * self.kappa
        peak, rms, slope = _stats(Rc)
        keep = ~self.absent
        self.thr = np.array([_pot_threshold(x[:, keep].ravel(), self.fp_rate) for x in (peak, rms, slope)]) * self.thr_scale
        self.fitted = True
        return self

    def _z(self, X):
        Z = (X - self.mu) / self.sd
        return np.where(np.isfinite(Z), Z, 0.0)

    def _zr(self, X):
        """z-scores as seen by the cross-sensor model: optionally with each window's own mean removed, which makes the
        residuals insensitive to shot-to-shot level shifts (only intra-window dynamics are cross-checked)."""
        Z = self._z(X)
        return Z - Z.mean(axis=-2, keepdims=True) if self.demean else Z

    def _precision(self, removed):
        keep = np.flatnonzero(~removed)
        P = np.zeros_like(self.Sreg)
        if keep.size:
            P[np.ix_(keep, keep)] = np.linalg.inv(self.Sreg[np.ix_(keep, keep)])
        return P

    # ------------------------------------------------------------------ detect
    def detect(self, X, psi_pred=None):
        X = np.asarray(X, float)
        T, C = X.shape
        ms = self.window_ms / T
        finite = np.isfinite(X)
        Z = (X - self.mu) / self.sd
        Zc = np.where(finite, Z, 0.0)
        Zr = Zc - Zc.mean(axis=0, keepdims=True) if self.demean else Zc
        absent = self.absent
        kind = np.full(C, "", dtype=object)
        smask = np.zeros((T, C), bool)
        strength = np.zeros(C)

        # -- local rules ---------------------------------------------------
        isz = (X == 0.0) | ~finite
        pair = isz[:-1] & isz[1:]
        zrun = np.zeros((T, C), bool)
        zrun[:-1] |= pair
        zrun[1:] |= pair
        zrun |= ~finite
        drop = zrun.any(axis=0) & ~absent
        eq = finite[1:] & finite[:-1] & (X[1:] == X[:-1])
        rl, re = _longest_run(eq)
        flat_run = (rl >= self.min_run) & ~absent & ~drop
        flat_var = (np.var(Zc, axis=0) < self.flat_var) & finite.all(axis=0) & ~absent & ~drop & ~flat_run
        spk = (np.abs(Zc) > self.spike_thr) & ~absent[None, :]
        spike = spk.any(axis=0) & ~drop & ~flat_run & ~flat_var
        for c in np.flatnonzero(drop):
            kind[c], smask[:, c], strength[c] = "dropout", zrun[:, c], 30.0 + zrun[:, c].sum()
        for c in np.flatnonzero(flat_run):
            s0, e0 = re[c] - rl[c] + 1, re[c] + 1                     # equal-step run covers samples s0..e0
            m = np.zeros(T, bool)
            m[s0 + 1:e0 + 1] = True
            kind[c] = "rail" if abs(Zc[e0, c]) > self.rail_sigma else "flatline"
            smask[:, c], strength[c] = m, 20.0 + rl[c]
        for c in np.flatnonzero(flat_var):
            kind[c], smask[:, c], strength[c] = "flatline", True, 15.0
        for c in np.flatnonzero(spike):
            kind[c], smask[:, c], strength[c] = "spike", spk[:, c], 10.0 + np.abs(Zc[:, c]).max() / self.spike_thr[c]
        local = kind != ""
        order = [int(c) for c in np.argsort(-strength) if local[c]]

        # -- cross-sensor residuals with isolation by peeling -----------------
        removed = absent | local
        P = self.P_base.copy()
        loc_idx = np.flatnonzero(local)
        if loc_idx.size > 8:
            P = self._precision(removed)
        else:
            for c in loc_idx:
                P = _remove(P, c)
        thr = self.thr
        cap = self.max_faults if self.max_faults is not None else max(3, C // 4)
        ratio = np.zeros(C)
        picks = 0
        res_info = {}
        while True:
            d = np.sqrt(np.maximum(np.diag(P), 1e-12))
            R = (Zr @ P) / d * self.kappa
            R[:, removed] = 0.0
            pk, rm, sl = _stats(R)
            ratio = np.max([pk / thr[0], rm / thr[1], sl / thr[2]], axis=0)
            ratio[removed] = -np.inf
            c = int(np.argmax(ratio))
            if not removed.all() and ratio[c] > 1.0 and picks < cap:
                dom = np.argmax([pk[c] / thr[0], rm[c] / thr[1], sl[c] / thr[2]])
                if dom == 0 and rm[c] / thr[1] < 1.0:
                    kind[c] = "spike"
                    smask[:, c] = np.abs(R[:, c]) > thr[0]
                    smask[np.argmax(np.abs(R[:, c])), c] = True
                else:
                    kind[c] = "drift" if dom == 2 else "offset"
                    smask[:, c] = True
                strength[c] = ratio[c]
                order.append(c)
                removed[c] = True
                P = _remove(P, c)
                picks += 1
            else:
                break
        resid_score = float(np.nanmax(np.where(np.isfinite(ratio), ratio, 0.0))) if ratio.size else 0.0

        # -- flux-loop residual (second opinion against the reconstruction) ----
        fc = self.flux_check
        if fc is not None and psi_pred is not None and fc.a is not None:
            cols = fc.channels
            fm = np.nanmean(np.where(finite[:, cols], X[:, cols], np.nan), axis=0)
            z = np.abs(fc.residuals(fm, psi_pred))
            for l, c in enumerate(cols):
                if kind[c] == "" and not absent[c] and np.isfinite(z[l]) and z[l] > fc.thr:
                    kind[c], smask[:, c], strength[c] = "flux_residual", True, 1.0 + z[l] / fc.thr
                    order.append(int(c))

        flagged = kind != ""
        score = np.where(np.isfinite(ratio), np.minimum(ratio, 0.999), 0.0)
        for pos, c in enumerate(dict.fromkeys(order)):
            score[c] = 1e3 - pos
        onset = np.where(flagged, np.argmax(smask, axis=0) * ms, np.nan)
        dur = np.where(flagged, smask.sum(axis=0) * ms, np.nan)
        return Detection(flagged, score, kind, onset, dur, smask, list(dict.fromkeys(order)), resid_score, absent.copy())

    def detect_window(self, window, layout, psi_pred=None):
        return self.detect(layout.flatten(window), psi_pred)


# ====================================================================== v2 router
@dataclass
class Decision:
    route: str
    branch_mask: np.ndarray        # (n_branches,) float32 for the failure-aware model
    channel_actions: dict          # channel -> 'forward-fill' | 'virtual' | 'masked'
    transient: np.ndarray          # (C,) bool, repaired by forward-fill
    virtual: np.ndarray            # (C,) bool, repaired by the virtual sensor
    lost: np.ndarray               # (C,) bool, unavailable to the reconstruction (masked branch or unrepairable)
    use_fallback: bool = False
    log: dict = field(default_factory=dict)


class RouterV2(AdaptiveRouter):
    """Step 2 (v2). Defaults are policy placeholders (not tuned): set them from module 3 / validation results.

        short_ms         faults of dropout / spike type no longer than this are 'short transients' (forward-fill)
        multi_channels   this many simultaneously lost (non-transient) probes is a multi-sensor outage: the virtual
                         sensors' neighbours are failing too, so the masking threshold drops to multi_branch_frac
        branch_frac      a branch is masked for the failure-aware model when this fraction of its probes is lost
        virtual_r2_min   virtual sensor only if its R^2 given the surviving neighbours reaches this
        max_masked_branches  the failure-aware model was trained with at most this many branches masked
                         (structured augmentation: magnetics_all = 5); beyond it, or with no model, use the fallback
        critical_policy  'mask' (tier-1 loss -> failure-aware model, as specified) or 'virtual' (try virtual first)"""

    def __init__(self, tiers=None, layout=None, bank: VirtualSensorBank = None, fallback: LinearFluxFallback = None,
                 short_ms=1.0, multi_channels=6, branch_frac=0.5, multi_branch_frac=0.25, virtual_r2_min=0.9,
                 max_masked_branches=5, critical_policy="mask"):
        super().__init__(DEFAULT_TIERS if tiers is None else tiers)
        self.layout, self.bank, self.fallback = layout, bank, fallback
        self.short_ms, self.multi_channels, self.branch_frac = short_ms, multi_channels, branch_frac
        self.multi_branch_frac = multi_branch_frac
        self.virtual_r2_min, self.max_masked_branches, self.critical_policy = virtual_r2_min, max_masked_branches, critical_policy

    def decide(self, det: Detection, layout=None, have_model=True):
        L = layout or self.layout
        C = L.n_channels
        absent = det.absent if det.absent is not None else np.zeros(L.n_channels, bool)
        bad = det.flagged & ~absent
        kind = det.kind
        transient = bad & ((kind == "spike") | ((kind == "dropout") & (det.duration_ms <= self.short_ms + 1e-9)))
        long = bad & ~transient
        branch_mask = np.ones(L.n_branches, np.float32)
        if not bad.any():
            return Decision("healthy", branch_mask, {}, transient, long & False, long & False)
        tier = np.array([self.tiers.get(int(b), 2) for b in L.branch_of])
        actions = {int(c): "forward-fill" for c in np.flatnonzero(transient)}
        virtual = np.zeros(C, bool)
        lost = np.zeros(C, bool)
        route = "transient_ffill"
        if long.any():
            n_long = int(long.sum())
            whole_branch = any(long[L.channels_of(b)].mean() >= self.branch_frac for b in range(L.n_branches))
            critical = bool((long & (tier == 1)).any()) and self.critical_policy == "mask"
            eligible = self.bank is not None and not (critical or whole_branch or n_long >= self.multi_channels)
            if eligible:
                r2 = np.array([self.bank.r2_given(int(c), bad | absent) for c in np.flatnonzero(long)])
                eligible = bool((r2 >= self.virtual_r2_min).all())
            if eligible:
                virtual = long.copy()
                actions.update({int(c): "virtual" for c in np.flatnonzero(long)})
                route = "virtual_sensor"
            else:
                route = "module2"
                branch_mask = L.branch_mask(long, self.multi_branch_frac if n_long >= self.multi_channels else self.branch_frac)
                for b in range(L.n_branches):                           # a lost tier-1 probe masks its branch
                    if tier[L.channels_of(b)[0]] == 1 and long[L.channels_of(b)].any():
                        branch_mask[b] = 0.0
                masked_ch = np.isin(L.branch_of, np.flatnonzero(branch_mask == 0))
                rest = long & ~masked_ch                                  # lost probes of branches we keep
                if rest.any() and self.bank is not None:
                    ok = np.array([self.bank.r2_given(int(c), bad | absent | masked_ch) >= self.virtual_r2_min
                                   for c in np.flatnonzero(rest)])
                    unfixable = np.flatnonzero(rest)[~ok]
                else:
                    unfixable = np.flatnonzero(rest)
                for c in unfixable:                                       # cannot be repaired: tell the model, mask the branch
                    branch_mask[L.branch_of[c]] = 0.0
                masked_ch = np.isin(L.branch_of, np.flatnonzero(branch_mask == 0))
                lost = masked_ch.copy()
                virtual = long & ~masked_ch
                actions.update({int(c): "virtual" for c in np.flatnonzero(virtual)})
                actions.update({int(c): "masked" for c in np.flatnonzero(masked_ch & bad)})
                if branch_mask.all():
                    route = "virtual_sensor"
                n_masked = int((branch_mask == 0).sum())
                if self.fallback is not None and (not have_model or n_masked > self.max_masked_branches):
                    route = "physics_fallback"
        use_fb = route == "physics_fallback"
        return Decision(route, branch_mask, actions, transient, virtual, lost, use_fb)

    def apply(self, window, det: Detection, dec: Decision, layout=None):
        """Mitigation: returns the repaired window (native shapes)."""
        L = layout or self.layout
        if dec.route == "healthy":
            return window
        X = L.flatten(window)
        fixed = X.copy()
        touched = np.zeros_like(det.sample_mask)
        for c in np.flatnonzero(dec.transient):
            m = det.sample_mask[:, c]
            good = np.flatnonzero(~m & np.isfinite(X[:, c]))
            if good.size == 0:
                continue
            last = X[good[0], c]
            for t in range(X.shape[0]):
                if m[t]:
                    fixed[t, c] = last
                else:
                    last = X[t, c]
            touched[:, c] = m
        if dec.virtual.any():
            fixed = self.bank.repair(fixed, dec.virtual, sample_mask=det.sample_mask, failed=det.flagged | det.absent)
            touched |= det.sample_mask & dec.virtual[None, :]
        out = L.write_back(window, fixed, touched)
        for b in np.flatnonzero(dec.branch_mask == 0):                      # masked branch: zeros, like the model's training
            out[b] = np.zeros_like(out[b])
        return out

    # legacy-compatible entry point: route_and_mitigate(x_window, mask, diagnoses)
    def route_and_mitigate(self, x_window, mask=None, diagnoses=None, oracle_faults=None, layout=None, have_model=True):
        """x_window: branch-list window. diagnoses: a Detection (from FaultDetector.detect_window) -- ignored in oracle
        mode. oracle_faults: FaultLabels (or list of FaultSpec + `layout`) from the injector, which skips detection and
        routes on the true fault labels (the Oracle row = upper bound of detector + router). `mask` is accepted for
        signature compatibility and unused: the health mask is derived from the detection.
        Returns (repaired window, health mask (n_branches,), route label, per-channel action log)."""
        L = layout or self.layout
        det = diagnoses
        if oracle_faults is not None:
            if not isinstance(oracle_faults, FaultLabels):
                raise TypeError("oracle_faults must be the FaultLabels returned by fault_models.inject / inject_window")
            det = detection_from_labels(oracle_faults)
            if det.absent is None or det.absent.size != L.n_channels:
                det.absent = np.zeros(L.n_channels, bool)
        if not isinstance(det, Detection):
            raise TypeError("pass a Detection as `diagnoses` (FaultDetector.detect_window) or give oracle_faults")
        dec = self.decide(det, L, have_model)
        return self.apply(x_window, det, dec, L), dec.branch_mask, dec.route, dec.channel_actions


# ====================================================================== step 3: confidence
def conformal_quantile(scores, alpha):
    s = np.sort(np.asarray(scores, float))
    k = min(s.size, math.ceil((s.size + 1) * (1 - alpha)))
    return float(s[k - 1])


def confidence_flag(clearance_cm, delta_cm, wall_cm=SAFETY_CLEARANCE_CM, high_frac=0.5, abstain_frac=1.0, valid=True):
    """Controller-facing flag, thresholds derived from the wall margin (default 5 cm) and the conformal radius delta.
        ABSTAIN  delta > abstain_frac * wall (interval as wide as the whole margin) or the output is not valid
        HIGH     the alarm decision is unambiguous (|clearance - wall| > delta) and delta <= high_frac * wall
        LOW      otherwise (decision within the uncertainty band, or a wide interval)
    high_frac / abstain_frac are policy parameters, not tuned."""
    if not valid or not np.isfinite(delta_cm) or delta_cm > abstain_frac * wall_cm:
        return "ABSTAIN"
    return "HIGH" if (abs(clearance_cm - wall_cm) > delta_cm and delta_cm <= high_frac * wall_cm) else "LOW"


class ConformalGauge:
    """Step 3. Split-conformal radius delta (cm) per route (Mondrian), from boundary errors on a calibration set.

    Coverage logic: delta = the ceil((n+1)(1-alpha))-th smallest calibration error, which gives
    P(error <= delta) >= 1 - alpha for a new window exchangeable with the n calibration windows of that route.
    Caveats (see MODULE4_V2.md): physics_study replays the same validation windows under 8 damage scenarios and
    neighbouring windows of one shot are correlated, so the calibration set is not i.i.d.; a single delta per route
    is marginal coverage only (too wide for easy windows, too tight for hard ones) -> AdaptiveConformalGauge."""

    def __init__(self, alpha=0.05, wall_threshold_cm=SAFETY_CLEARANCE_CM, high_frac=0.5, abstain_frac=1.0):
        self.alpha, self.wall = alpha, wall_threshold_cm
        self.high_frac, self.abstain_frac = high_frac, abstain_frac
        self.delta = {}

    def calibrate(self, errors_by_route, min_samples=20):
        """errors_by_route: {route: [boundary error in cm per calibration window]}. Routes with fewer than
        `min_samples` windows fall back to the largest delta of the routes that have enough."""
        for route, errs in errors_by_route.items():
            errs = np.asarray(errs, dtype=np.float64)
            if errs.size >= min_samples:
                self.delta[route] = conformal_quantile(errs, self.alpha)
        fallback = max(self.delta.values()) if self.delta else float("inf")
        for route in errors_by_route:
            self.delta.setdefault(route, fallback)
        return self.delta

    def delta_for(self, route, difficulty=None):
        return self.delta.get(route, max(self.delta.values()) if self.delta else float("inf"))

    def assess_clearance(self, clearance, route, difficulty=None, valid=True):
        delta = self.delta_for(route, difficulty)
        conservative = max(0.0, clearance - delta)
        return dict(clearance_cm=clearance, delta_cm=delta, conservative_clearance_cm=conservative,
                    alarm=bool(conservative < self.wall), point_alarm=bool(clearance < self.wall),
                    confidence=confidence_flag(clearance, delta, self.wall, self.high_frac, self.abstain_frac, valid))

    def assess(self, psi_pred, route, difficulty=None):
        if compute_wall_proximity is None:
            raise ImportError("physics_evaluation is required to compute wall clearance from psi")
        return self.assess_clearance(compute_wall_proximity(extract_lcfs_contour(psi_pred)), route, difficulty)


class AdaptiveConformalGauge(ConformalGauge):
    """Difficulty-normalised (locally adaptive) split conformal.

    A linear difficulty model sigma_hat(f) = max(floor, [1, f_std] . beta) is fitted on one half of the calibration
    errors; nonconformity score = error / sigma_hat on the other half; delta(f) = q * sigma_hat(f), q = the conformal
    quantile of the scores. Marginal coverage stays 1 - alpha but delta shrinks on easy windows and grows on hard ones.
    f = difficulty features, e.g. `difficulty_features(detection, decision)` (largest residual ratio, fraction of lost
    channels, masked branches, fallback)."""

    def calibrate_adaptive(self, errors, feats, fit_frac=0.5, seed=0, floor_frac=0.1, ridge=1e-3):
        e = np.asarray(errors, float)
        F = np.asarray(feats, float).reshape(len(e), -1)
        idx = np.random.default_rng(seed).permutation(len(e))
        nfit = int(round(fit_frac * len(e)))
        a, b = idx[:nfit], idx[nfit:]
        self.f_mu, self.f_sd = F[a].mean(0), np.where(F[a].std(0) > 1e-9, F[a].std(0), 1.0)
        D = np.hstack([np.ones((len(a), 1)), (F[a] - self.f_mu) / self.f_sd])
        self.beta = np.linalg.solve(D.T @ D + ridge * np.eye(D.shape[1]), D.T @ e[a])
        self.floor = floor_frac * float(e[a].mean())
        self.q = conformal_quantile(e[b] / self._sigma(F[b]), self.alpha)
        self.global_delta = conformal_quantile(e[b], self.alpha)
        return self.q

    def _sigma(self, F):
        F = np.atleast_2d(np.asarray(F, float))
        D = np.hstack([np.ones((len(F), 1)), (F - self.f_mu) / self.f_sd])
        return np.maximum(D @ self.beta, self.floor)

    def delta_for(self, route, difficulty=None):
        if difficulty is None or not hasattr(self, "q"):
            return super().delta_for(route)
        return float(self.q * self._sigma(difficulty)[0])


def empirical_coverage(deltas, errors):
    """Fraction of held-out windows whose boundary error is inside the reported radius."""
    return float(np.mean(np.asarray(errors, float) <= np.asarray(deltas, float)))


def difficulty_features(det: Detection, dec: Decision):
    """Per-window features for AdaptiveConformalGauge: [largest residual ratio, lost-channel fraction, masked branches,
    route severity index]."""
    return np.array([min(det.resid_score, 20.0), float(dec.lost.mean()), float((dec.branch_mask == 0).sum()),
                     float(ROUTES_V2.index(dec.route))])


def summarise_latencies(rows, keys=("detector", "router", "mitigation", "model", "gauge", "total")):
    """rows: list of the `latencies_ms` dicts returned by process(); returns {stage: {mean, p50, p95, p99}} in ms."""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r], float)
        if v.size:
            out[k] = dict(mean=float(v.mean()), p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)),
                          p99=float(np.percentile(v, 99)), n=int(v.size))
    return out


# ====================================================================== the brain
class SmartAdaptiveMitigationBrain:
    """Legacy mode (no v2 component): detector -> router -> model -> gauge, exactly as before.
    v2 mode (detector=FaultDetector): detect -> decide -> mitigate -> reconstruct (model or physics fallback) -> gauge.

    known_missing: legacy list of (P,) bool arrays per branch (v2 builds the equivalent channel mask itself).
    oracle=True (or process(..., oracle_faults=labels)) skips detection and uses the true labels."""

    def __init__(self, tiers=None, gauge=None, known_missing=None, detector=None, bank=None, fallback=None,
                 router_kw=None, oracle=False):
        self.gauge = gauge or ConformalGauge()
        self.v2 = detector is not None or oracle
        self.oracle, self.fault_detector, self.known_missing = oracle, detector, known_missing
        if self.v2:
            self.router = RouterV2(tiers, None, bank, fallback, **(router_kw or {}))
            self.fallback = fallback
            self.layout = None
        else:
            self.detector, self.router = RealTimeFaultDetector(known_missing), AdaptiveRouter(tiers)

    def process(self, window, model_fn, oracle_faults=None):
        """window -> dict(route, mask, psi, assessment, actions, latencies_ms). model_fn(window, mask) -> (65, 65)."""
        if not self.v2:
            return self._process_legacy(window, model_fn)
        pc = time.perf_counter
        if self.layout is None:
            self.layout = self.router.layout = WindowLayout(window)
        L = self.layout
        t0 = pc()
        if oracle_faults is not None or self.oracle:
            if oracle_faults is None:
                raise ValueError("oracle brain needs oracle_faults=FaultLabels for every window")
            det = detection_from_labels(oracle_faults)
        else:
            det = self.fault_detector.detect_window(window, L)
        t1 = pc()
        dec = self.router.decide(det, L, have_model=model_fn is not None)
        t2 = pc()
        repaired = self.router.apply(window, det, dec, L)
        t3 = pc()
        if dec.use_fallback or model_fn is None:
            Xr = L.flatten(repaired)
            if self.fallback is None:
                raise RuntimeError("no model_fn and no physics fallback: nothing can reconstruct psi")
            avail = (~det.flagged | dec.transient | dec.virtual) & ~dec.lost & ~det.absent
            psi = self.fallback.predict(self.fallback.features(Xr), avail)
            dec.route = "physics_fallback"
        else:
            psi = model_fn(repaired, dec.branch_mask)
        t4 = pc()
        assessment = self.gauge.assess(psi, dec.route, difficulty_features(det, dec))
        t5 = pc()
        return dict(route=dec.route, mask=dec.branch_mask, psi=psi, assessment=assessment, actions=dec.channel_actions,
                    detection=det, decision=dec,
                    latencies_ms=dict(detector=(t1 - t0) * 1e3, router=(t2 - t1) * 1e3, mitigation=(t3 - t2) * 1e3,
                                      model=(t4 - t3) * 1e3, gauge=(t5 - t4) * 1e3, total=(t5 - t0) * 1e3))

    def _process_legacy(self, window, model_fn):
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
    # legacy self-check on a synthetic window: a spike, a dead redundant probe, and a dead critical branch
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
    print("adaptive_mitigation self-check ok (full v2 tests: python test_mitigation_v2.py)")
