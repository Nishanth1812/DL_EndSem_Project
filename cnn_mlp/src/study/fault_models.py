#!/usr/bin/env python
"""Module 4 v2: fault injection with ground-truth labels, plus the window <-> channel-matrix layout.

A model input window is a list with one array per sensor branch, each (T_b, P_b) = time x probes (a 1-D branch is
P = 1; leading singleton axes are fine). Different branches may have different T_b (e.g. magnetics 25 samples,
PF coils 20, saddle loops 250 in a 5 ms window). `WindowLayout` flattens such a window to a canonical channel matrix
X (t_ref, C) (C = total probes, 115 for task_1-3) on which all detectors / virtual sensors work, and writes repairs
back at native resolution.

Faults (all in standardised units; sigma = per-channel std, `scale=(mean, sigma)` arrays, default 0 / 1):
    stuck_rail   constant at +/- `magnitude` sigma from the onset            (amplifier at its rail)
    stuck_value  frozen at the last good value before the onset              (frozen ADC / held sample)
    drift        additive ramp `magnitude` sigma/ms from the onset           (thermal / integrator drift)
    spike        `n` samples (default 1) offset by `magnitude` sigma at the onset
    saturation   clipped to mean +/- `magnitude` sigma from the onset        (clipping, flat tops)
    dropout      samples set to `fill` (0.0 = the pipeline's zero fill, or NaN) for the duration
    group_loss   `group_loss_specs`: a dropout of the whole window on every probe of the chosen branches
Onset and duration are in ms of the 5 ms window, so the same spec applies at any sampling rate.
"""
from dataclasses import dataclass, field

import numpy as np

WINDOW_MS = 5.0
KINDS = ("stuck_rail", "stuck_value", "drift", "spike", "saturation", "dropout")
DEFAULT_MAGNITUDE = dict(stuck_rail=8.0, stuck_value=0.0, drift=0.6, spike=12.0, saturation=0.8, dropout=0.0)


@dataclass
class FaultSpec:
    channel: int
    kind: str
    onset_ms: float = 0.0
    duration_ms: float = None      # None = until the end of the window
    magnitude: float = None        # see module doc; None = DEFAULT_MAGNITUDE[kind]
    n: int = 1                     # spike: number of samples
    fill: float = 0.0              # dropout: value written into the hole (0.0 or nan)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown fault kind {self.kind!r}; choose from {KINDS}")
        if self.magnitude is None:
            self.magnitude = DEFAULT_MAGNITUDE[self.kind]


@dataclass
class FaultLabels:
    """Ground truth for one window. kind[c] == '' means channel c is healthy."""
    kind: np.ndarray               # (C,) object
    onset_ms: np.ndarray           # (C,) nan if healthy
    duration_ms: np.ndarray        # (C,) nan if healthy
    mask: np.ndarray               # (t_ref, C) bool, samples actually altered (canonical resolution)
    specs: list = field(default_factory=list)

    @property
    def faulty(self):
        return np.array([k != "" for k in self.kind])

    @property
    def channels(self):
        return np.flatnonzero(self.faulty)


def as_tp(a):
    """(T, P) view of one branch array (a 1-D branch becomes P = 1) and a function restoring the original shape."""
    a = np.asarray(a, dtype=np.float64)
    tp = a.squeeze()
    if tp.ndim == 0:
        tp = tp.reshape(1, 1)
    if tp.ndim == 1:
        tp = tp.reshape(-1, 1)
    if tp.ndim != 2:
        raise ValueError(f"expected one window per branch, got shape {a.shape}")
    return tp, (lambda x: x.reshape(a.shape))


def _to_ref(tp, t_ref):
    """Resample (T_b, P) to (t_ref, P): block mean when T_b is a multiple, linear interpolation otherwise (NaN-safe)."""
    T = tp.shape[0]
    if T == t_ref:
        return tp
    if T % t_ref == 0:
        blk = tp.reshape(t_ref, T // t_ref, -1)
        ok = np.isfinite(blk)
        n = ok.sum(axis=1)
        s = np.where(ok, blk, 0.0).sum(axis=1)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)
    pos = (np.arange(T) + 0.5) * t_ref / T - 0.5
    return np.stack([np.interp(np.arange(t_ref), pos, tp[:, p]) for p in range(tp.shape[1])], axis=1)


class WindowLayout:
    """Maps a branch-list window to / from the canonical channel matrix (t_ref, C)."""

    def __init__(self, window, t_ref=None):
        tps = [as_tp(b)[0] for b in window]
        self.T = [tp.shape[0] for tp in tps]
        self.probes = [tp.shape[1] for tp in tps]
        self.n_branches = len(tps)
        self.t_ref = int(t_ref or max(set(self.T), key=self.T.count))   # most common branch length
        self.branch_of = np.concatenate([np.full(p, b) for b, p in enumerate(self.probes)])
        self.probe_of = np.concatenate([np.arange(p) for p in self.probes])
        self.offsets = np.concatenate([[0], np.cumsum(self.probes)])
        self.n_channels = int(self.offsets[-1])

    def flatten(self, window):
        return np.concatenate([_to_ref(as_tp(b)[0], self.t_ref) for b in window], axis=1)

    def channels_of(self, branch):
        return np.arange(self.offsets[branch], self.offsets[branch + 1])

    def branch_mask(self, channel_bad, frac=0.5):
        """(n_branches,) float32 health mask: a branch is masked when >= frac of its probes are bad."""
        out = np.ones(self.n_branches, dtype=np.float32)
        for b in range(self.n_branches):
            if channel_bad[self.channels_of(b)].mean() >= frac:
                out[b] = 0.0
        return out

    def write_back(self, window, X, sample_mask):
        """Copy of `window` where, per channel, native samples that fall in canonical samples flagged by
        sample_mask (t_ref, C) take their value from the canonical series X (linear interpolation)."""
        out = [np.array(b, dtype=np.float64, copy=True) for b in window]
        for b in range(self.n_branches):
            cols = self.channels_of(b)
            if not sample_mask[:, cols].any():
                continue
            tp, restore = as_tp(out[b])
            tp = tp.copy()
            T = self.T[b]
            j = np.arange(T)
            pos = (j + 0.5) * self.t_ref / T - 0.5
            k = np.minimum(j * self.t_ref // T, self.t_ref - 1)
            for p, c in enumerate(cols):
                m = sample_mask[k, c]
                if m.any():
                    tp[m, p] = np.interp(pos[m], np.arange(self.t_ref), X[:, c])
            out[b] = restore(tp)
        return out


# ------------------------------------------------------------------ injection
def _span(spec, T, window_ms):
    a = int(np.clip(np.floor(spec.onset_ms / window_ms * T + 1e-9), 0, T - 1))
    if spec.duration_ms is None:
        b = T
    else:
        b = int(np.clip(np.ceil((spec.onset_ms + spec.duration_ms) / window_ms * T - 1e-9), a + 1, T))
    return a, b


def inject_series(x, spec, window_ms=WINDOW_MS, mean=0.0, sigma=1.0):
    """Apply one fault to a 1-D series; returns (faulty copy, bool mask of samples that actually changed)."""
    x = np.asarray(x, dtype=np.float64)
    T = x.size
    a, b = _span(spec, T, window_ms)
    y = x.copy()
    t_ms = (np.arange(T) - a) * (window_ms / T)
    k = spec.kind
    if k == "stuck_rail":
        y[a:b] = mean + np.sign(spec.magnitude) * abs(spec.magnitude) * sigma
    elif k == "stuck_value":
        y[a:b] = x[a - 1] if a > 0 else x[a]
    elif k == "drift":
        y[a:b] = x[a:b] + spec.magnitude * sigma * t_ms[a:b]
    elif k == "spike":
        y[a:min(T, a + spec.n)] += spec.magnitude * sigma
    elif k == "saturation":
        lo, hi = mean - abs(spec.magnitude) * sigma, mean + abs(spec.magnitude) * sigma
        y[a:b] = np.clip(x[a:b], lo, hi)
    elif k == "dropout":
        y[a:b] = spec.fill
    changed = ~((y == x) | (np.isnan(y) & np.isnan(x)))
    return y, changed


def _labels(specs, changed_by_spec, layout_C, t_ref):
    kind = np.full(layout_C, "", dtype=object)
    on = np.full(layout_C, np.nan)
    du = np.full(layout_C, np.nan)
    mask = np.zeros((t_ref, layout_C), dtype=bool)
    for s, ch in zip(specs, changed_by_spec):
        if ch is None or not ch.any():       # e.g. clipping that clipped nothing is not a fault
            continue
        kind[s.channel], on[s.channel] = s.kind, s.onset_ms
        du[s.channel] = np.nan if s.duration_ms is None else s.duration_ms
        if s.duration_ms is None:
            du[s.channel] = WINDOW_MS - s.onset_ms
        # canonical-resolution mask: a canonical sample is altered if any native sample in it changed
        T = ch.size
        idx = np.minimum(np.flatnonzero(ch) * t_ref // T, t_ref - 1)
        mask[idx, s.channel] = True
    return FaultLabels(kind, on, du, mask, list(specs))


def inject(X, specs, window_ms=WINDOW_MS, scale=None):
    """Canonical-matrix injection. X (T, C); returns (faulty copy, FaultLabels)."""
    X = np.asarray(X, dtype=np.float64)
    mean, sigma = (np.zeros(X.shape[1]), np.ones(X.shape[1])) if scale is None else scale
    Y, ch = X.copy(), []
    for s in specs:
        Y[:, s.channel], c = inject_series(Y[:, s.channel], s, window_ms, mean[s.channel], sigma[s.channel])
        ch.append(c)
    return Y, _labels(specs, ch, X.shape[1], X.shape[0])


def inject_window(window, layout, specs, window_ms=WINDOW_MS, scale=None):
    """Native-resolution injection into a branch-list window; returns (faulty window, FaultLabels)."""
    mean, sigma = (np.zeros(layout.n_channels), np.ones(layout.n_channels)) if scale is None else scale
    out = [as_tp(b)[0].copy() for b in window]
    restore = [as_tp(b)[1] for b in window]
    ch = []
    for s in specs:
        b, p = layout.branch_of[s.channel], layout.probe_of[s.channel]
        out[b][:, p], c = inject_series(out[b][:, p], s, window_ms, mean[s.channel], sigma[s.channel])
        ch.append(c)
    return [r(o) for r, o in zip(restore, out)], _labels(specs, ch, layout.n_channels, layout.t_ref)


def group_loss_specs(layout, branches, fill=0.0):
    """A dropout of the whole window on every probe of the given branches (a group / cable / DAQ-card loss)."""
    return [FaultSpec(int(c), "dropout", 0.0, None, fill=fill) for b in branches for c in layout.channels_of(b)]


def random_specs(rng, n_channels, kind, n=1, channels=None, window_ms=WINDOW_MS, **kw):
    """n random faults of one kind on distinct channels. Onset is uniform over the first 80 % of the window for the
    persistent kinds; dropout duration is uniform in [0.2 ms, remainder]. kw overrides magnitude / fill / n / duration_ms."""
    chans = rng.choice(n_channels if channels is None else channels, size=n, replace=False)
    specs = []
    for c in chans:
        onset = float(rng.uniform(0.0, 0.8 * window_ms))
        s = dict(onset_ms=onset, duration_ms=None)
        if kind == "dropout":
            s["duration_ms"] = float(rng.uniform(0.2, max(0.2, window_ms - onset)))
        if kind == "spike":
            s["onset_ms"] = float(rng.uniform(0.0, window_ms * 0.95))
        if kind in ("stuck_rail",):
            s["magnitude"] = float(rng.choice([-1.0, 1.0]) * DEFAULT_MAGNITUDE["stuck_rail"])
        if kind == "drift":
            s["magnitude"] = float(rng.choice([-1.0, 1.0]) * DEFAULT_MAGNITUDE["drift"])
        if kind == "spike":
            s["magnitude"] = float(rng.choice([-1.0, 1.0]) * DEFAULT_MAGNITUDE["spike"])
        s.update(kw)
        specs.append(FaultSpec(int(c), kind, **s))
    return specs


# ------------------------------------------------------------------ detection scoring
def benchmark_detector(detect_fn, X_windows, kinds=KINDS, n_faults=1, seed=0, scale=None, spec_kw=None,
                       window_ms=WINDOW_MS):
    """Score a detector on clean held-out windows X_windows (N, T, C) with injected faults of each kind.

    detect_fn(X (T, C)) -> object with .flagged (C,) bool and .score (C,) float (higher = more suspicious).
    Per kind (each kind is run on its own copy of every window, `n_faults` faults per window):
      recall      faulty channels flagged / faulty channels injected
      precision   flagged faulty channels / all flagged channels in those windows
      f1
      iso_top1    fraction of windows whose highest-score channel is an injected one
      iso_top3    fraction of windows with an injected channel in the 3 highest scores
    plus 'clean': fpr_window (fraction of clean windows with any flag), fpr_channel (flagged / channels)."""
    rng = np.random.default_rng(seed)
    N, T, C = X_windows.shape
    spec_kw = spec_kw or {}
    out = {}
    fl = [detect_fn(X_windows[i]).flagged for i in range(N)]
    out["clean"] = dict(fpr_window=float(np.mean([f.any() for f in fl])), fpr_channel=float(np.mean(fl)), n_windows=N)
    for kind in kinds:
        tp = fn = fp = hit1 = hit3 = used = 0
        for i in range(N):
            specs = random_specs(rng, C, kind, n_faults, window_ms=window_ms, **spec_kw.get(kind, {}))
            Y, lab = inject(X_windows[i], specs, window_ms, scale)
            truth = lab.faulty
            if not truth.any():
                continue
            used += 1
            d = detect_fn(Y)
            tp += int((d.flagged & truth).sum()); fn += int((~d.flagged & truth).sum()); fp += int((d.flagged & ~truth).sum())
            order = np.argsort(-np.nan_to_num(d.score, nan=-np.inf))
            hit1 += int(truth[order[0]]); hit3 += int(truth[order[:3]].any())
        rec = tp / max(1, tp + fn)
        prec = tp / max(1, tp + fp)
        out[kind] = dict(n_windows=used, recall=rec, precision=prec, f1=2 * rec * prec / max(1e-12, rec + prec),
                         iso_top1=hit1 / max(1, used), iso_top3=hit3 / max(1, used))
    return out
