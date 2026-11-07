#!/usr/bin/env python
"""Module 3 v2: model-agnostic sensor attribution (Shapley, worst-k, minimum sensor set).

Everything works on a callable  value_fn(mask) -> float  with  mask: bool ndarray (n_sensors,), True = sensor AVAILABLE.
The value is an ERROR (boundary error in cm, X-point error in cm or NRMSE): bigger = worse. With vectorised=True the
callable receives a bool array (m, n_sensors) and returns (m,), so one model pass can be batched over many masks.

Cooperative game used throughout (the "failure game"):
    players = sensors (or the 8 groups);  f(F) = value_fn(~F)  for a set F of FAILED players.
    Shapley phi_i = average over orderings of  f(F + i) - f(F)   ("damage attributable to losing i", same unit as the value).
    Efficiency: sum_i phi_i = f(all failed) - f(none failed).  f(all failed) means NO sensor at all: read it as an
    extrapolation of the model (usually saturated), not a physical state.  The `max_failed` option limits the orderings to
    the first K failures, which is the realistic regime (K ~ 5-20 simultaneous failures) and is cheaper.
Interaction index (Grabisch-Roubens) I_ij = E_S[f(S+i+j) - f(S+i) - f(S+j) + f(S)]:
    I_ij > 0  -> BACKUP / REDUNDANT pair: the damage appears only when both are lost (i and j cover for each other);
    I_ij < 0  -> OVERLAP: either loss already does the damage (shared critical information, saturation).
Classification (see classify_sensors): drop-one delta_i = f({i}) - f(none).
    critical  : delta_i >= crit_tol            (losing it alone is already unsafe)
    redundant : delta_i <  red_tol and |phi_i| < red_tol   (matters neither alone nor in company)
    backup    : delta_i <  red_tol but phi_i >= red_tol    (hidden criticality: harmless alone, harmful with others gone)
    minor     : everything else.

No torch needed; the model lives behind value_fn. Evaluations are cached by mask bytes and counted.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Inventory (verified against zarr shots 12384 / 13222 and the run log "input shapes"): 15+40+18+18+12+10+1+1 = 115
# ----------------------------------------------------------------------------------------------------------------------
SIGNAL_SHORT = ("flux_loop", "ccbv", "obr", "obv", "saddle_v", "coil_i", "solenoid_i", "ip")
GROUP_COUNTS = (15, 40, 18, 18, 12, 10, 1, 1)
_FL = ["CC03", "CC04", "CC05", "CC07", "CC09", "P3L/4", "P3U/1", "P3U/4", "P4L/1", "P4L/4", "P4U/4", "P5L/1", "P5L/4",
       "P5U/1", "P6L/1"]
_COIL = ["P2IL", "P2IU", "P2OL", "P2OU", "P3L", "P3U", "P4L", "P4U", "P5L", "P5U"]


@dataclass
class SensorInventory:
    group_names: List[str]
    counts: List[int]
    sensor_names: List[str]
    group_of: np.ndarray = field(init=False)
    offsets: np.ndarray = field(init=False)

    def __post_init__(self):
        assert sum(self.counts) == len(self.sensor_names)
        self.group_of = np.repeat(np.arange(len(self.counts)), self.counts)
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)])

    @property
    def n_sensors(self) -> int:
        return len(self.sensor_names)

    @property
    def n_groups(self) -> int:
        return len(self.group_names)

    def indices(self, g: int) -> np.ndarray:
        return np.arange(self.offsets[g], self.offsets[g + 1])

    def mask_from_failed_sensors(self, failed: Sequence[int]) -> np.ndarray:
        m = np.ones(self.n_sensors, bool)
        m[list(failed)] = False
        return m

    def mask_from_failed_groups(self, failed_groups: Sequence[int]) -> np.ndarray:
        return ~np.isin(self.group_of, list(failed_groups))

    def branch_health(self, mask: np.ndarray) -> np.ndarray:
        """(8,) float: 0 only if every sensor of the branch is gone (what the failure-aware model's health input can express)."""
        return np.array([float(mask[self.indices(g)].any()) for g in range(self.n_groups)], dtype=np.float32)

    def apply_mask_to_inputs(self, inputs, mask: np.ndarray, fill=0.0):
        """Zero (or fill) failed sensors in a list of 8 input tensors/arrays. Works for numpy and torch (copies).

        Profile branches 0-5 carry the sensor axis LAST: (..., T, n_sensors) as in the run log shapes
        [(1,1,25,15),(1,1,25,40),(1,1,25,18),(1,1,25,18),(1,1,250,12),(1,1,20,10),(1,1,20),(1,1,20)];
        the scalar branches 6-7 (solenoid, ip) have no sensor axis and are blanked as a whole.
        NOTE the existing robustness._zero_channels indexes axis -2, which for the profile branches is TIME, not sensors.
        """
        out = []
        for g, x in enumerate(inputs):
            x = x.clone() if hasattr(x, "clone") else np.array(x, copy=True)
            keep = mask[self.indices(g)]
            if self.counts[g] == 1:
                if not keep[0]:
                    x[...] = fill
            else:
                if x.shape[-1] != self.counts[g]:
                    raise ValueError(f"branch {g} ({self.group_names[g]}): last axis {x.shape[-1]} != {self.counts[g]} sensors")
                for c in np.flatnonzero(~keep):
                    x[..., int(c)] = fill
            out.append(x)
        return out


def default_inventory() -> SensorInventory:
    names = ([f"AMB_FL/{n}" for n in _FL] + [f"AMB_CCBV{i:02d}" for i in range(1, 41)] +
             [f"AMB_OBR{i:02d}" for i in range(1, 19)] + [f"AMB_OBV{i:02d}" for i in range(1, 19)] +
             [f"XMB_SAD/OUT/M{i:02d}" for i in range(1, 13)] + [f"AMC_{c} FEED CURRENT" for c in _COIL] +
             ["AMC_SOL CURRENT", "summary/ip"])
    return SensorInventory(list(SIGNAL_SHORT), list(GROUP_COUNTS), names)


# ----------------------------------------------------------------------------------------------------------------------
# Cached evaluator
# ----------------------------------------------------------------------------------------------------------------------
class Evaluator:
    """Caches value_fn by mask bytes; batches uncached masks (vectorised=True) and counts the real model evaluations."""

    def __init__(self, value_fn: Callable, vectorised: bool = False, batch_size: int = 64):
        self.fn, self.vec, self.bs = value_fn, vectorised, batch_size
        self.cache: Dict[bytes, float] = {}
        self.n_requests = 0
        self.n_batches = 0

    @property
    def n_evals(self) -> int:
        return len(self.cache)

    def many(self, masks: Sequence[np.ndarray]) -> np.ndarray:
        keys = [np.packbits(np.asarray(m, bool)).tobytes() + bytes([len(m) % 8]) for m in masks]
        self.n_requests += len(keys)
        todo = {}
        for k, m in zip(keys, masks):
            if k not in self.cache and k not in todo:
                todo[k] = np.asarray(m, bool)
        items = list(todo.items())
        step = self.bs if self.vec else 1
        for s in range(0, len(items), step):
            chunk = items[s:s + step]
            if self.vec:
                vals = np.asarray(self.fn(np.stack([m for _, m in chunk])), float).reshape(-1)
                assert len(vals) == len(chunk), "vectorised value_fn must return one value per mask"
            else:
                vals = [float(self.fn(m)) for _, m in chunk]
            self.n_batches += 1
            for (k, _), v in zip(chunk, vals):
                self.cache[k] = float(v)
        return np.array([self.cache[k] for k in keys])

    def __call__(self, mask) -> float:
        return float(self.many([mask])[0])


class Game:
    """Failure game over p players. value(failed_bool(p)) -> error. Players are sensors or groups."""

    def __init__(self, ev: Evaluator, p: int, expand: Optional[Callable] = None, names: Optional[List[str]] = None):
        self.ev, self.p = ev, p
        self.expand = expand or (lambda failed: ~failed)  # failed players -> sensor availability mask
        self.names = names or [str(i) for i in range(p)]

    @classmethod
    def sensors(cls, ev, inv: SensorInventory):
        return cls(ev, inv.n_sensors, names=inv.sensor_names)

    @classmethod
    def groups(cls, ev, inv: SensorInventory):
        return cls(ev, inv.n_groups, expand=lambda failed: ~failed[inv.group_of], names=inv.group_names)

    def many(self, faileds) -> np.ndarray:
        return self.ev.many([self.expand(np.asarray(f, bool)) for f in faileds])

    def value(self, failed) -> float:
        return float(self.many([failed])[0])

    def value_bits(self, bits: int) -> float:
        return self.value(_bits_to_bool(bits, self.p))


def _bits_to_bool(bits: int, p: int) -> np.ndarray:
    return np.array([(bits >> i) & 1 for i in range(p)], bool)


# ----------------------------------------------------------------------------------------------------------------------
# Exact enumeration (groups: 2^8 = 256 coalitions)
# ----------------------------------------------------------------------------------------------------------------------
def full_table(game: Game, max_players: int = 20) -> np.ndarray:
    """vals[b] = f(failed set with bitmask b) for all 2^p sets, evaluated as one batch."""
    assert game.p <= max_players
    return game.many([_bits_to_bool(b, game.p) for b in range(2 ** game.p)])


def exact_shapley(vals: np.ndarray, p: int) -> np.ndarray:
    w = [math.factorial(s) * math.factorial(p - s - 1) / math.factorial(p) for s in range(p)]
    phi = np.zeros(p)
    allb = np.arange(2 ** p)
    pop = np.array([bin(b).count("1") for b in allb])
    for i in range(p):
        S = allb[(allb >> i) & 1 == 0]
        phi[i] = np.sum(np.array(w)[pop[S]] * (vals[S | (1 << i)] - vals[S]))
    return phi


def exact_interaction(vals: np.ndarray, p: int) -> np.ndarray:
    """(p, p) Shapley interaction index I_ij (diagonal 0). >0 backup/redundant pair, <0 overlap."""
    allb = np.arange(2 ** p)
    pop = np.array([bin(b).count("1") for b in allb])
    w = np.array([math.factorial(s) * math.factorial(p - s - 2) / math.factorial(p - 1) for s in range(p - 1)])
    I = np.zeros((p, p))
    for i, j in itertools.combinations(range(p), 2):
        S = allb[((allb >> i) & 1 == 0) & ((allb >> j) & 1 == 0)]
        d = vals[S | (1 << i) | (1 << j)] - vals[S | (1 << i)] - vals[S | (1 << j)] + vals[S]
        I[i, j] = I[j, i] = np.sum(w[pop[S]] * d)
    return I


# ----------------------------------------------------------------------------------------------------------------------
# Sampled Shapley: antithetic permutations, Welford running mean/SE, CI-width + budget stopping rule
# ----------------------------------------------------------------------------------------------------------------------
def sampled_shapley(game: Game, rng: np.random.Generator, max_pairs: int = 200, budget_evals: Optional[int] = None,
                    ci_tol: Optional[float] = None, top_k: Optional[int] = None, min_pairs: int = 8,
                    check_every: int = 4, max_failed: Optional[int] = None, z: float = 1.96) -> dict:
    """Permutation-sampling Shapley of the failure game.

    One SAMPLE = a permutation and its reverse (antithetic); the sample value is the mean of their marginal vectors, so
    samples are i.i.d. and SE = sd/sqrt(n_pairs). Each path costs K+1 evaluations (K = max_failed or p), batched in one
    value_many call, and prefixes already in the cache are free. With max_failed=K only the first K failures of each
    ordering count and the estimate is rescaled by p/K: it estimates the MEAN MARGINAL DAMAGE over coalition sizes
    0..K-1 (a truncated semi-value; efficiency holds only for K=p).

    Stopping: n_pairs >= max_pairs, or unique evaluations >= budget_evals, or (after min_pairs) the CI half-width
    z*SE of the top_k largest |phi| (all players if None) is <= ci_tol.
    """
    p = game.p
    K = p if max_failed is None else int(min(max_failed, p))
    scale = p / K
    n, mean, m2 = 0, np.zeros(p), np.zeros(p)
    reason = "max_pairs"
    while n < max_pairs:
        if budget_evals is not None and game.ev.n_evals + K + 1 > budget_evals:
            reason = "budget"
            break
        perm = rng.permutation(p)
        c = np.zeros(p)
        for order in (perm, perm[::-1]):
            fail = np.zeros(p, bool)
            masks = [fail.copy()]
            for t in range(K):
                fail[order[t]] = True
                masks.append(fail.copy())
            v = game.many(masks)
            c[order[:K]] += 0.5 * scale * np.diff(v)
        n += 1
        d = c - mean
        mean += d / n
        m2 += d * (c - mean)
        if ci_tol is not None and n >= min_pairs and n % check_every == 0:
            se = np.sqrt(m2 / (n - 1) / n)
            idx = np.argsort(-np.abs(mean))[: (top_k or p)]
            if np.max(z * se[idx]) <= ci_tol:
                reason = "ci_tol"
                break
    se = np.sqrt(m2 / max(n - 1, 1) / max(n, 1))
    return dict(phi=mean, se=se, n_pairs=n, n_evals=game.ev.n_evals, stop=reason, max_failed=K)


def sampled_interaction(game: Game, i: int, j: int, rng: np.random.Generator, n_samples: int = 64,
                        max_failed: Optional[int] = None) -> tuple:
    """Monte-Carlo I_ij: draw |S| uniform in 0..K-2 (K=max_failed or p), S random among the others. Returns (mean, se)."""
    p = game.p
    others = np.array([k for k in range(p) if k not in (i, j)])
    top = (p if max_failed is None else min(max_failed, p)) - 1  # sizes 0..top-1
    out = []
    for _ in range(n_samples):
        s = int(rng.integers(0, max(top, 1)))
        S = np.zeros(p, bool)
        S[rng.choice(others, size=min(s, len(others)), replace=False)] = True
        Si, Sj, Sij = S.copy(), S.copy(), S.copy()
        Si[i] = Sj[j] = Sij[i] = Sij[j] = True
        v = game.many([S, Si, Sj, Sij])
        out.append(v[3] - v[1] - v[2] + v[0])
    out = np.array(out)
    return float(out.mean()), float(out.std(ddof=1) / math.sqrt(len(out))) if len(out) > 1 else 0.0


# ----------------------------------------------------------------------------------------------------------------------
# Drop-one, classification, rankings
# ----------------------------------------------------------------------------------------------------------------------
def drop_one(game: Game, context_failed: Optional[np.ndarray] = None) -> dict:
    """delta_i = f(context + i) - f(context). Default context = nothing failed. Cost p+1 evaluations."""
    base = np.zeros(game.p, bool) if context_failed is None else np.asarray(context_failed, bool)
    masks = [base] + [np.where(np.arange(game.p) == i, True, base) for i in range(game.p)]
    v = game.many(masks)
    delta = v[1:] - v[0]
    delta[base] = np.nan
    return dict(base=float(v[0]), delta=delta, value=v[1:])


def alone_value(game: Game) -> np.ndarray:
    """f(everything except i failed): the error when ONLY player i is available (sufficiency)."""
    p = game.p
    return game.many([~(np.arange(p) == i) for i in range(p)])


def classify_sensors(delta: np.ndarray, phi: np.ndarray, crit_tol: float = 0.5, red_tol: float = 0.1) -> List[str]:
    out = []
    for d, f in zip(delta, phi):
        if d >= crit_tol:
            out.append("critical")
        elif d < red_tol and abs(f) < red_tol:
            out.append("redundant")
        elif d < red_tol:
            out.append("backup")
        else:
            out.append("minor")
    return out


def ranking(score: np.ndarray, names: Sequence[str], descending: bool = True) -> List[dict]:
    order = np.argsort(-np.asarray(score, float) if descending else np.asarray(score, float), kind="stable")
    return [dict(rank=r + 1, index=int(i), name=names[i], score=float(score[i])) for r, i in enumerate(order)]


# ----------------------------------------------------------------------------------------------------------------------
# Worst-k failure sets
# ----------------------------------------------------------------------------------------------------------------------
def worst_k_exhaustive(game: Game, ks: Sequence[int], limit: int = 300000) -> Dict[int, dict]:
    res = {}
    for k in ks:
        if math.comb(game.p, k) > limit:
            continue
        combos = list(itertools.combinations(range(game.p), k))
        fl = []
        for c in combos:
            f = np.zeros(game.p, bool)
            f[list(c)] = True
            fl.append(f)
        v = game.many(fl)
        b = int(np.argmax(v))
        res[k] = dict(k=k, set=list(combos[b]), value=float(v[b]), method="exhaustive")
    return res


def worst_k_beam(game: Game, kmax: int, width: int = 1) -> Dict[int, dict]:
    """width=1 is greedy forward selection; width>=p is exhaustive-ish. Cost ~ width * p per level."""
    p = game.p
    beam = [(frozenset(), game.value(np.zeros(p, bool)))]
    res = {}
    for k in range(1, kmax + 1):
        cand = {}
        for s, _ in beam:
            for i in range(p):
                if i not in s:
                    cand.setdefault(s | {i}, None)
        keys = list(cand)
        fl = []
        for s in keys:
            f = np.zeros(p, bool)
            f[list(s)] = True
            fl.append(f)
        v = game.many(fl)
        order = np.argsort(-v, kind="stable")[:width]
        beam = [(keys[o], float(v[o])) for o in order]
        res[k] = dict(k=k, set=sorted(beam[0][0]), value=beam[0][1], method=f"beam{width}" if width > 1 else "greedy")
    return res


# ----------------------------------------------------------------------------------------------------------------------
# Minimum sensor set under a tolerance
# ----------------------------------------------------------------------------------------------------------------------
def _summ(game, failed_idx, curve, tol, method):
    feas = [c for c in curve if c["value"] <= tol]
    best = min(feas, key=lambda c: c["n_available"]) if feas else None
    return dict(method=method, tol=tol, curve=curve, min_available=None if best is None else best["n_available"],
                min_value=None if best is None else best["value"],
                failed=None if best is None else best["failed"])


def greedy_backward(game: Game, tol: float, continue_past_tol: bool = True) -> dict:
    """Drop, at each step, the player whose loss increases the error least. The curve is recorded even beyond tol."""
    p, failed = game.p, []
    curve = [dict(n_available=p, value=game.value(np.zeros(p, bool)), failed=[])]
    while len(failed) < p:
        rest = [i for i in range(p) if i not in failed]
        fl = []
        for i in rest:
            f = np.zeros(p, bool)
            f[failed + [i]] = True
            fl.append(f)
        v = game.many(fl)
        b = int(np.argmin(v))
        failed = failed + [rest[b]]
        curve.append(dict(n_available=p - len(failed), value=float(v[b]), failed=sorted(failed)))
        if v[b] > tol and not continue_past_tol:
            break
    return _summ(game, failed, curve, tol, "greedy_backward")


def beam_backward(game: Game, tol: float, width: int = 5, max_levels: Optional[int] = None) -> dict:
    """Beam search over FEASIBLE failed sets (value <= tol): the deepest feasible level is the smallest sensor set found."""
    p = game.p
    beam = [frozenset()]
    curve = [dict(n_available=p, value=game.value(np.zeros(p, bool)), failed=[])]
    for lvl in range(1, (max_levels or p) + 1):
        cand = {}
        for s in beam:
            for i in range(p):
                if i not in s:
                    cand.setdefault(s | {i}, None)
        keys = list(cand)
        fl = []
        for s in keys:
            f = np.zeros(p, bool)
            f[list(s)] = True
            fl.append(f)
        v = game.many(fl)
        feas = [(float(v[k]), keys[k]) for k in range(len(keys)) if v[k] <= tol]
        if not feas:
            break
        feas.sort(key=lambda t: t[0])
        beam = [s for _, s in feas[:width]]
        curve.append(dict(n_available=p - lvl, value=feas[0][0], failed=sorted(beam[0])))
    return _summ(game, None, curve, tol, f"beam_backward_w{width}")


def ranked_backward(game: Game, order_least_important_first: Sequence[int], tol: float) -> dict:
    """Shapley/drop-one guided: try to fail players in the given order, keep each one only if the error stays <= tol.
    Cost: one evaluation per player (p). Not optimal, but a cheap upper bound on the minimum set size."""
    p, failed = game.p, []
    curve = [dict(n_available=p, value=game.value(np.zeros(p, bool)), failed=[])]
    for i in order_least_important_first:
        f = np.zeros(p, bool)
        f[failed + [int(i)]] = True
        v = game.value(f)
        if v <= tol:
            failed.append(int(i))
            curve.append(dict(n_available=p - len(failed), value=v, failed=sorted(failed)))
    return _summ(game, failed, curve, tol, "ranked_backward")


def exhaustive_group_curve(vals: np.ndarray, p: int) -> List[dict]:
    """Best (lowest-error) and worst subset for every number of FAILED groups, from the full 2^p table."""
    pop = np.array([bin(b).count("1") for b in range(2 ** p)])
    out = []
    for k in range(p + 1):
        idx = np.flatnonzero(pop == k)
        b, w = idx[np.argmin(vals[idx])], idx[np.argmax(vals[idx])]
        out.append(dict(n_failed=k, n_available=p - k, best_value=float(vals[b]),
                        best_failed=[i for i in range(p) if (b >> i) & 1], worst_value=float(vals[w]),
                        worst_failed=[i for i in range(p) if (w >> i) & 1]))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# End-to-end driver, JSON, summaries
# ----------------------------------------------------------------------------------------------------------------------
def run_full_analysis(value_fn: Callable, inv: SensorInventory, tol: float = 2.0, vectorised: bool = False, seed: int = 0,
                      sensor_budget_evals: int = 6000, sensor_max_failed: Optional[int] = 20, ci_tol: Optional[float] = None,
                      worst_k_max: int = 10, beam_width: int = 5, crit_tol: Optional[float] = None, red_tol: Optional[float] = None,
                      do_sensor_shapley: bool = True, do_min_set: bool = True) -> dict:
    """Run the full v2 analysis. Returns a JSON-serialisable dict (use json.dumps(result)).

    Cost (unique model evaluations): groups 256 + drop-one 116 + sensor Shapley <= sensor_budget_evals + worst-k
    ~ beam_width*p*worst_k_max (cached overlaps are free) + min-set greedy ~ p^2/2 (6.6k for 115 sensors).
    """
    ev = Evaluator(value_fn, vectorised=vectorised)
    rng = np.random.default_rng(seed)
    gg, gs = Game.groups(ev, inv), Game.sensors(ev, inv)
    res = dict(tol=tol, n_sensors=inv.n_sensors, groups=inv.group_names, counts=inv.counts)

    vals = full_table(gg)
    res["baseline"] = float(vals[0])
    res["all_groups_failed"] = float(vals[-1])
    phi_g = exact_shapley(vals, inv.n_groups)
    I_g = exact_interaction(vals, inv.n_groups)
    res["group"] = dict(shapley=phi_g, drop_one=drop_one(gg)["delta"], alone=alone_value(gg), interaction=I_g,
                        ranking_shapley=ranking(phi_g, inv.group_names),
                        worst_k_exhaustive=exhaustive_group_curve(vals, inv.n_groups),
                        worst_k_greedy=worst_k_beam(gg, inv.n_groups, 1), worst_k_beam=worst_k_beam(gg, inv.n_groups, beam_width),
                        n_evals=ev.n_evals)

    margin = tol - res["baseline"] if tol > res["baseline"] else tol  # headroom the tolerance leaves above the clean error
    crit_tol = 0.5 * margin if crit_tol is None else crit_tol        # one loss eats >= half the headroom
    red_tol = 0.05 * margin if red_tol is None else red_tol          # < 5 % of the headroom, alone and in company
    res["class_thresholds"] = dict(critical=crit_tol, redundant=red_tol)
    d1 = drop_one(gs)
    res["sensor"] = dict(drop_one=d1["delta"], ranking_drop_one=ranking(d1["delta"], inv.sensor_names))
    if do_sensor_shapley:
        sh = sampled_shapley(gs, rng, max_pairs=10 ** 6, budget_evals=sensor_budget_evals, ci_tol=ci_tol,
                             max_failed=sensor_max_failed)
        cls = classify_sensors(d1["delta"], sh["phi"], crit_tol, red_tol)
        res["sensor"].update(shapley=sh["phi"], shapley_se=sh["se"], shapley_pairs=sh["n_pairs"], shapley_stop=sh["stop"],
                             shapley_max_failed=sh["max_failed"], cls=cls,
                             ranking_shapley=ranking(sh["phi"], inv.sensor_names),
                             shapley_group_sum=[float(sh["phi"][inv.indices(g)].sum()) for g in range(inv.n_groups)])
    kmax = min(worst_k_max, inv.n_sensors)
    res["sensor"]["worst_k_greedy"] = worst_k_beam(gs, kmax, 1)
    res["sensor"]["worst_k_beam"] = worst_k_beam(gs, kmax, beam_width)
    if do_min_set:
        ms = dict(groups_greedy=greedy_backward(gg, tol), sensors_greedy=greedy_backward(gs, tol),
                  sensors_beam=beam_backward(gs, tol, beam_width))
        if do_sensor_shapley:
            order = np.argsort(sh["phi"])  # least damaging first
            ms["sensors_shapley_ranked"] = ranked_backward(gs, order, tol)
        ms["groups_exhaustive"] = exhaustive_group_curve(vals, inv.n_groups)
        ms["note"] = ("Any earlier '47 -> 25 probes' claim is unsupported: 47 came from a hard-coded per-branch probe table "
                      "that does not match the real 115-channel inventory. Only the numbers in this structure are measured.")
        res["min_set"] = ms
    res["cost"] = dict(unique_evals=ev.n_evals, requests=ev.n_requests, batches=ev.n_batches)
    return jsonable(res)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def to_json(res: dict, path: Optional[str] = None) -> str:
    s = json.dumps(res, indent=1)
    if path:
        open(path, "w", encoding="utf8").write(s)
    return s


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def latex_table(header, rows, caption="", label="") -> str:
    cols = "l" + "r" * (len(header) - 1)
    body = " \\\\\n".join(" & ".join(str(c).replace("_", "\\_") for c in r) for r in rows)
    return (f"\\begin{{table}}[t]\\centering\\caption{{{caption}}}\\label{{{label}}}\n\\begin{{tabular}}{{{cols}}}\\toprule\n"
            f"{' & '.join(h.replace('_', chr(92) + '_') for h in header)} \\\\\\midrule\n{body} \\\\\n\\bottomrule\\end{{tabular}}\\end{{table}}")


def summary_tables(res: dict, top: int = 10, fmt: str = "md", unit: str = "cm") -> str:
    """Markdown (fmt='md') or LaTeX (fmt='tex') tables: groups, top sensors, worst-k, minimum set."""
    mk = _md_table if fmt == "md" else (lambda h, r: latex_table(h, r))
    g = res["group"]
    rows = [[n, c, f"{s:.3f}", f"{d:.3f}", f"{a:.3f}"] for n, c, s, d, a in
            zip(res["groups"], res["counts"], g["shapley"], g["drop_one"], g["alone"])]
    parts = [f"Groups (baseline {res['baseline']:.3f} {unit}, tolerance {res['tol']} {unit}); Shapley/drop-one in {unit}; "
             f"'alone' = error with only that group available", mk(["group", "n", "shapley", "drop_one", "alone"], rows)]
    s = res["sensor"]
    key = "ranking_shapley" if "ranking_shapley" in s else "ranking_drop_one"
    rows = []
    for r in s[key][:top]:
        i = r["index"]
        rows.append([r["name"], f"{r['score']:.3f}", f"{s['drop_one'][i]:.3f}",
                     f"{s['shapley_se'][i]:.3f}" if "shapley_se" in s else "-", s["cls"][i] if "cls" in s else "-"])
    parts += [f"Top {top} sensors by {key.replace('ranking_', '')}", mk(["sensor", "score", "drop_one", "se", "class"], rows)]
    def _at(d, k):  # keys are int before json round-trip and str after it
        e = d.get(k, d.get(str(k)))
        return "-" if e is None else f"{e['value']:.3f}"

    wk = [[k, f"{g['worst_k_exhaustive'][k]['worst_value']:.3f}", _at(g["worst_k_greedy"], k), _at(g["worst_k_beam"], k)]
          for k in range(1, len(res["groups"]) + 1)]
    parts += ["Worst-k group failures: exhaustive optimum vs greedy vs beam", mk(["k", "exhaustive", "greedy", "beam"], wk)]
    if "min_set" in res:
        rows = [[m["method"], m["min_available"], "-" if m["min_value"] is None else f"{m['min_value']:.3f}"]
                for m in (res["min_set"][k] for k in res["min_set"] if isinstance(res["min_set"][k], dict) and "method" in res["min_set"][k])]
        parts += [f"Smallest available set with error <= {res['tol']} {unit}", mk(["method", "n_available", "error"], rows)]
    return "\n\n".join(parts)
