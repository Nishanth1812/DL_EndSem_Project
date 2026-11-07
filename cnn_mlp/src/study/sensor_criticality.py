#!/usr/bin/env python
"""Module 3 (v2): physics-grounded sensor criticality on a REAL trained model, built on sensor_attribution.py.

value_fn(mask) = mean EFIT-based boundary error (cm) of the frozen network over a fixed window set, with the sensors of ~mask zeroed
(zero = train mean in the standardised input = the pipeline's zero fill). For the failure-aware network the per-branch health input
is set to 0 ONLY when a whole branch is gone (that is all the model's mask can express); single dead sensors reach it as zeros.
Analysis windows come from the VALIDATION shots, the chosen sets are confirmed on disjoint TEST shots.

    run_module3(ctx, out_dir, ...)   -> criticality.json, criticality_report.md, figures/*.png
Legacy tier helpers (TIER1_CM, TIER2_CM, tier_of) are kept for old callers; the v2 tier rule is physics_study.tiers_from_criticality.
"""
import json
import math
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sensor_attribution as SA  # noqa: E402

TIER1_CM, TIER2_CM = 2.0, 1.0


def tier_of(boundary_error_cm):
    """Legacy absolute tier rule (kept for old callers)."""
    return 1 if boundary_error_cm > TIER1_CM else 2 if boundary_error_cm >= TIER2_CM else 3


def _t():
    return time.strftime("%H:%M:%S")


# ======================================================================================================== value function
class ValueFn:
    """mask (bool 115, True = available) -> dict of scalar summaries on a fixed window set; cached by mask and persisted."""

    def __init__(self, ctx, ws, truths, inv, cache_path=None, announce=None):
        """announce: how the network is told about lost sensors. "branch" = per-branch health, 0 only for fully lost branches (CNNs);
        "sensor" = per-sensor mask (Transformer); "silent" = nothing is announced, the zeros are just seen."""
        import physics_study as PS
        self.announce = announce or ("sensor" if getattr(ctx, "sensor_masks", False) else "branch")
        self.PS, self.ctx, self.ws, self.truths, self.inv = PS, ctx, ws, truths, inv
        self.yt = PS.yt_wb(ctx, ws)
        self.cache_path = cache_path
        self.cache = pickle.load(open(cache_path, "rb")) if cache_path and os.path.exists(cache_path) else {}
        self.n_new, self.seconds, self._last_save = 0, 0.0, 0

    @staticmethod
    def key(mask):
        m = np.asarray(mask, bool)
        return np.packbits(m).tobytes() + bytes([len(m) % 8])

    def _forward(self, mask):
        x = self.inv.apply_mask_to_inputs(self.ws.x, mask)
        if self.announce == "silent":
            h = None
        elif self.announce == "sensor":
            h = np.tile(np.asarray(mask, np.float32), (len(self.ws), 1))
        else:
            h = np.tile(self.inv.branch_health(mask), (len(self.ws), 1))
        return self.ctx.net.predict(x, h)

    def metrics(self, mask):
        k = self.key(mask)
        if k in self.cache:
            return self.cache[k]
        PS = self.PS
        t0 = time.time()
        yp = self._forward(mask)
        a = PS.rec_arrays(self.ctx.scorer.score(self.yt, yp, self.truths))
        a["rmse"] = PS.window_rmse(self.yt, yp)
        xp = a["xpoint_err_cm"]
        d = dict(boundary=float(np.nanmean(a["boundary_mean_cm"])), hausdorff=float(np.nanmean(a["boundary_hausdorff_cm"])),
                 boundary_median=float(np.nanmedian(a["boundary_mean_cm"])), axis=float(np.nanmean(a["axis_shift_cm"])),
                 xpoint=float(np.nanmean(xp)) if np.isfinite(xp).any() else float("nan"), xpoint_n=int(np.isfinite(xp).sum()),
                 fail_rate=float(np.nanmean(a["boundary_failed"])), nrmse=PS.nrmse_official(self.ws.sid, a["rmse"], self.ctx.sd))
        self.cache[k] = d
        self.n_new += 1
        self.seconds += time.time() - t0
        if self.cache_path and self.n_new - self._last_save >= 150:
            self.save()
        return d

    def save(self):
        if self.cache_path:
            tmp = self.cache_path + ".tmp"
            pickle.dump(self.cache, open(tmp, "wb"))
            os.replace(tmp, self.cache_path)
            self._last_save = self.n_new

    def view(self, name="boundary"):
        return lambda mask: self.metrics(mask)[name]

    def pred(self, mask):
        return self._forward(mask)


def _game(fn, inv, kind):
    ev = SA.Evaluator(fn)
    return ev, (SA.Game.groups(ev, inv) if kind == "groups" else SA.Game.sensors(ev, inv))


# ======================================================================================================== searches
def lazy_greedy_backward(game, d1_delta, batch=1, stop_above=None, on_step=None, max_reeval=None):
    """Greedy backward elimination with lazy (CELF-style) re-evaluation: stale increments are treated as lower bounds on the true
    increment, which is NOT guaranteed (damage can be super- or sub-additive), so this is an approximate greedy. Every value in the
    returned curve is a real evaluation of the failed set listed. Curve covers all levels down to 0 sensors.
    batch > 1 re-evaluates up to `batch` stale candidates per round with game.many (parallel); it only freshens more candidates than
    batch=1 would, so the choice is at least as good as the sequential one.
    stop_above: once the curve value exceeds it, the remaining levels are filled in order of the stale increments (each point is still a
    real evaluation, flagged tail=True) instead of continuing the greedy search (the min-set answers sit well below that value).
    max_reeval: at most this many candidate re-evaluations per step, then the best fresh one is taken (an extra approximation on top of
    the lazy bound: a never-re-evaluated candidate could have been better)."""
    p = game.p
    failed, cur = [], game.value(np.zeros(p, bool))
    inc = {i: float(d1_delta[i]) for i in range(p)}
    curve = [dict(n_available=p, value=cur, failed=[])]
    while len(failed) < p:
        fresh = {}
        while True:
            pool = sorted((i for i in range(p) if i not in failed and i not in fresh), key=lambda i: inc[i])
            best = min(fresh, key=fresh.get) if fresh else None
            cands = [i for i in pool[:batch] if best is None or fresh[best] > inc[i]]
            if not cands or (max_reeval is not None and fresh and len(fresh) >= max_reeval):
                break
            masks = []
            for c in cands:
                f = np.zeros(p, bool)
                f[failed + [c]] = True
                masks.append(f)
            vals = game.many(masks) if batch > 1 else [game.value(masks[0])]
            for c, v in zip(cands, vals):
                inc[c] = fresh[c] = float(v) - cur
        b = best
        failed = failed + [b]
        cur = cur + fresh[b]
        curve.append(dict(n_available=p - len(failed), value=cur, failed=sorted(failed)))
        if on_step is not None:
            on_step(len(failed))
        if stop_above is not None and cur > stop_above and len(failed) < p:
            rest = sorted((i for i in range(p) if i not in failed), key=lambda i: inc[i])
            sets = [failed + rest[:k] for k in range(1, len(rest) + 1)]
            masks = []
            for f_ in sets:
                m_ = np.zeros(p, bool)
                m_[f_] = True
                masks.append(m_)
            for f_, v in zip(sets, game.many(masks)):
                curve.append(dict(n_available=p - len(f_), value=float(v), failed=sorted(f_), tail=True))
            break
    return curve


def min_set_at(curve, tol):
    """Smallest n_available whose curve value is <= tol, and that set's failed list (None if the tolerance is never met)."""
    feas = [c for c in curve if c["value"] <= tol]
    if not feas:
        return None
    b = min(feas, key=lambda c: c["n_available"])
    return dict(n_available=b["n_available"], value=b["value"], failed=b["failed"])


def truthful_curve(game, curve):
    """Re-evaluate the failed set of every curve point (cached, so only Shapley/ranked paths cost anything)."""
    out = []
    for c in curve:
        f = np.zeros(game.p, bool)
        f[c["failed"]] = True
        out.append(dict(n_available=c["n_available"], value=game.value(f), failed=c["failed"]))
    return out


def ranked_curve(game, order_least_first, tol_unused=None):
    """Fail players in the given order WITHOUT a tolerance cut: the whole error-vs-size curve of an importance ranking."""
    failed, curve = [], []
    p = game.p
    curve.append(dict(n_available=p, value=game.value(np.zeros(p, bool)), failed=[]))
    for i in order_least_first:
        failed.append(int(i))
        f = np.zeros(p, bool)
        f[failed] = True
        curve.append(dict(n_available=p - len(failed), value=game.value(f), failed=sorted(failed)))
    return curve


# ======================================================================================================== spatial maps
def limiter_region(geo, R_axis, Z_axis):
    from matplotlib.path import Path
    lr, lz = geo.limiter
    RR, ZZ = np.meshgrid(R_axis, Z_axis)
    return Path(np.c_[lr, lz]).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)


def map_vs_group(err_map, geo, idx, region, R_axis, Z_axis):
    """Spearman(error, distance to the NEAREST sensor of the failed set) and distance error-centroid -> sensor-centroid (m)."""
    from scipy import stats
    pts = [(geo.R[i], geo.Z[i]) for i in idx if np.isfinite(geo.R[i]) and np.isfinite(geo.Z[i])]
    if not pts:
        return dict(spearman_distance=float("nan"), centroid_distance_m=float("nan"), n_positioned=0)
    RR, ZZ = np.meshgrid(R_axis, Z_axis)
    d = np.min([np.hypot(RR - r, ZZ - z) for r, z in pts], axis=0)
    ok = np.isfinite(err_map) & region
    w = err_map[ok]
    cx, cz = (w * RR[ok]).sum() / w.sum(), (w * ZZ[ok]).sum() / w.sum()
    sr, sz = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
    return dict(spearman_distance=float(stats.spearmanr(err_map[ok], d[ok])[0]), centroid_distance_m=float(math.hypot(cx - sr, cz - sz)),
                n_positioned=len(pts))


# ======================================================================================================== driver
def run_module3(ctx, out, n_eval=120, tol_abs=2.0, sensor_budget=3000, max_failed=20, worst_k=8, beam=3, seed=0, pairs=1, **kw):
    import greens_sensitivity as GS
    import physics_study as PS
    import sensor_geometry as SG
    n_eval, sensor_budget, max_failed, worst_k, beam, seed, pairs = (int(v) for v in (n_eval, sensor_budget, max_failed, worst_k, beam, seed, pairs))
    tol_abs = float(tol_abs)
    os.makedirs(out, exist_ok=True)
    figdir = os.path.join(out, "figures")
    os.makedirs(figdir, exist_ok=True)
    log = lambda *a: print(_t(), *a, flush=True)  # noqa: E731
    t_start = time.time()
    inv = SA.default_inventory()
    wsA, trA, _ = PS.pick_set(ctx, "val", n_eval, seed)
    wsB, trB, _ = PS.pick_set(ctx, "test", n_eval, seed + 1)
    log(f"analysis windows: val {len(wsA)} ({len(np.unique(wsA.sid))} shots) | confirmation: test {len(wsB)} ({len(np.unique(wsB.sid))} shots)")
    cdir = ctx.cache_dir
    mode = "sensor" if getattr(ctx, "sensor_masks", False) else "branch"
    vfA = ValueFn(ctx, wsA, trA, inv, os.path.join(cdir, f"m3_{ctx.tag}_{mode}_val_{n_eval}_{seed}.pkl"), announce=mode)
    vfB = ValueFn(ctx, wsB, trB, inv, os.path.join(cdir, f"m3_{ctx.tag}_{mode}_test_{n_eval}_{seed + 1}.pkl"), announce=mode)
    allm = np.ones(inv.n_sensors, bool)
    t0 = time.time()
    clean = vfA.metrics(allm)
    clean_B = vfB.metrics(allm)
    log(f"clean boundary (val) {clean['boundary']:.2f} cm, hausdorff {clean['hausdorff']:.2f}, xpoint {clean['xpoint']:.2f} ({clean['xpoint_n']} windows); "
        f"test {clean_B['boundary']:.2f} cm; first eval {time.time() - t0:.1f}s")
    t0 = time.time()
    vfA.metrics(~np.isin(inv.group_of, [0]))
    t_eval = time.time() - t0
    log(f"one masked evaluation = {t_eval:.2f}s on {len(wsA)} windows")

    # tolerance policy ------------------------------------------------------------------------------------------------
    floor = None
    tols = {"abs_2cm": tol_abs, "clean_x1.25": 1.25 * clean["boundary"], "clean_x1.5": 1.5 * clean["boundary"], "clean_plus_1cm": clean["boundary"] + 1.0}
    primary = "clean_x1.25" if clean["boundary"] >= 0.8 * tol_abs else "abs_2cm"
    tol = tols[primary]
    log("tolerances", {k: round(v, 2) for k, v in tols.items()}, "-> primary", primary, round(tol, 2))

    res = dict(model=ctx.tag, n_eval_windows=len(wsA), n_eval_shots=int(len(np.unique(wsA.sid))), n_confirm_windows=len(wsB),
               n_confirm_shots=int(len(np.unique(wsB.sid))), clean_val=clean, clean_test=clean_B, tolerances=tols, primary_tolerance=primary,
               tolerance_note=("The model's own clean boundary error already exceeds the 2 cm design tolerance, so 'abs_2cm' is infeasible even with "
                               "every sensor; the headline set uses a tolerance RELATIVE to the clean error (primary_tolerance)."
                               if clean["boundary"] >= tol_abs else "clean error is below the absolute tolerance"),
               value="mean over windows of boundary_mean_cm (EFIT-mode, 50 cm when no closed contour); X-point = mean xpoint_err_cm over windows where defined",
               failure_aware_note=("per-sensor mask: a lost sensor is ANNOUNCED and dropped from attention (Transformer); the silent comparison is in "
                                   "silent_vs_announced" if mode == "sensor" else
                                   "health input = 0 only for fully lost branches; single dead sensors enter as zeros (unmasked)"),
               announce_mode=mode, backbone=getattr(ctx, "backbone", "?"),
               device=PS.CPU_NOTE % (ctx.torch_version, ctx.torch_threads), seconds_per_eval=t_eval)

    # 1. groups (exact, 256 coalitions) on val, and again on test -----------------------------------------------------
    evG, gg = _game(vfA.view("boundary"), inv, "groups")
    vals = SA.full_table(gg)
    phi_g = SA.exact_shapley(vals, inv.n_groups)
    I_g = SA.exact_interaction(vals, inv.n_groups)
    d1g = SA.drop_one(gg)
    alone_g = SA.alone_value(gg)
    pop = np.array([bin(b).count("1") for b in range(256)])
    xp_vals = np.array([vfA.metrics(inv_mask(inv, b))["xpoint"] for b in range(256)])
    xp_phi = SA.exact_shapley(np.nan_to_num(xp_vals, nan=np.nanmax(xp_vals)), inv.n_groups)
    evGB, ggB = _game(vfB.view("boundary"), inv, "groups")
    valsB = SA.full_table(ggB)
    phi_gB = SA.exact_shapley(valsB, inv.n_groups)
    from scipy import stats
    res["group"] = dict(names=inv.group_names, counts=inv.counts, shapley=phi_g, shapley_test=phi_gB, shapley_xpoint=xp_phi,
                        drop_one=d1g["delta"], drop_one_test=SA.drop_one(ggB)["delta"], alone=alone_g, interaction=I_g,
                        ranking_shapley=SA.ranking(phi_g, inv.group_names), table_boundary_val=vals, table_boundary_test=valsB,
                        table_xpoint_val=xp_vals, val_test_shapley_spearman=float(stats.spearmanr(phi_g, phi_gB)[0]),
                        efficiency_check=dict(sum_shapley=float(phi_g.sum()), f_all_minus_f_none=float(vals[-1] - vals[0])),
                        worst_k_exhaustive=SA.exhaustive_group_curve(vals, inv.n_groups), worst_k_greedy=SA.worst_k_beam(gg, 8, 1),
                        worst_k_beam=SA.worst_k_beam(gg, 8, beam))
    log("groups done; Shapley (cm):", {n: round(float(v), 2) for n, v in zip(inv.group_names, phi_g)}, f"| {vfA.n_new + vfB.n_new} evals, {vfA.seconds + vfB.seconds:.0f}s")
    vfA.save(); vfB.save()
    if int(kw.get("tiers_only", 0)):     # cheap pass: only what Module 4's tier rule needs (group drop-one with the model told)
        path = os.path.join(out, "criticality_tiers.json")
        json.dump(SA.jsonable(dict(res, note="tiers_only run: groups only")), open(path, "w"), indent=1)
        log("wrote", path)
        return res

    # 1b. Transformer only: SILENT loss (zeros seen, nothing announced) vs ANNOUNCED loss (per-sensor mask) -------------------------
    if mode == "sensor" and int(kw.get("silent", 1)):
        vfS = ValueFn(ctx, wsA, trA, inv, os.path.join(cdir, f"m3_{ctx.tag}_silent_val_{n_eval}_{seed}.pkl"), announce="silent")
        vfSB = ValueFn(ctx, wsB, trB, inv, os.path.join(cdir, f"m3_{ctx.tag}_silent_test_{n_eval}_{seed + 1}.pkl"), announce="silent")
        _, ggS = _game(vfS.view("boundary"), inv, "groups")
        valsS = SA.full_table(ggS)
        d1gS = SA.drop_one(ggS)["delta"]
        _, gsS = _game(vfS.view("boundary"), inv, "sensors")
        d1S = SA.drop_one(gsS)["delta"]
        _, gsA_ = _game(vfA.view("boundary"), inv, "sensors")
        d1A = SA.drop_one(gsA_)["delta"]
        d1S_B = [vfSB.metrics(inv.mask_from_failed_sensors([i]))["boundary"] - vfSB.metrics(allm)["boundary"] for i in range(inv.n_sensors)]
        d1A_B = [vfB.metrics(inv.mask_from_failed_sensors([i]))["boundary"] - clean_B["boundary"] for i in range(inv.n_sensors)]
        grp_B = {}
        for g_ in range(inv.n_groups):
            m_ = ~np.isin(inv.group_of, [g_])
            grp_B[inv.group_names[g_]] = dict(silent=vfSB.metrics(m_)["boundary"] - vfSB.metrics(allm)["boundary"], announced=vfB.metrics(m_)["boundary"] - clean_B["boundary"])
        res["silent_vs_announced"] = dict(
            clean_silent_val=vfS.metrics(allm)["boundary"], clean_announced_val=clean["boundary"],
            group_names=inv.group_names, group_shapley_silent=SA.exact_shapley(valsS, inv.n_groups), group_shapley_announced=phi_g,
            group_drop_one_silent=d1gS, group_drop_one_announced=d1g["delta"], group_table_silent=valsS,
            group_drop_one_test=grp_B,
            sensor_drop_one_silent=d1S, sensor_drop_one_announced=d1A, sensor_drop_one_silent_test=d1S_B, sensor_drop_one_announced_test=d1A_B,
            per_group_mean_drop_one=[dict(group=inv.group_names[g_], silent=float(np.mean(np.array(d1S)[inv.indices(g_)])),
                                          announced=float(np.mean(np.array(d1A)[inv.indices(g_)]))) for g_ in range(inv.n_groups)],
            worst_silent_sensors=SA.ranking(d1S, inv.sensor_names)[:10], worst_announced_sensors=SA.ranking(d1A, inv.sensor_names)[:10],
            n_sensors_silent_ge_1cm=int((np.array(d1S) >= 1.0).sum()), n_sensors_announced_ge_1cm=int((np.array(d1A) >= 1.0).sum()),
            n_sensors_silent_ge_2cm=int((np.array(d1S) >= 2.0).sum()), n_sensors_announced_ge_2cm=int((np.array(d1A) >= 2.0).sum()),
            spearman_silent_vs_announced_dropone=float(stats.spearmanr(d1S, d1A)[0]))
        sv = res["silent_vs_announced"]
        log(f"silent vs announced: clean {sv['clean_silent_val']:.2f}/{sv['clean_announced_val']:.2f} cm; sensors with drop-one >= 1 cm: silent "
            f"{sv['n_sensors_silent_ge_1cm']} vs announced {sv['n_sensors_announced_ge_1cm']}; worst silent {sv['worst_silent_sensors'][0]['name']} "
            f"{sv['worst_silent_sensors'][0]['score']:.1f} cm; groups drop-one silent {np.round(d1gS, 2).tolist()} announced {np.round(d1g['delta'], 2).tolist()}")
        vfS.save(); vfSB.save()

    # 2. sensors: drop-one, sampled Shapley with CI -------------------------------------------------------------------
    evS, gs = _game(vfA.view("boundary"), inv, "sensors")
    d1 = SA.drop_one(gs)
    xp_d1 = np.array([vfA.metrics(inv.mask_from_failed_sensors([i]))["xpoint"] for i in range(inv.n_sensors)]) - clean["xpoint"]
    log(f"sensor drop-one done ({vfA.n_new} new evals so far); max delta {np.nanmax(d1['delta']):.2f} cm")
    rng = np.random.default_rng(seed)
    sh = SA.sampled_shapley(gs, rng, max_pairs=10 ** 6, budget_evals=evS.n_evals + sensor_budget, max_failed=max_failed)
    vfA.save()
    headroom = max(tol - clean["boundary"], 0.25)
    cls = SA.classify_sensors(d1["delta"], sh["phi"], 0.5 * headroom, 0.05 * headroom)
    z = 1.96 * sh["se"]
    res["sensor"] = dict(names=inv.sensor_names, group_of=inv.group_of, drop_one=d1["delta"], drop_one_xpoint=xp_d1, shapley=sh["phi"],
                         shapley_se=sh["se"], shapley_ci95=z, shapley_pairs=sh["n_pairs"], shapley_max_failed=sh["max_failed"],
                         shapley_stop=sh["stop"], classes=cls, class_thresholds=dict(critical=0.5 * headroom, redundant=0.05 * headroom),
                         ranking_drop_one=SA.ranking(d1["delta"], inv.sensor_names), ranking_shapley=SA.ranking(sh["phi"], inv.sensor_names),
                         shapley_group_sum=[float(sh["phi"][inv.indices(g)].sum()) for g in range(inv.n_groups)],
                         n_distinguishable_top=int(np.sum(np.sort(np.abs(sh['phi']))[::-1][:10] > z[np.argsort(-np.abs(sh['phi']))[:10]])))
    log(f"sensor Shapley: {sh['n_pairs']} antithetic pairs, K={sh['max_failed']}, stop={sh['stop']}, {evS.n_evals} unique evals; classes",
        {c: cls.count(c) for c in set(cls)})

    # interactions among the top sensors (sampled)
    if pairs:
        top = list(np.argsort(-np.abs(sh["phi"]))[:6])
        inter = []
        for a_ in range(len(top)):
            for b_ in range(a_ + 1, len(top)):
                m, se = SA.sampled_interaction(gs, int(top[a_]), int(top[b_]), rng, n_samples=8, max_failed=max_failed)
                inter.append(dict(i=int(top[a_]), j=int(top[b_]), names=[inv.sensor_names[top[a_]], inv.sensor_names[top[b_]]], mean=m, se=se,
                                  kind="backup/redundant" if m - 1.96 * se > 0 else "overlap" if m + 1.96 * se < 0 else "not distinguishable"))
        res["sensor"]["interactions_top6"] = inter
        vfA.save()

    # 3. worst-k failure sets --------------------------------------------------------------------------------------------
    wg = SA.worst_k_beam(gs, worst_k, 1)
    log("sensor worst-k greedy:", {k: round(v["value"], 2) for k, v in wg.items()})
    wb = SA.worst_k_beam(gs, worst_k, beam)
    log(f"sensor worst-k beam{beam}:", {k: round(v["value"], 2) for k, v in wb.items()}, f"| {vfA.n_new} new evals, {vfA.seconds:.0f}s")
    rnd = {}
    for k in range(1, worst_k + 1):   # random-k baseline: 20 random k-sets
        v = gs.many([np.isin(np.arange(inv.n_sensors), rng.choice(inv.n_sensors, k, replace=False)) for _ in range(12)])
        rnd[k] = dict(mean=float(v.mean()), p90=float(np.percentile(v, 90)))
    res["sensor"]["worst_k_greedy"], res["sensor"]["worst_k_beam"], res["sensor"]["random_k"] = wg, wb, rnd
    res["sensor"]["worst_k_beam_width"] = beam
    for w_ in (wg, wb):
        for k, v in w_.items():
            v["names"] = [inv.sensor_names[i] for i in v["set"]]
            v["value_test"] = float(vfB.view("boundary")(inv.mask_from_failed_sensors(v["set"])))
    vfA.save(); vfB.save()

    # 4. minimum sensor set -----------------------------------------------------------------------------------------------
    gcurve = SA.greedy_backward(gg, tol)["curve"]
    lazy = lazy_greedy_backward(gs, d1["delta"], batch=int(kw.get("lazy_batch", 1)), stop_above=2.0 * max(tols.values()),
                                max_reeval=(int(kw["max_reeval"]) if "max_reeval" in kw else None),
                                on_step=lambda k: (vfA.save() if k % 5 == 0 else None))
    ranked_sh = ranked_curve(gs, np.argsort(sh["phi"]))
    ranked_d1 = ranked_curve(gs, np.argsort(np.nan_to_num(d1["delta"])))
    log(f"minimum-set curves done ({vfA.n_new} new evals, {vfA.seconds:.0f}s)")
    ms = dict(groups_greedy=gcurve, groups_exhaustive=res["group"]["worst_k_exhaustive"], sensors_lazy_greedy=lazy,
              sensors_shapley_ranked=ranked_sh, sensors_dropone_ranked=ranked_d1, per_tolerance={})
    for name, tv in tols.items():
        entry = {}
        for cname, curve in (("lazy_greedy", lazy), ("shapley_ranked", ranked_sh), ("dropone_ranked", ranked_d1), ("groups_greedy", gcurve)):
            m = min_set_at(curve, tv)
            if m is not None and cname != "groups_greedy":   # confirm on the disjoint test shots
                m["value_test"] = float(vfB.view("boundary")(inv.mask_from_failed_sensors(m["failed"])))
                m["kept"] = [inv.sensor_names[i] for i in range(inv.n_sensors) if i not in m["failed"]]
                m["kept_per_group"] = {inv.group_names[g]: int(sum(1 for i in inv.indices(g) if i not in m["failed"])) for g in range(inv.n_groups)}
                m["failed"] = [int(i) for i in m["failed"]]
            entry[cname] = m
        ex = SA.exhaustive_group_curve(vals, inv.n_groups)
        feas = [c for c in ex if c["best_value"] <= tv]
        entry["groups_exhaustive_best"] = min(feas, key=lambda c: c["n_available"]) if feas else None
        ms["per_tolerance"][name] = dict(tol=tv, **entry)
    ms["clean_test_reference"] = clean_B["boundary"]
    res["min_set"] = ms
    for name, e in ms["per_tolerance"].items():
        lz = e["lazy_greedy"]
        lz_detail = "" if not lz else f" (val {lz['value']:.2f}, test {lz['value_test']:.2f} cm)"
        log(f"min set @ {name} ({e['tol']:.2f} cm): lazy-greedy {lz['n_available'] if lz else None} sensors"
            f"{lz_detail}; shapley-ranked "
            f"{e['shapley_ranked']['n_available'] if e['shapley_ranked'] else None}")
    vfA.save(); vfB.save()

    # 5. Green's-function sensitivity (vacuum linear-Gaussian model) vs the data-driven rankings ---------------------------
    geo = SG.load_geometry()
    gpath = os.path.join(cdir, f"greens_cache_{max_failed}_{seed}.pkl")     # model independent: computed once, shared by every checkpoint
    if os.path.exists(gpath):
        gm_has, gsc, gphi_g, gdrop_g, gsh = pickle.load(open(gpath, "rb"))
    else:
        gm = GS.build_model(geo)
        gsc = gm.scores()
        evg = SA.Evaluator(gm.value_fn)
        ggg, gsg = SA.Game.groups(evg, inv), SA.Game.sensors(evg, inv)
        gvals = SA.full_table(ggg)
        gphi_g = SA.exact_shapley(gvals, inv.n_groups)
        gdrop_g = SA.drop_one(ggg)["delta"]
        gsh = SA.sampled_shapley(gsg, np.random.default_rng(seed + 5), max_pairs=10 ** 6, budget_evals=4000, max_failed=max_failed)
        gm_has = gm.has_model
        pickle.dump((gm_has, gsc, gphi_g, gdrop_g, gsh), open(gpath, "wb"))
    glabels = [inv.group_names[g] for g in inv.group_of]
    res["greens"] = dict(
        model="vacuum Green's functions, linear-Gaussian information model (greens_sensitivity.build_model defaults); saddle loops unmodelled "
              "(voltage sensors), Ip = Rogowski sum; modelling choices (prior, noise, kernel) are parameters, see MODULE3_V2.md",
        has_model=gm_has, loo_loss=gsc["loo_loss"], alone_gain=gsc["alone_gain"], noise_prop=gsc["noise_prop"], shapley=gsh["phi"],
        group_shapley=gphi_g, group_drop_one=gdrop_g,
        agreement=dict(
            dropone_vs_loo=GS.rank_agreement_by_group(d1["delta"], gsc["loo_loss"], glabels, n_perm=2000),
            shapley_vs_loo=GS.rank_agreement_by_group(sh["phi"], gsc["loo_loss"], glabels, n_perm=2000),
            shapley_vs_green_shapley=GS.rank_agreement_by_group(sh["phi"], gsh["phi"], glabels, n_perm=2000),
            group_shapley=GS.rank_agreement(phi_g, gphi_g, n_perm=5000),
            group_dropone=GS.rank_agreement(d1g["delta"], gdrop_g, n_perm=5000)))
    ag = res["greens"]["agreement"]
    log("Green vs data: sensor Shapley~loo rho", round(ag["shapley_vs_loo"]["all"]["spearman"], 3), "tau", round(ag["shapley_vs_loo"]["all"]["kendall"], 3),
        "| groups rho", round(ag["group_shapley"]["spearman"], 3))
    # Green-guided minimal set evaluated on the REAL model (least informative sensors dropped first)
    order_g = np.argsort(np.nan_to_num(gsc["loo_loss"], nan=-1.0))
    gcurve_real = ranked_curve(gs, order_g)
    res["min_set"]["sensors_green_ranked"] = gcurve_real
    res["min_set"]["per_tolerance_green"] = {name: min_set_at(gcurve_real, tv) for name, tv in tols.items()}
    vfA.save()

    # 6. spatial error maps vs sensor location ------------------------------------------------------------------------------
    R_axis, Z_axis = np.linspace(0.06, 1.98, 65), np.linspace(-2.0, 2.0, 65)
    region = limiter_region(geo, R_axis, Z_axis)
    yt = vfA.yt
    yp0 = vfA.pred(allm)
    base_err = np.abs(yp0 - yt).mean(0)
    cases = []
    gorder = np.argsort(-d1g["delta"])
    for g in gorder[:3]:
        cases.append(("group", f"{inv.group_names[g]} lost", list(inv.indices(g))))
    sorder = [int(i) for i in np.argsort(-np.nan_to_num(d1["delta"]))[:4]]
    for i in sorder:
        cases.append(("sensor", f"{inv.sensor_names[i]} lost", [i]))
    k3 = wb[min(3, worst_k)]["set"]
    cases.append(("worst3", f"worst-3 sensors ({', '.join(inv.sensor_names[i] for i in k3)})", list(k3)))
    maps = []
    for kind, label, idx in cases:
        m = ~np.isin(np.arange(inv.n_sensors), idx)
        yp = vfA.pred(m)
        e_fail = np.abs(yp - yt).mean(0)
        delta = np.abs(yp - yp0).mean(0)
        st = map_vs_group(delta, geo, idx, region, R_axis, Z_axis)
        if len(idx) == 1:
            i = idx[0]
            st.update(GS.error_map_vs_location(delta, geo.R[i], geo.Z[i], R_axis, Z_axis, region, GS.sensor_response_map(geo, i)))
        st_err = map_vs_group(np.maximum(e_fail - base_err, 0), geo, idx, region, R_axis, Z_axis)
        maps.append(dict(kind=kind, label=label, sensors=[int(i) for i in idx], delta_map=delta, err_map=e_fail, stats_delta_vs_clean_prediction=st,
                         stats_excess_error_vs_clean=st_err, boundary_cm=vfA.metrics(m)["boundary"]))
    res["spatial"] = dict(R_axis=R_axis, Z_axis=Z_axis, base_err_map=base_err, cases=maps,
                          note="delta_map = mean |psi_pred(lost) - psi_pred(clean)| over the analysis windows (Wb/rad); stats are Spearman(map, distance "
                               "to the nearest lost sensor) over pixels inside the limiter (negative = effect concentrated near the sensor) and the distance "
                               "between the map's centroid and the sensor(s).")
    res["cost"] = dict(unique_evals_val=len(vfA.cache), unique_evals_test=len(vfB.cache), seconds=time.time() - t_start,
                       evals_new_this_run=vfA.n_new + vfB.n_new)
    vfA.save(); vfB.save()
    _figures(res, inv, geo, figdir, tol, primary, region)
    path = os.path.join(out, "criticality.json")
    np.savez_compressed(os.path.join(out, "spatial_maps.npz"), base_err=base_err,
                        **{f"case{i}_{k}": c.pop(k).astype(np.float32) for i, c in enumerate(res["spatial"]["cases"]) for k in ("delta_map", "err_map")})
    res["spatial"].pop("base_err_map")
    json.dump(SA.jsonable(res), open(path, "w"), indent=1)
    _report(res, inv, os.path.join(out, "criticality_report.md"))
    log("wrote", path)
    return res


def inv_mask(inv, bits):
    """Availability mask of the group coalition whose failed groups are the set bits of `bits`."""
    return ~np.isin(inv.group_of, [g for g in range(inv.n_groups) if (bits >> g) & 1])


# ======================================================================================================== outputs
def _figures(res, inv, geo, figdir, tol, primary, region):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = plt.get_cmap("tab10")
    # groups
    g = res["group"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(inv.n_groups)
    ax[0].bar(x - 0.2, g["drop_one"], 0.4, label="drop-one")
    ax[0].bar(x + 0.2, g["shapley"], 0.4, label="Shapley (val)")
    ax[0].plot(x + 0.2, g["shapley_test"], "k_", ms=14, label="Shapley (test)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(inv.group_names, rotation=40, ha="right"); ax[0].set_ylabel("boundary error added (cm)")
    ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("branch (group) importance, exact over 256 coalitions")
    im = ax[1].imshow(np.array(g["interaction"]), cmap="coolwarm", vmin=-np.abs(g["interaction"]).max(), vmax=np.abs(g["interaction"]).max())
    ax[1].set_xticks(x); ax[1].set_yticks(x); ax[1].set_xticklabels(inv.group_names, rotation=60, ha="right", fontsize=8); ax[1].set_yticklabels(inv.group_names, fontsize=8)
    ax[1].set_title("pair interaction (+ backup, - overlap), cm"); fig.colorbar(im, ax=ax[1]); fig.tight_layout()
    fig.savefig(os.path.join(figdir, "group_shapley.png"), dpi=140); plt.close(fig)
    # sensor shapley top 25 with CI
    s = res["sensor"]
    order = np.argsort(-np.abs(s["shapley"]))[:25][::-1]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.barh(range(len(order)), np.array(s["shapley"])[order], xerr=np.array(s["shapley_ci95"])[order], color=[cm(int(inv.group_of[i])) for i in order])
    ax.set_yticks(range(len(order))); ax.set_yticklabels([inv.sensor_names[i] for i in order], fontsize=7)
    ax.set_xlabel(f"sampled Shapley damage (cm), K<={s['shapley_max_failed']}, {s['shapley_pairs']} antithetic pairs, 95% CI"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "sensor_shapley_top25.png"), dpi=140); plt.close(fig)
    # worst-k
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ks = list(range(1, len(s["worst_k_greedy"]) + 1))
    ax.plot(ks, [s["worst_k_greedy"][str(k) if str(k) in s["worst_k_greedy"] else k]["value"] for k in ks], "o-", label="sensors, greedy")
    ax.plot(ks, [s["worst_k_beam"][str(k) if str(k) in s["worst_k_beam"] else k]["value"] for k in ks], "s-", label=f"sensors, beam {s['worst_k_beam_width']}")
    ax.plot(ks, [s["random_k"][str(k) if str(k) in s["random_k"] else k]["mean"] for k in ks], "k--", label="random k sensors (mean)")
    gx = res["group"]["worst_k_exhaustive"]
    ax.plot([c["n_failed"] for c in gx[1:9]], [c["worst_value"] for c in gx[1:9]], "^-", label="GROUPS, exhaustive")
    ax.axhline(res["clean_val"]["boundary"], c="gray", ls=":", label="clean")
    ax.set_xlabel("k simultaneous failures"); ax.set_ylabel("mean boundary error (cm)"); ax.set_yscale("log"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "worst_k.png"), dpi=140); plt.close(fig)
    # min-set curves
    ms = res["min_set"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, lab in (("sensors_lazy_greedy", "lazy greedy"), ("sensors_shapley_ranked", "Shapley-ranked"), ("sensors_dropone_ranked", "drop-one-ranked"),
                     ("sensors_green_ranked", "Green's-ranked")):
        c = ms[key]
        ax.plot([p["n_available"] for p in c], [p["value"] for p in c], label=lab)
    for name, tv in res["tolerances"].items():
        ax.axhline(tv, ls="--", lw=.8, c="r" if name == primary else "gray")
        ax.text(1, tv, name, fontsize=7, va="bottom")
    ax.set_xlabel("sensors kept (of 115)"); ax.set_ylabel("mean boundary error (cm)"); ax.set_yscale("log"); ax.invert_xaxis(); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "minimal_sensor_set_curve.png"), dpi=140); plt.close(fig)
    # data vs Green
    gr = res["greens"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a_, (xv, yv, xl) in zip(ax, ((s["drop_one"], gr["loo_loss"], "data drop-one (cm)"), (s["shapley"], gr["loo_loss"], "data Shapley (cm)"))):
        for gi in range(inv.n_groups):
            m = inv.group_of == gi
            a_.scatter(np.array(xv)[m], np.array(yv, float)[m], s=14, color=cm(gi), label=inv.group_names[gi])
        a_.set_xlabel(xl); a_.set_ylabel("Green's leave-one-out loss"); a_.set_yscale("symlog", linthresh=1e-6); a_.set_xscale("symlog", linthresh=1e-2); a_.grid(alpha=.3)
    ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(figdir, "data_vs_greens.png"), dpi=140); plt.close(fig)
    if "silent_vs_announced" in res:
        sv = res["silent_vs_announced"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        x = np.arange(inv.n_groups)
        ax[0].bar(x - 0.2, sv["group_drop_one_silent"], 0.4, label="silent (zeros seen)")
        ax[0].bar(x + 0.2, sv["group_drop_one_announced"], 0.4, label="announced (mask)")
        ax[0].set_xticks(x); ax[0].set_xticklabels(inv.group_names, rotation=40, ha="right"); ax[0].set_ylabel("boundary error added by losing the whole group (cm)")
        ax[0].legend(); ax[0].grid(alpha=.3)
        for gi in range(inv.n_groups):
            m = inv.group_of == gi
            ax[1].scatter(np.array(sv["sensor_drop_one_announced"])[m], np.array(sv["sensor_drop_one_silent"])[m], s=14, color=cm(gi), label=inv.group_names[gi])
        lim = max(np.nanmax(sv["sensor_drop_one_silent"]), np.nanmax(sv["sensor_drop_one_announced"]))
        ax[1].plot([0, lim], [0, lim], "k:", lw=.8); ax[1].set_xlabel("single sensor lost, ANNOUNCED (cm)"); ax[1].set_ylabel("single sensor lost, SILENT (cm)")
        ax[1].legend(fontsize=6); ax[1].grid(alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(figdir, "silent_vs_announced.png"), dpi=140); plt.close(fig)
    # spatial maps with sensor positions
    sp = res["spatial"]
    cases = sp["cases"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.2))
    lr, lz = geo.limiter
    for a_, c in zip(axes.ravel(), cases):
        a_.imshow(np.array(c["delta_map"]), origin="lower", extent=[0.06, 1.98, -2, 2], cmap="magma", aspect="auto")
        a_.plot(lr, lz, "w-", lw=.6)
        idx = c["sensors"]
        a_.scatter(geo.R[idx], geo.Z[idx], s=26 if len(idx) < 20 else 10, c="cyan", edgecolors="k", linewidths=.4)
        st = c["stats_delta_vs_clean_prediction"]
        a_.set_title(f"{c['label'][:44]}\nbnd {c['boundary_cm']:.1f} cm, rho(dist) {st['spearman_distance']:.2f}", fontsize=8)
    for a_ in axes.ravel()[len(cases):]:
        a_.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "spatial_error_vs_sensor_location.png"), dpi=130); plt.close(fig)


def _report(res, inv, path):
    g, s, ms = res["group"], res["sensor"], res["min_set"]
    L = ["# Module 3 (v2) on the real model: " + res["model"], "",
         f"Value = mean EFIT-mode boundary error (cm). Analysis: {res['n_eval_windows']} val windows / {res['n_eval_shots']} shots; confirmation: "
         f"{res['n_confirm_windows']} test windows / {res['n_confirm_shots']} shots. Device: {res['device']}.",
         f"Clean boundary error: val {res['clean_val']['boundary']:.2f} cm, test {res['clean_test']['boundary']:.2f} cm. {res['tolerance_note']}", "",
         "## Groups (exact Shapley, 256 coalitions)", "", "| group | drop-one (cm) | Shapley val (cm) | Shapley test (cm) | Shapley X-point (cm) |", "|---|---|---|---|---|"]
    for i in np.argsort(-np.array(g["shapley"])):
        L.append(f"| {inv.group_names[i]} | {g['drop_one'][i]:.2f} | {g['shapley'][i]:.2f} | {g['shapley_test'][i]:.2f} | {g['shapley_xpoint'][i]:.2f} |")
    if "silent_vs_announced" in res:
        sv = res["silent_vs_announced"]
        L += ["", "## Silent vs announced loss (per-sensor mask model)", "",
              f"clean val boundary: silent-mode run {sv['clean_silent_val']:.2f} cm, announced-mode run {sv['clean_announced_val']:.2f} cm (identical when no sensor is lost). "
              f"Sensors whose single loss adds >= 1 cm: silent {sv['n_sensors_silent_ge_1cm']}, announced {sv['n_sensors_announced_ge_1cm']} (>= 2 cm: "
              f"{sv['n_sensors_silent_ge_2cm']} vs {sv['n_sensors_announced_ge_2cm']}).", "",
              "| group | whole-group loss silent (cm) | announced (cm) | mean single-sensor loss silent | announced |", "|---|---|---|---|---|"]
        for i, n in enumerate(inv.group_names):
            pg = sv["per_group_mean_drop_one"][i]
            L.append(f"| {n} | {sv['group_drop_one_silent'][i]:.2f} | {sv['group_drop_one_announced'][i]:.2f} | {pg['silent']:.2f} | {pg['announced']:.2f} |")
    L += ["", f"Val/test Shapley Spearman: {g['val_test_shapley_spearman']:.2f}.", "", "## Top sensors (sampled Shapley, 95% CI)", "",
          "| sensor | class | drop-one (cm) | Shapley (cm) | +-CI |", "|---|---|---|---|---|"]
    for i in np.argsort(-np.abs(s["shapley"]))[:15]:
        L.append(f"| {inv.sensor_names[i]} | {s['classes'][i]} | {s['drop_one'][i]:.2f} | {s['shapley'][i]:.2f} | {s['shapley_ci95'][i]:.2f} |")
    L += ["", "## Minimum sensor sets (val-selected, test-confirmed)", "", "| tolerance | cm | lazy greedy | Shapley-ranked | drop-one-ranked | groups best |", "|---|---|---|---|---|---|"]
    for name, e in ms["per_tolerance"].items():
        f = lambda m: "infeasible" if m is None else f"{m['n_available']} (val {m['value']:.2f} / test {m.get('value_test', float('nan')):.2f})"  # noqa: E731
        gb = e["groups_exhaustive_best"]
        L.append(f"| {name} | {e['tol']:.2f} | {f(e['lazy_greedy'])} | {f(e['shapley_ranked'])} | {f(e['dropone_ranked'])} | "
                 f"{'infeasible' if gb is None else str(gb['n_available']) + ' groups'} |")
    gr = res["greens"]["agreement"]
    L += ["", "## Green's-function sensitivity vs data-driven ranking", "",
          f"sensors, Shapley vs Green's LOO: Spearman {gr['shapley_vs_loo']['all']['spearman']:.2f} (p {gr['shapley_vs_loo']['all']['p_spearman']:.3f}), "
          f"Kendall {gr['shapley_vs_loo']['all']['kendall']:.2f}; drop-one vs LOO: Spearman {gr['dropone_vs_loo']['all']['spearman']:.2f}; "
          f"groups (n=8): Spearman {gr['group_shapley']['spearman']:.2f}.", "", "Figures: figures/*.png. Full numbers: criticality.json."]
    open(path, "w", encoding="utf8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    print(__doc__)
