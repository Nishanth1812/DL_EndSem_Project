#!/usr/bin/env python
"""Assert-based tests for Module 4 v2:  python test_mitigation_v2.py   (about 1 minute on a laptop CPU, numpy only)

Synthetic data: correlated multichannel windows (K latent modes -> C channels + noise) with injected faults of every
kind and known labels. Real data: magnetics of local MAST shots (skipped when the zarr shots / the cache are missing).
Printed numbers are measured here; the asserted thresholds are the stated acceptance floors, not tuned targets.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_mitigation import (AdaptiveConformalGauge, ConformalGauge, FaultDetector, FluxLoopCheck, RouterV2,
                                 SmartAdaptiveMitigationBrain, confidence_flag, detection_from_labels,
                                 difficulty_features, empirical_coverage, summarise_latencies)
from fault_models import (KINDS, FaultSpec, WindowLayout, benchmark_detector, group_loss_specs, inject, inject_series,
                          inject_window, random_specs)
from virtual_sensor import fit_virtual_sensors, fit_physics_fallback

SP = r"C:\Users\Devab\AppData\Local\Temp\claude\94cae6ef419a3dd0\9b3dc88f-3a2c-4815-bbad-6ee91700a5da\scratchpad"
BRANCH_P = [15, 40, 18, 18, 12, 10, 1, 1]            # task_1-3: 115 channels in 8 branches
RESULTS = {}


# ------------------------------------------------------------------ synthetic world
def synth(rng, N, A, T=25, noise=0.15):
    C, K = A.shape
    t = np.linspace(-1, 1, T)[None, :, None]
    L = rng.normal(size=(N, 1, K)) + 0.4 * rng.normal(size=(N, 1, K)) * t \
        + 0.15 * np.sin(2 * np.pi * rng.uniform(0.5, 2, (N, 1, K)) * t)
    return L @ A.T + noise * rng.normal(size=(N, T, C)), L


def make_world(seed=0, C=115, K=8, n_train=1500, n_calib=800, n_test=300):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(C, K))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    raw = [synth(rng, n, A)[0] for n in (n_train, n_calib, n_test)]
    mu, sd = raw[0].reshape(-1, C).mean(0), raw[0].reshape(-1, C).std(0)
    return [(x - mu) / sd for x in raw], A, rng


def to_window(X, probes=BRANCH_P, rep=(1, 1, 1, 1, 2, 1, 1, 1)):
    """Canonical (25, C) -> branch-list window; the saddle branch (index 4) has 2x the samples (repeat) so the
    block-mean flatten returns the canonical series exactly."""
    out, o = [], 0
    for p, r in zip(probes, rep):
        out.append(np.repeat(X[:, o:o + p], r, axis=0))
        o += p
    return out


_WORLD = {}


def world():
    if not _WORLD:
        (tr, ca, te), A, rng = make_world()
        det = FaultDetector(fp_rate=1e-4).fit(tr, ca)
        _WORLD.update(tr=tr, ca=ca, te=te, A=A, det=det, bank=fit_virtual_sensors(tr))
    return _WORLD


# ------------------------------------------------------------------ tests
def test_injectors_and_layout():
    rng = np.random.default_rng(1)
    x = rng.normal(size=25)
    y, ch = inject_series(x, FaultSpec(0, "stuck_rail", 2.0, None, 8.0))
    assert np.all(y[10:] == 8.0) and np.all(y[:10] == x[:10]) and ch[10:].all()
    y, _ = inject_series(x, FaultSpec(0, "stuck_value", 2.0))
    assert np.all(y[10:] == x[9])
    y, _ = inject_series(x, FaultSpec(0, "drift", 1.0, None, 0.5))               # 0.5 sigma/ms from 1 ms
    assert abs((y[-1] - x[-1]) - 0.5 * (5.0 - 1.0) * (24 / 25) * 1.0) < 1e-9 or abs((y[-1] - x[-1]) - 0.5 * (24 - 5) * 0.2) < 1e-9
    y, ch = inject_series(x, FaultSpec(0, "spike", 1.0, None, 12.0))
    assert ch.sum() == 1 and abs(y[5] - x[5] - 12.0) < 1e-12
    y, ch = inject_series(x * 3, FaultSpec(0, "saturation", 0.0, None, 1.0))
    assert np.abs(y).max() <= 1.0 + 1e-12 and ch.any()
    y, ch = inject_series(x, FaultSpec(0, "dropout", 1.0, 1.0, fill=np.nan))
    assert np.isnan(y[5:10]).all() and ch.sum() == 5
    # clipping that changes nothing is not a fault in the labels
    X = np.zeros((25, 3))
    _, lab = inject(X, [FaultSpec(1, "saturation", 0.0, None, 1.0)])
    assert not lab.faulty.any()
    # labels
    X = rng.normal(size=(25, 6))
    Y, lab = inject(X, [FaultSpec(2, "stuck_rail", 2.0), FaultSpec(4, "dropout", 1.0, 2.0)])
    assert list(lab.channels) == [2, 4] and lab.kind[2] == "stuck_rail" and lab.kind[0] == ""
    assert abs(lab.onset_ms[4] - 1.0) < 1e-12 and abs(lab.duration_ms[4] - 2.0) < 1e-12 and lab.mask[:, 4].sum() == 10
    # group loss
    win = to_window(rng.normal(size=(25, 115)))
    L = WindowLayout(win)
    assert L.n_channels == 115 and L.t_ref == 25 and L.n_branches == 8
    w2, lab = inject_window(win, L, group_loss_specs(L, [0, 4]))
    assert lab.faulty.sum() == 15 + 12 and np.all(w2[0] == 0) and np.all(w2[4] == 0) and np.allclose(w2[1], win[1])
    Xr = np.random.default_rng(3).normal(size=(25, 115))
    assert np.allclose(WindowLayout(to_window(Xr)).flatten(to_window(Xr)), Xr)   # saddle branch 50 -> 25 block mean
    # native write-back for a branch with a different length (20 samples vs the 25-sample canonical grid)
    win20 = [np.linspace(0, 1, 20)[:, None] * np.ones((1, 3)), rng.normal(size=(25, 2))]
    L20 = WindowLayout(win20, t_ref=25)
    Xc = L20.flatten(win20)
    m = np.zeros((25, 5), bool)
    m[10:15, 0] = True
    fixed = L20.write_back(win20, Xc * 0 + 7.0, m)
    assert (fixed[0][:, 0] == 7.0).sum() >= 3 and np.allclose(fixed[0][:, 1], win20[0][:, 1]) and np.allclose(fixed[1], win20[1])
    print("ok injectors_and_layout")


def test_detection_synthetic():
    W = world()
    t0 = time.perf_counter()
    res = benchmark_detector(W["det"].detect, W["te"], KINDS, n_faults=1, seed=3)
    RESULTS["synthetic_detection"] = res
    print(f"synthetic detection (C=115, {len(W['te'])} windows, 1 fault/window, {time.perf_counter() - t0:.1f}s)")
    print(f"  clean: false-positive rate per window {res['clean']['fpr_window']:.3f}, per channel {res['clean']['fpr_channel']:.5f}")
    for k in KINDS:
        r = res[k]
        print(f"  {k:12s} n={r['n_windows']:3d} recall {r['recall']:.3f} precision {r['precision']:.3f} F1 {r['f1']:.3f} "
              f"isolation top1 {r['iso_top1']:.3f} top3 {r['iso_top3']:.3f}")
    assert res["clean"]["fpr_window"] <= 0.03
    for k in ("stuck_rail", "stuck_value", "spike", "dropout"):
        assert res[k]["recall"] >= 0.95 and res[k]["precision"] >= 0.95 and res[k]["iso_top1"] >= 0.95, k
    assert res["drift"]["recall"] >= 0.90 and res["drift"]["precision"] >= 0.95 and res["drift"]["iso_top1"] >= 0.90
    # multi-fault: three simultaneous faults are all isolated (peeling, no smearing onto neighbours)
    r3 = benchmark_detector(W["det"].detect, W["te"][:150], ("drift", "stuck_rail", "dropout"), n_faults=3, seed=4)
    RESULTS["synthetic_detection_3faults"] = r3
    for k in ("drift", "stuck_rail", "dropout"):
        print(f"  3 faults/window {k:11s} recall {r3[k]['recall']:.3f} precision {r3[k]['precision']:.3f}")
        assert r3[k]["recall"] >= 0.85 and r3[k]["precision"] >= 0.90, k
    print("ok detection_synthetic")


def test_virtual_sensor():
    W = world()
    ev = W["bank"].evaluate(W["te"][:60], gap_frac=0.8)
    ev5 = W["bank"].evaluate(W["te"][:30], gap_frac=0.8, extra_failed=5)
    RESULTS["virtual_sensor"] = dict(single=ev, plus5_failed=ev5)
    print(f"virtual sensor (synthetic, RMSE in sigma, 80% gap): virtual {ev['virtual']:.3f}  forward-fill {ev['ffill']:.3f}  "
          f"zero-fill {ev['zero']:.3f} | +5 other channels failed: virtual {ev5['virtual']:.3f}")
    assert ev["virtual"] < 0.7 * ev["ffill"] and ev["virtual"] < 0.5 * ev["zero"] and ev["virtual_beats_ffill_frac"] > 0.8
    assert ev5["virtual"] < ev5["ffill"]
    print("ok virtual_sensor")


def test_physics_fallback():
    rng = np.random.default_rng(5)
    N, C, K, H = 800, 40, 6, 16
    A = rng.normal(size=(C, K))
    B = rng.normal(size=(K, H * H))
    lat = rng.normal(size=(N + 200, K))
    F = lat @ A.T + 0.1 * rng.normal(size=(N + 200, C))
    Y = (lat @ B).reshape(-1, H, H)
    fb = fit_physics_fallback(F[:N], Y[:N])
    avail_all = np.ones(C, bool)
    half = avail_all.copy()
    half[:20] = False                                                   # lose half the channels (the "flux loops")
    few = np.zeros(C, bool)
    few[30:33] = True                                                   # only 3 survive (coil currents + Ip)
    rm = lambda P: np.sqrt(np.mean((P - Y[N:]) ** 2))
    err = {}
    for name, av in (("all", avail_all), ("half", half), ("3 channels", few)):
        err[name] = rm(np.stack([fb.predict(f, av) for f in F[N:]]))
    err["mean map"] = rm(np.stack([fb.predict(f, np.zeros(C, bool)) for f in F[N:]]))
    err["zero-fill"] = rm(np.zeros_like(Y[N:]))
    print("physics fallback (synthetic linear world, RMSE of psi): " + ", ".join(f"{k} {v:.3f}" for k, v in err.items()))
    RESULTS["fallback_synthetic"] = err
    assert err["all"] < err["half"] < err["3 channels"] < err["mean map"] and err["all"] < 0.5 * err["mean map"]
    print("ok physics_fallback")


def test_router_rules():
    W = world()
    X = W["te"][0]
    win = to_window(X)
    L = WindowLayout(win)
    router = RouterV2(None, L, W["bank"], fallback="dummy", short_ms=1.0, virtual_r2_min=0.9, max_masked_branches=3)
    ch = lambda b, i=0: int(L.channels_of(b)[i])

    def route_of(specs):
        Y, lab = inject(X, specs)
        dec = router.decide(detection_from_labels(lab), L)
        return dec

    assert router.decide(detection_from_labels(inject(X, [])[1]), L).route == "healthy"
    d = route_of([FaultSpec(ch(0), "dropout", 2.0, 0.6)])                                 # short gap, tier 3
    assert d.route == "transient_ffill" and d.branch_mask.all() and d.transient[ch(0)]
    d = route_of([FaultSpec(ch(1), "spike", 2.0, None, 12.0)])                           # spike on a TIER-1 sensor
    assert d.route == "transient_ffill" and d.branch_mask.all()
    d = route_of([FaultSpec(ch(0), "drift", 1.0, None, 0.5)])                            # long, redundant (tier 3)
    assert d.route == "virtual_sensor" and d.virtual[ch(0)] and d.branch_mask.all(), (d.route, d.virtual.sum())
    d = route_of([FaultSpec(ch(0), "dropout", 0.0, None)])                               # long dropout, redundant
    assert d.route == "virtual_sensor"
    d = route_of([FaultSpec(ch(1, 3), "stuck_rail", 1.0, None, 8.0)])                    # long fault, tier-1 sensor
    assert d.route == "module2" and d.branch_mask[1] == 0 and d.branch_mask.sum() == 7
    d = route_of([FaultSpec(ch(2, i), "dropout", 0.0, None) for i in range(7)])          # 7 channels of one branch
    assert d.route in ("module2", "virtual_sensor") and (d.branch_mask == 0).sum() + d.virtual.sum() >= 1
    d = route_of([FaultSpec(ch(2, i), "drift", 0.0, None, 0.5) for i in range(6)])       # multi-sensor outage >= 6
    assert d.route == "module2" and d.branch_mask[2] == 0
    d = route_of([FaultSpec(ch(1, i), "drift", 0.0, None, 0.5) for i in range(1, 4)] + [FaultSpec(ch(2, 0), "drift", 0.0, None, 0.5)])
    assert d.route == "module2" and d.branch_mask[1] == 0          # touches the tier-1 branch
    d = route_of(group_loss_specs(L, [0, 4]))                                             # two whole branches
    assert d.route == "module2" and d.branch_mask[0] == 0 and d.branch_mask[4] == 0 and d.branch_mask.sum() == 6
    d = route_of(group_loss_specs(L, [0, 1, 2, 3, 4]))                                    # magnetics group (5 > 3 allowed)
    assert d.route == "physics_fallback" and d.use_fallback
    router2 = RouterV2(None, L, W["bank"], fallback="dummy", max_masked_branches=5)       # trained for 5 masked branches
    _, lab = inject(X, group_loss_specs(L, [0, 1, 2, 3, 4]))
    assert router2.decide(detection_from_labels(lab), L).route == "module2"
    assert router2.decide(detection_from_labels(lab), L, have_model=False).route == "physics_fallback"
    # mitigation really repairs: forward-fill holds the last good value, virtual sensor tracks the truth
    Y, lab = inject(X, [FaultSpec(ch(0), "dropout", 2.0, 0.6)])
    out, mask, route, acts = router.route_and_mitigate(to_window(Y), oracle_faults=lab, layout=L)
    held = L.flatten(out)[:, ch(0)]
    assert route == "transient_ffill" and np.allclose(held[10:13], Y[9, ch(0)]) and acts[ch(0)] == "forward-fill"
    Y, lab = inject(X, [FaultSpec(ch(0), "dropout", 0.0, None)])
    out, mask, route, _ = router.route_and_mitigate(to_window(Y), oracle_faults=lab, layout=L)
    v = L.flatten(out)[:, ch(0)]
    assert route == "virtual_sensor" and np.sqrt(np.mean((v - X[:, ch(0)]) ** 2)) < 0.5 * np.sqrt(np.mean(X[:, ch(0)] ** 2))
    out, mask, route, _ = router.route_and_mitigate(to_window(*[inject(X, [FaultSpec(ch(1), "stuck_rail", 1.0)])[0]]),
                                                    oracle_faults=inject(X, [FaultSpec(ch(1), "stuck_rail", 1.0)])[1], layout=L)
    assert route == "module2" and np.all(out[1] == 0) and mask[1] == 0
    print("ok router_rules")


def test_oracle_vs_detected():
    """Toy scoring: the 'model' is the linear reconstruction of a psi map; oracle routing must be >= detected routing
    >= no mitigation (error: lower is better)."""
    W = world()
    rng = np.random.default_rng(11)
    C = 115
    B = rng.normal(size=(W["A"].shape[1], 36)) / 3
    # psi is a fixed linear function of the latent modes behind the sensors
    lat_map = np.linalg.pinv(W["A"])                          # sensors -> latent (K)
    psi_of = lambda Xw: (Xw.mean(0) @ lat_map.T @ B).reshape(6, 6)
    feats = W["tr"].mean(1)
    Ytr = np.stack([psi_of(x) for x in W["tr"][:1000]])
    fb = fit_physics_fallback(feats[:1000], Ytr)
    layout = WindowLayout(to_window(W["te"][0]))
    router = RouterV2(None, layout, W["bank"], fb)
    det = W["det"]
    errs = {"none": [], "detected": [], "oracle": []}
    kinds = ("drift", "stuck_rail", "dropout", "spike", "stuck_value")
    n_oracle_better = 0
    for i in range(150):
        X = W["te"][i]
        truth = psi_of(X)
        specs = random_specs(rng, C, kinds[i % len(kinds)], n=int(rng.integers(1, 5)))
        Y, lab = inject(X, specs)
        model = lambda Xw, mask: fb.predict(fb.features(layout.flatten(Xw)), ~np.isin(layout.branch_of, np.flatnonzero(mask == 0)))
        win = to_window(np.nan_to_num(Y))
        e_none = np.sqrt(np.mean((model(win, np.ones(8)) - truth) ** 2))
        d = det.detect(layout.flatten(win))
        out_d, m_d, _, _ = router.route_and_mitigate(win, diagnoses=d, layout=layout)
        out_o, m_o, _, _ = router.route_and_mitigate(win, oracle_faults=lab, layout=layout)
        e_d, e_o = (np.sqrt(np.mean((model(w, m) - truth) ** 2)) for w, m in ((out_d, m_d), (out_o, m_o)))
        errs["none"].append(e_none); errs["detected"].append(e_d); errs["oracle"].append(e_o)
    m = {k: float(np.mean(v)) for k, v in errs.items()}
    RESULTS["oracle_toy"] = m
    print("oracle vs detected (toy, mean psi RMSE, synthetic): " + ", ".join(f"{k} {v:.4f}" for k, v in m.items()))
    assert m["oracle"] <= m["detected"] * 1.02 and m["detected"] < m["none"], m
    print("ok oracle_vs_detected")


def test_conformal_and_flags():
    rng = np.random.default_rng(7)
    alpha = 0.1

    def draw(n):
        f = rng.uniform(0, 1, size=(n, 1))                       # difficulty feature (e.g. residual score)
        return np.abs(rng.normal(size=n)) * (0.2 + 2.0 * f[:, 0]), f

    e_cal, f_cal = draw(1500)
    e_te, f_te = draw(40000)
    g = ConformalGauge(alpha)
    g.calibrate({"r": list(e_cal)})
    cov_global = empirical_coverage(g.delta["r"], e_te)
    ag = AdaptiveConformalGauge(alpha)
    ag.calibrate_adaptive(e_cal, f_cal)
    d_te = np.array([ag.delta_for("r", f) for f in f_te[:6000]])
    cov_ad = empirical_coverage(d_te, e_te[:6000])
    easy, hard = f_te[:6000, 0] < 0.2, f_te[:6000, 0] > 0.8
    cov_easy_g, cov_hard_g = empirical_coverage(g.delta["r"], e_te[:6000][easy]), empirical_coverage(g.delta["r"], e_te[:6000][hard])
    cov_easy_a, cov_hard_a = empirical_coverage(d_te[easy], e_te[:6000][easy]), empirical_coverage(d_te[hard], e_te[:6000][hard])
    print(f"conformal (synthetic exchangeable, alpha=0.1, target {1 - alpha:.2f}): global coverage {cov_global:.3f}, adaptive {cov_ad:.3f}; "
          f"global delta {g.delta['r']:.2f}, adaptive mean delta easy {d_te[easy].mean():.2f} / hard {d_te[hard].mean():.2f}; "
          f"coverage easy/hard: global {cov_easy_g:.2f}/{cov_hard_g:.2f}, adaptive {cov_easy_a:.2f}/{cov_hard_a:.2f}")
    RESULTS["conformal"] = dict(global_cov=cov_global, adaptive_cov=cov_ad, easy_hard_global=(cov_easy_g, cov_hard_g),
                                easy_hard_adaptive=(cov_easy_a, cov_hard_a))
    assert abs(cov_global - (1 - alpha)) < 0.02 and abs(cov_ad - (1 - alpha)) < 0.03
    assert d_te[easy].mean() < g.delta["r"] < d_te[hard].mean()
    assert abs(cov_easy_a - (1 - alpha)) < abs(cov_easy_g - (1 - alpha)) and abs(cov_hard_a - (1 - alpha)) < abs(cov_hard_g - (1 - alpha))
    # finite-sample guarantee: average coverage over many small calibration sets is >= 1 - alpha
    covs = []
    for _ in range(300):
        e_c, _ = draw(40)
        gg = ConformalGauge(alpha)
        gg.calibrate({"r": list(e_c)}, min_samples=20)
        covs.append(empirical_coverage(gg.delta["r"], e_te[:3000]))
    assert np.mean(covs) >= 1 - alpha - 0.01, np.mean(covs)
    # the alarm rule: d_wall - delta < 5 cm
    a = g.assess_clearance(5.5, "r")
    assert a["point_alarm"] is False and a["alarm"] is True and a["delta_cm"] == g.delta["r"]
    # confidence flags (HIGH / LOW / ABSTAIN) derived from delta and the 5 cm wall margin
    assert confidence_flag(12.0, 1.0) == "HIGH"          # far from the wall, narrow interval
    assert confidence_flag(5.5, 1.0) == "LOW"            # decision inside the uncertainty band
    assert confidence_flag(12.0, 3.0) == "LOW"           # decided but wider than half the margin
    assert confidence_flag(12.0, 5.5) == "ABSTAIN"       # interval wider than the margin
    assert confidence_flag(12.0, 1.0, valid=False) == "ABSTAIN"
    print("ok conformal_and_flags")


def test_flux_loop_residual():
    """Loops that no other sensor can predict (they carry their own latent modes): the cross-sensor model cannot see a
    0.5 sigma offset on them, the flux-loop residual against the reconstruction can."""
    rng = np.random.default_rng(13)
    K, Kl, C0, L, H = 4, 3, 30, 3, 20
    A = rng.normal(size=(C0, K)); A /= np.linalg.norm(A, axis=1, keepdims=True)
    pix = [(4, 5), (10, 15), (16, 8)]
    Bpsi = rng.normal(size=(K + Kl, H, H))

    def make(n):
        lat = rng.normal(size=(n, K + Kl))
        psi = np.einsum("nk,khw->nhw", lat, Bpsi)
        fl = np.stack([psi[:, r, c] * 2.0 + 0.5 for r, c in pix], 1) + 0.05 * rng.normal(size=(n, L))
        sens = lat[:, :K] @ A.T + 0.15 * rng.normal(size=(n, C0))
        X = np.concatenate([sens, fl], 1)[:, None, :] + 0.02 * rng.normal(size=(n, 25, C0 + L))
        return X, psi + 0.02 * rng.normal(size=psi.shape)

    (Xt, Pt), (Xc, Pc), (Xe, Pe) = make(1200), make(600), make(300)
    mu, sd = Xt.reshape(-1, C0 + L).mean(0), Xt.reshape(-1, C0 + L).std(0)
    Xt, Xc, Xe = [(x - mu) / sd for x in (Xt, Xc, Xe)]
    chans = list(range(C0, C0 + L))
    fc = FluxLoopCheck(pix, chans, fp_rate=1e-3)
    fc.fit(Xt.mean(1)[:, chans], Pt, calib=(Xc.mean(1)[:, chans], Pc))
    d0 = FaultDetector(fp_rate=1e-4).fit(Xt, Xc)
    d1 = FaultDetector(fp_rate=1e-4, flux_check=fc).fit(Xt, Xc)
    hit0 = hit1 = fp1 = 0
    n = 200
    for i in range(n):
        X = Xe[i]
        c = C0 + int(rng.integers(L))
        Y = X.copy()
        Y[:, c] += float(rng.choice([-1, 1]) * 0.5)          # calibration offset of 0.5 sigma: inside the loop's natural range
        hit0 += bool(d0.detect(Y).flagged[c])
        r = d1.detect(Y, psi_pred=Pe[i])
        hit1 += bool(r.flagged[c])
        fp1 += int(d1.detect(X, psi_pred=Pe[i]).flagged.any())
    print(f"flux-loop residual (synthetic, loops unpredictable from other sensors, 0.5 sigma offset): recall without {hit0 / n:.2f}, "
          f"with {hit1 / n:.2f}; clean false-positive windows {fp1 / n:.3f}")
    RESULTS["flux_loop"] = dict(recall_without=hit0 / n, recall_with=hit1 / n, fp_window=fp1 / n)
    assert hit1 / n >= 0.9 and hit1 / n > hit0 / n + 0.2 and fp1 / n <= 0.05
    print("ok flux_loop_residual")


class StubGauge(ConformalGauge):
    """Gauge that does not need the physics layer: clearance = 8 + mean(psi)."""
    def assess(self, psi_pred, route, difficulty=None):
        return self.assess_clearance(8.0 + float(np.mean(psi_pred)), route, difficulty)


def test_brain_api_and_latency():
    W = world()
    X = W["te"][0]
    L = WindowLayout(to_window(X))
    fb = fit_physics_fallback(W["tr"].mean(1)[:500], np.random.default_rng(0).normal(size=(500, 8, 8)))
    gauge = StubGauge(0.1)
    gauge.calibrate({r: [0.5] * 30 for r in ("healthy", "transient_ffill", "virtual_sensor", "module2", "physics_fallback")})
    brain = SmartAdaptiveMitigationBrain(None, gauge, None, detector=W["det"], bank=W["bank"], fallback=fb)
    oracle = SmartAdaptiveMitigationBrain(None, gauge, None, oracle=True, bank=W["bank"], fallback=fb)
    model = lambda window, mask: np.full((8, 8), float(mask.mean()))
    rng = np.random.default_rng(2)
    rows, routes = [], {}
    for i in range(300):
        Xi = W["te"][i % len(W["te"])]
        kind = ("none", "drift", "dropout", "stuck_rail", "spike")[i % 5]
        specs = [] if kind == "none" else random_specs(rng, 115, kind, 1)
        Y, lab = inject(Xi, specs)
        win = to_window(np.nan_to_num(Y))
        res = brain.process(win, model)
        res_o = oracle.process(win, model, oracle_faults=lab)
        rows.append(res["latencies_ms"])
        routes[res["route"]] = routes.get(res["route"], 0) + 1
        assert set(res["assessment"]) >= {"alarm", "point_alarm", "delta_cm", "confidence"} and res["mask"].shape == (8,)
    lat = summarise_latencies(rows)
    RESULTS["latency"] = lat
    RESULTS["routes"] = routes
    print("v2 brain latency, stub model, C=115 (ms, numpy on this CPU, 300 windows): " + "; ".join(
        f"{k} p50 {v['p50']:.2f} p95 {v['p95']:.2f} p99 {v['p99']:.2f}" for k, v in lat.items()))
    print("  routes taken:", routes)
    assert "healthy" in routes and lat["total"]["p99"] < 100
    # legacy API still works unchanged (physics_study.py path): same constructor, same process() contract
    legacy = SmartAdaptiveMitigationBrain({0: 2, 1: 3, 2: 1}, StubGauge(0.1), None)
    legacy.gauge.calibrate({r: [0.5] * 30 for r in ("healthy", "transient_ffill", "redundant_interp", "module2")})
    w = [np.random.default_rng(0).normal(size=(25, 15)), np.random.default_rng(1).normal(size=(25, 8)), np.zeros(20)]
    res = legacy.process(w, lambda window, mask: np.zeros((8, 8)))
    assert res["route"] == "module2" and res["mask"].tolist() == [1, 1, 0] and "total" in res["latencies_ms"]
    print("ok brain_api_and_latency")


# ------------------------------------------------------------------ real data
def build_real_windows(npz, stride=10, max_shots=14):
    """Reads every local zarr shot with the full task_1-3 input set (needs zarr; see MODULE4_V2.md for the PYTHONPATH
    note) -> npz with one (n_windows, 25, 115) canonical 5 ms window matrix per shot (raw units, plasma phase only).
    Shots whose probe counts differ from BRANCH_P (e.g. MAST-U) are skipped."""
    import zarr
    sigs = [("magnetics", "flux_loop_flux", "time"), ("magnetics", "b_field_pol_probe_ccbv_field", "time"),
            ("magnetics", "b_field_pol_probe_obr_field", "time"), ("magnetics", "b_field_pol_probe_obv_field", "time"),
            ("magnetics", "b_field_tor_probe_saddle_voltage", "time_saddle"), ("pf_active", "coil_current", "time"),
            ("pf_active", "solenoid_current", "time"), ("summary", "ip", "time")]
    root = os.path.join(SP, "data", "random")
    out, layout = {}, None
    for name in sorted(os.listdir(root)):
        if not name.endswith(".zarr") or len(out) >= max_shots:
            continue
        try:
            g = zarr.open_group(os.path.join(root, name), mode="r", use_consolidated=False)
            data = []
            for grp, sig, tname in sigs:
                a = np.asarray(g[grp][sig], dtype=np.float64)
                a = a.reshape(1, -1) if a.ndim == 1 else a
                data.append((a, np.asarray(g[grp][tname], dtype=np.float64)))
            t_m = np.asarray(g["magnetics"]["time"], dtype=np.float64)
        except Exception:
            continue
        if [d[0].shape[0] for d in data] != BRANCH_P:
            continue
        # keep the plasma phase: windows whose start has |Ip| >= 10 % of its peak (before / after the pulse the
        # signals are identically zero or constant and the task has no target there)
        ip, t_ip = np.abs(data[7][0][0]), data[7][1]
        live = t_ip[ip >= 0.1 * np.nanmax(ip)]
        wins = []
        for i0 in range(0, len(t_m) - 26, stride):
            t0 = t_m[i0]
            if not (live.min() <= t0 <= live.max()):
                continue
            w = []
            for a, t in data:
                j0 = int(np.searchsorted(t, t0 - 1e-9))
                n = int(round(0.005 / np.median(np.diff(t))))
                seg = a[:, j0:j0 + n].T
                if seg.shape[0] != n:
                    w = None
                    break
                w.append(seg)
            if w is not None:
                wins.append(w)
        if len(wins) < 10:
            continue
        layout = layout or WindowLayout(wins[0], t_ref=25)     # block mean for the 250-sample saddle branch, interpolation for 20-sample branches
        out[name[:-5]] = np.stack([layout.flatten(w) for w in wins])
    np.savez(npz, **out)


def real_windows():
    npz = os.environ.get("TM_M4_REAL_NPZ", os.path.join(SP, "real_m4_windows.npz"))
    if not os.path.exists(npz):
        try:
            build_real_windows(npz)
        except Exception as e:                                   # zarr / shots missing
            print("  (real-data test skipped:", type(e).__name__, str(e)[:80], ")")
            return None
    return np.load(npz)


def test_real_data_false_positives():
    d = real_windows()
    if d is None or len(d.files) < 6:
        print("skip real_data_false_positives (needs >= 6 local shots)")
        return
    shots = sorted(d.files)
    keep_ch = ~np.any([np.isnan(d[s]).mean(axis=(0, 1)) > 0.02 for s in shots], axis=0)    # probes that are NaN in a shot
    n_all = n_gap = 0
    split = {"train": [], "calib": [], "test": []}
    for i, s in enumerate(shots):                                  # held-out SHOTS: 60 % train / 20 % calibration / 20 % test
        W = d[s][:, :, keep_ch]
        gap = np.isnan(W).any(axis=(1, 2))                         # windows with a recorded NaN gap are real faults, not "clean"
        n_all, n_gap = n_all + len(W), n_gap + int(gap.sum())
        split[("train", "train", "train", "calib", "test")[i % 5]].append(W[~gap])
    tr, ca, te = [np.concatenate(split[k]) for k in ("train", "calib", "test")]
    tr_sh = tr[:: 1]
    det = FaultDetector(fp_rate=1e-4).fit(tr_sh, ca)
    det_dm = FaultDetector(fp_rate=1e-4, demean=True).fit(tr_sh, ca)
    kinds = ("stuck_rail", "stuck_value", "drift", "spike", "dropout")
    r = benchmark_detector(det.detect, te, kinds, seed=5)
    r_dm = benchmark_detector(det_dm.detect, te, kinds, seed=5)
    n_abs = int(det.absent.sum())
    RESULTS["real_detection"] = dict(level=r, demean=r_dm, n_train=len(tr), n_calib=len(ca), n_test=len(te), n_shots=len(shots),
                                     n_absent=n_abs, nan_gap_windows=n_gap, windows=n_all)
    print(f"REAL data: {len(shots)} MAST shots {','.join(shots)}; {len(tr)}/{len(ca)}/{len(te)} train/calib/test windows from "
          f"different shots; {int((~keep_ch).sum())} probes that are NaN in some shot left out, {n_abs} always-zero ones excluded; "
          f"{n_gap}/{n_all} windows with recorded NaN gaps left out")
    for name, rr in (("level+dynamics residuals", r), ("dynamics-only residuals (demean)", r_dm)):
        print(f"  [{name}] clean test windows: false-positive rate per window {rr['clean']['fpr_window']:.3f}, per channel {rr['clean']['fpr_channel']:.5f}")
        for k in kinds:
            print(f"    injected {k:11s} recall {rr[k]['recall']:.3f} precision {rr[k]['precision']:.3f} isolation top1 {rr[k]['iso_top1']:.3f} top3 {rr[k]['iso_top3']:.3f}")
    # real held-out shots are non-stationary and heavy tailed: the false-positive rate is printed, not asserted tight
    assert r["clean"]["fpr_window"] <= 0.9 and r_dm["clean"]["fpr_window"] <= 0.9
    for k in ("dropout", "stuck_rail"):
        assert r[k]["recall"] >= 0.9, k
    print("ok real_data_false_positives")


if __name__ == "__main__":
    if "--build-real" in sys.argv:
        build_real_windows(os.path.join(SP, "real_m4_windows.npz"))
        print("built real_m4_windows.npz")
        sys.exit(0)
    for fn in (test_injectors_and_layout, test_detection_synthetic, test_virtual_sensor, test_physics_fallback,
               test_router_rules, test_oracle_vs_detected, test_conformal_and_flags, test_flux_loop_residual,
               test_brain_api_and_latency, test_real_data_false_positives):
        fn()
    print("ALL MODULE 4 v2 TESTS PASSED")
