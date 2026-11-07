#!/usr/bin/env python
"""Assert-based tests of the Module 3/4 study code (physics_study.py, sensor_criticality.py). numpy + scipy only (no torch, no zarr).

    python test_module34_study.py            (about 20 s)
An optional check against REAL collated windows runs when a window cache exists (env M34_CACHE=<dir with val.npz>).
"""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import physics_study as PS  # noqa: E402
import sensor_attribution as SA  # noqa: E402
import sensor_criticality as SC  # noqa: E402
from fault_models import WindowLayout  # noqa: E402

COUNTS = (15, 40, 18, 18, 12, 10, 1, 1)
TIMES = (25, 25, 25, 25, 250, 20, 20, 20)


def synth_set(n=40, seed=0):
    """Windows with the real layout: x[b] (N,1,1,T,n) profile / (N,1,1,T) scalar, low-rank + noise so channels are predictable."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 6))
    x = []
    for b, (c, T) in enumerate(zip(COUNTS, TIMES)):
        load = rng.normal(size=(6, c))
        base = (z @ load)[:, None, :] + 0.05 * rng.normal(size=(n, T, c))
        x.append((base[:, None, None] if c > 1 else base[:, None, None, :, 0]).astype(np.float32))
    return PS.WindowSet(np.repeat(np.arange(4), n // 4), np.arange(n), x, rng.normal(size=(n, 65, 65)))


def test_windowset_roundtrip(tmp="_ws_test.npz"):
    ws = synth_set()
    w = ws.window(3)
    assert [a.shape for a in w] == [(T, c) for T, c in zip(TIMES, COUNTS)]
    back = ws.put([ws.window(i) for i in range(len(ws))])
    assert all(np.allclose(a, b) for a, b in zip(back, ws.x))
    ws.save(os.path.join(HERE, tmp))
    ws2 = PS.WindowSet.load(os.path.join(HERE, tmp))
    os.remove(os.path.join(HERE, tmp))
    assert len(ws2) == len(ws) and all(np.array_equal(a, b) for a, b in zip(ws.x, ws2.x))
    assert len(ws.subset([0, 5, 9])) == 3


def test_scenarios_and_imputation():
    ws = synth_set()
    L = WindowLayout(ws.window(0))
    assert L.n_channels == 115
    rng = np.random.default_rng(1)
    usable = np.arange(115)
    from fault_models import inject_window
    for scen, expect in (("channel_loss", None), ("group_loss", 15 + 40 + 18 + 18 + 12), ("temporal_gap", 115), ("pre_disruption", 115),
                         ("sensor_loss", 6), ("stuck", 3), ("drift", 3), ("spike", 3), ("clean", 0)):
        specs = PS.make_faults(scen, L, rng, usable)
        win, lab = inject_window(ws.window(0), L, specs)
        n_f = int(lab.faulty.sum())
        if expect is not None:
            assert n_f == expect, (scen, n_f)
    # channel loss k=3: whole branches only, 3 distinct branches, input is zero there
    specs = PS.make_faults("channel_loss", L, np.random.default_rng(2), usable, channel_k=3)
    win, lab = inject_window(ws.window(0), L, specs)
    lost = {int(L.branch_of[c]) for c in lab.channels}
    assert len(lost) == 3 and all((win[b] == 0).all() for b in lost)
    # temporal gap keeps >= 60 % of the samples; pre-disruption blanks the END of the window
    specs = PS.make_faults("pre_disruption", L, rng, usable)
    win, lab = inject_window(ws.window(0), L, specs)
    assert lab.mask[:18].sum() == 0 and lab.mask[-1].mean() > 0.85 and (win[0][-1] == 0).all() and not (win[0][0] == 0).all()
    # imputation of KNOWN holes: zero / mean / ffill
    specs = [PS.FaultSpec(2, "dropout", 2.0, 2.0)]
    clean = ws.window(1)
    win, lab = inject_window(clean, L, specs)
    mean_c = np.full(115, 7.0)
    f = PS.impute_window(win, L, lab, "ffill", mean_c)
    k = np.flatnonzero(lab.mask[:, 2])
    assert len(k) > 0 and np.allclose(L.flatten(f)[k, 2], L.flatten(clean)[k[0] - 1, 2], atol=1e-6)
    m = PS.impute_window(win, L, lab, "mean", mean_c)
    assert np.allclose(L.flatten(m)[k, 2], 7.0)
    z = PS.impute_window(win, L, lab, "zero", mean_c)
    assert np.allclose(L.flatten(z)[k, 2], 0.0)


def test_sensor_axis_masking_synthetic():
    """The lost-sensor mask zeroes the SENSOR axis (-1) of profile branches, whole scalar branches, and sets branch health only when
    the branch is entirely gone."""
    inv = SA.default_inventory()
    ws = synth_set(8)
    mask = np.ones(115, bool)
    mask[inv.indices(0)[3]] = False                     # flux loop #3
    out = inv.apply_mask_to_inputs(ws.x, mask)
    assert (out[0][..., 3] == 0).all() and np.allclose(out[0][..., [0, 1, 2, 4]], ws.x[0][..., [0, 1, 2, 4]])
    assert all(np.array_equal(out[b], ws.x[b]) for b in range(1, 8))
    assert inv.branch_health(mask).tolist() == [1.0] * 8
    mask2 = mask.copy()
    mask2[inv.indices(7)] = False                       # ip gone: scalar branch zeroed entirely, health 0
    mask2[inv.indices(0)] = False                       # whole flux-loop branch gone
    out2 = inv.apply_mask_to_inputs(ws.x, mask2)
    assert (out2[7] == 0).all() and (out2[0] == 0).all()
    assert inv.branch_health(mask2).tolist() == [0, 1, 1, 1, 1, 1, 1, 0]


def test_sensor_axis_masking_real_batch():
    cache = os.environ.get("M34_CACHE")
    if not cache or not os.path.exists(os.path.join(cache, "val.npz")):
        print("  (real-batch check skipped: set M34_CACHE=<cache dir>)")
        return
    ws = PS.WindowSet.load(os.path.join(cache, "val.npz"))
    inv = SA.default_inventory()
    assert [v.shape[1:] for v in ws.x][:6] == [(1, 1, T, c) for T, c in zip(TIMES[:6], COUNTS[:6])]
    assert ws.x[6].shape[1:] == (1, 1, 20) and ws.x[7].shape[1:] == (1, 1, 20)
    for g in (0, 1, 4, 5):
        mask = np.ones(115, bool)
        j = int(inv.indices(g)[2])
        mask[j] = False
        out = inv.apply_mask_to_inputs(ws.x, mask)
        assert (out[g][..., 2] == 0).all()
        keep = [c for c in range(COUNTS[g]) if c != 2]
        assert np.array_equal(out[g][..., keep], ws.x[g][..., keep])          # every other sensor and all time steps untouched
    print("  real collated windows: sensor-axis masking verified on", len(ws), "windows")


def test_bootstrap_and_nrmse():
    shots = np.repeat(np.arange(5), 10)
    b = np.full(50, 3.0)
    r = np.full(50, 0.01)
    out = PS.shot_bootstrap(shots, b, r, 0.05, n_boot=200, seed=0)
    assert abs(out["boundary_cm"] - 3.0) < 1e-12 and out["boundary_ci95"] == [3.0, 3.0] and abs(out["nrmse"] - 0.2) < 1e-12
    assert out["n_shots"] == 5 and out["n_windows"] == 50
    # two shots with different rmse: official NRMSE = mean over shots of shot_rmse / sd
    rm = np.array([0.01] * 10 + [0.03] * 10)
    assert abs(PS.nrmse_official(np.repeat([0, 1], 10), rm, 0.1) - 0.2) < 1e-12
    # nan boundary windows are ignored, not counted as zero
    b2 = b.copy()
    b2[:5] = np.nan
    assert abs(PS.shot_bootstrap(shots, b2, r, 0.05, n_boot=50)["boundary_cm"] - 3.0) < 1e-12
    # CI contains the estimate and widens with between-shot spread
    rng = np.random.default_rng(0)
    b3 = np.repeat(rng.normal(3, 1, 5), 10)
    o3 = PS.shot_bootstrap(shots, b3, r, 0.05, n_boot=300)
    assert o3["boundary_ci95"][0] < o3["boundary_cm"] < o3["boundary_ci95"][1]


def test_clearance_alarm():
    a = dict(d_wall_true_cm=np.array([10.0, 3.0, 3.0, 10.0]), d_wall_pred_cm=np.array([4.0, 4.0, 10.0, 10.0]))
    r = PS.clearance_alarm(a)
    assert r["false_alarms"] == 1 and r["missed"] == 1 and r["n_true_alarm"] == 2 and abs(r["false_alarm_rate"] - 0.5) < 1e-12
    r2 = PS.clearance_alarm(a, delta=np.array([7.0, 0, 0, 7.0]))     # a wide interval turns the last window into a (false) alarm
    assert r2["false_alarms"] == 2


def test_lazy_greedy_matches_greedy_on_additive_game():
    w = np.array([5.0, 0.1, 3.0, 0.2, 0.05, 2.0])
    ev = SA.Evaluator(lambda m: float(w[~m].sum()))           # failing player i adds w_i to the error
    g = SA.Game(ev, 6)
    d1 = SA.drop_one(g)["delta"]
    curve = SC.lazy_greedy_backward(g, d1)
    ref = SA.greedy_backward(g, 1e9)["curve"]
    assert [c["failed"] for c in SC.lazy_greedy_backward(g, d1, batch=4)] == [c["failed"] for c in ref]   # batched re-evaluation
    early = SC.lazy_greedy_backward(g, d1, stop_above=2.0)                                            # early stop + real-evaluated tail
    assert [c["failed"] for c in early] == [c["failed"] for c in ref] and any(c.get("tail") for c in early)
    assert all(abs(c["value"] - w[c["failed"]].sum()) < 1e-9 for c in early)
    assert [c["failed"] for c in curve] == [c["failed"] for c in ref] and len(curve) == 7
    assert abs(curve[-1]["value"] - w.sum()) < 1e-9
    assert SC.min_set_at(curve, 0.5)["n_available"] == 3 and SC.min_set_at(curve, -1) is None
    rc = SC.ranked_curve(g, np.argsort(w))
    assert rc[0]["n_available"] == 6 and rc[-1]["n_available"] == 0 and abs(rc[-1]["value"] - w.sum()) < 1e-9


def test_map_vs_group_sign():
    R, Z = np.linspace(0.06, 1.98, 65), np.linspace(-2, 2, 65)
    RR, ZZ = np.meshgrid(R, Z)
    geo = types.SimpleNamespace(R=np.array([1.0, 1.2]), Z=np.array([0.0, 0.5]))
    near = np.exp(-np.hypot(RR - 1.0, ZZ) ** 2)
    reg = np.ones_like(near, bool)
    st = SC.map_vs_group(near, geo, [0], reg, R, Z)
    assert st["spearman_distance"] < -0.8 and st["centroid_distance_m"] < 0.1 and st["n_positioned"] == 1
    assert SC.map_vs_group(near, geo, [], reg, R, Z)["n_positioned"] == 0


def test_mitigators_oracle_and_detector():
    from fault_models import inject_window
    ws = synth_set(160, seed=3)
    ctx = types.SimpleNamespace(mu=0.0, sd=1.0)
    ctx.sets = {}
    train, cal = ws.subset(np.arange(0, 100)), ws.subset(np.arange(100, 160))
    mit = PS.Mitigators(ctx, train, cal)
    mit.set_policy(thr_scale=1.0, short_ms=1.0, max_masked=5)
    L = mit.L
    win = cal.window(0)
    # oracle: three whole branches lost -> module2 route with exactly those branches masked
    specs = PS.group_loss_specs(L, [1, 2, 4])
    dwin, lab = inject_window(win, L, specs)
    rep, health, route, det, dec, psi = mit.prepare(dwin, lab, oracle=True)
    assert route == "module2" and health.tolist() == [1, 0, 0, 1, 0, 1, 1, 1] and psi is None
    assert (rep[1] == 0).all() and (rep[2] == 0).all() and not (rep[0] == 0).all()
    # six lost branches exceed what the model was trained for -> physics fallback with a finite psi
    dwin, lab = inject_window(win, L, PS.group_loss_specs(L, [0, 1, 2, 3, 4, 5]))
    rep, health, route, det, dec, psi = mit.prepare(dwin, lab, oracle=True)
    assert route == "physics_fallback" and psi.shape == (65, 65) and np.isfinite(psi).all()
    mit.set_policy(1.0, 1.0, 8)                                   # raising the crossover keeps the network in charge
    _, health, route, *_ = mit.prepare(dwin, lab, oracle=True)
    assert route == "module2" and int((health == 0).sum()) == 6
    mit.set_policy(1.0, 1.0, 5)
    # a stuck rail on a usable channel is found by the real detector and isolated first
    hits = 0
    rng = np.random.default_rng(0)
    for i in range(10):
        c = int(rng.choice(mit.usable))
        dwin, lab = inject_window(cal.window(i), L, [PS.FaultSpec(c, "stuck_rail", 0.5, None, 8.0)])
        det = mit.det.detect_window(dwin, L)
        hits += int(det.flagged[c])
    assert hits >= 8, hits
    # clean windows: far fewer flags than faulty ones
    fp = np.mean([mit.det.detect_window(cal.window(i), L).flagged.any() for i in range(20)])
    assert fp <= 0.35, fp
    ds = PS.detection_stats(det, lab)
    assert {"tp", "fn", "fp", "fp_new", "iso_top1", "has_fault", "any_flag"} <= set(ds)


def test_sensor_level_policy():
    """A per-sensor-mask model gets a (115,) health vector that masks exactly the lost probes (not whole branches); short
    transients are forward-filled, whole-branch losses beyond max_masked fall back to the physics fallback."""
    from fault_models import inject_window
    ws = synth_set(160, seed=5)
    ctx = types.SimpleNamespace(mu=0.0, sd=1.0, sensor_masks=True)
    mit = PS.Mitigators(ctx, ws.subset(np.arange(0, 100)), ws.subset(np.arange(100, 160)))
    assert mit.sensor_level
    mit.set_policy(4.0, 2.5, 5, "mask", "mask")
    L = mit.L
    win = ws.window(101)
    ch = [int(L.channels_of(1)[3]), int(L.channels_of(1)[7])]                         # two probes of one branch, whole window
    dwin, lab = inject_window(win, L, [PS.FaultSpec(c, "dropout", 0.0, None) for c in ch])
    rep, health, route, det, dec, psi = mit.prepare(dwin, lab, oracle=True)
    assert health.shape == (115,) and sorted(np.flatnonzero(health == 0).tolist()) == sorted(ch) and route == "module2" and psi is None
    assert (rep[1][:, 3] == 0).all() and not (rep[1][:, 4] == 0).all()
    dwin, lab = inject_window(win, L, PS.group_loss_specs(L, [0, 1, 2, 3, 4, 5]))
    _, health, route, *_, psi = mit.prepare(dwin, lab, oracle=True)
    assert route == "physics_fallback" and psi.shape == (65, 65)
    dwin, lab = inject_window(win, L, [PS.FaultSpec(ch[0], "spike", 1.0, None, 12.0)])
    _, health, route, *_ = mit.prepare(dwin, lab, oracle=True)
    assert route == "transient_ffill" and health.all()


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok", name)
    print(f"{n} tests passed")
