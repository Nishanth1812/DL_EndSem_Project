#!/usr/bin/env python
"""Assert-based tests for sensor_attribution.py (synthetic value functions with analytically known answers).
Run:  python test_sensor_attribution.py
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sensor_attribution as sa  # noqa: E402


def tiny_inv(counts):
    names = [f"s{i}" for i in range(sum(counts))]
    return sa.SensorInventory([f"g{i}" for i in range(len(counts))], list(counts), names)


def game_from_f(f, p, vectorised=False):
    """f(failed_bool) -> error; value_fn takes availability masks."""
    if vectorised:
        ev = sa.Evaluator(lambda M: np.array([f(~m) for m in M]), vectorised=True, batch_size=7)
    else:
        ev = sa.Evaluator(lambda m: f(~m))
    return sa.Game(ev, p)


def test_inventory():
    inv = sa.default_inventory()
    assert inv.n_sensors == 115 and inv.n_groups == 8 and list(inv.counts) == [15, 40, 18, 18, 12, 10, 1, 1]
    assert len(set(inv.sensor_names)) == 115
    m = inv.mask_from_failed_groups([1])
    assert (~m).sum() == 40 and not m[inv.indices(1)].any()
    h = inv.branch_health(inv.mask_from_failed_sensors(list(inv.indices(2))))
    assert h[2] == 0 and h.sum() == 7
    h = inv.branch_health(inv.mask_from_failed_sensors([inv.indices(2)[0]]))
    assert h.sum() == 8  # one dead sensor does not kill the branch
    # input masking: sensor axis last on profile branches, scalar branches blanked as a whole
    shapes = [(2, 1, 1, 25, 15), (2, 1, 1, 25, 40), (2, 1, 1, 25, 18), (2, 1, 1, 25, 18), (2, 1, 1, 250, 12),
              (2, 1, 1, 20, 10), (2, 1, 1, 20), (2, 1, 1, 20)]
    x = [np.ones(s) for s in shapes]
    mask = inv.mask_from_failed_sensors([inv.indices(1)[3], inv.indices(6)[0]])
    y = sa.SensorInventory.apply_mask_to_inputs(inv, x, mask)
    assert (y[1][..., 3] == 0).all() and y[1][..., 2].all() and y[1].sum() == x[1].sum() - 2 * 25
    assert (y[6] == 0).all() and x[6].all() and y[7].all()
    try:
        sa.SensorInventory.apply_mask_to_inputs(inv, [np.ones((1, 15, 25))] * 8, mask)
        raise AssertionError("should have rejected a wrong sensor axis")
    except ValueError:
        pass


def test_additive_all_methods_exact():
    w = np.array([0.0, 1.0, 2.0, 0.5, 4.0, 0.25])
    g = game_from_f(lambda F: float(w[F].sum()), 6)
    vals = sa.full_table(g)
    assert np.allclose(sa.exact_shapley(vals, 6), w)
    assert np.allclose(sa.exact_interaction(vals, 6), 0)
    out = sa.sampled_shapley(g, np.random.default_rng(1), max_pairs=30)
    assert np.allclose(out["phi"], w) and np.allclose(out["se"], 0, atol=1e-12)
    out = sa.sampled_shapley(g, np.random.default_rng(1), max_pairs=30, max_failed=3)  # additive: truncation changes nothing
    assert np.allclose(out["phi"], w)


def test_pairwise_synergy():
    p = 5
    w = np.array([1.0, 0.0, 2.0, 0.5, 0.0])
    C = np.zeros((p, p))
    C[0, 1], C[2, 3], C[1, 4] = 3.0, -1.5, 0.7

    def f(F):
        x = F.astype(float)
        return float(w @ x + x @ C @ x)

    g = game_from_f(f, p)
    vals = sa.full_table(g)
    Cs = C + C.T
    assert np.allclose(sa.exact_shapley(vals, p), w + 0.5 * Cs.sum(1))
    assert np.allclose(sa.exact_interaction(vals, p), Cs)
    # Monte-Carlo interaction agrees (value of the second difference is constant here -> se 0)
    m, se = sa.sampled_interaction(g, 0, 1, np.random.default_rng(0), n_samples=20)
    assert abs(m - 3.0) < 1e-9 and se < 1e-9


def test_redundancy_and_saturation():
    # a,b are a backup pair: error only if BOTH fail. c,d: saturating any-of-two (either loss gives the full damage).
    f = lambda F: 1.0 * (F[0] and F[1]) + 1.0 * (F[2] or F[3]) + 0.0 * F[4]  # noqa: E731
    g = game_from_f(f, 5)
    vals = sa.full_table(g)
    phi = sa.exact_shapley(vals, 5)
    assert np.allclose(phi, [0.5, 0.5, 0.5, 0.5, 0.0])
    I = sa.exact_interaction(vals, 5)
    assert abs(I[0, 1] - 1.0) < 1e-12      # backup pair: positive
    assert abs(I[2, 3] + 1.0) < 1e-12      # overlap: negative
    d = sa.drop_one(g)["delta"]
    assert np.allclose(d, [0, 0, 1, 1, 0])
    cls = sa.classify_sensors(d, phi, crit_tol=0.5, red_tol=0.1)
    assert cls == ["backup", "backup", "critical", "critical", "redundant"]
    assert np.allclose(sa.alone_value(g), [f(~(np.arange(5) == i)) for i in range(5)])


def test_sampled_vs_exact_within_se():
    rng = np.random.default_rng(7)
    p = 10
    w = rng.uniform(0, 1, p)
    C = np.triu(rng.normal(0, 0.4, (p, p)), 1)

    def f(F):
        x = F.astype(float)
        return float(min(3.0, w @ x + x @ C @ x))  # saturating, interacting

    g = game_from_f(f, p)
    exact = sa.exact_shapley(sa.full_table(g), p)
    out = sa.sampled_shapley(g, np.random.default_rng(3), max_pairs=400)
    z = np.abs(out["phi"] - exact) / np.maximum(out["se"], 1e-12)
    assert z.max() < 4.0, z
    assert abs(out["phi"].sum() - (f(np.ones(p, bool)) - f(np.zeros(p, bool)))) < 1e-9  # efficiency holds sample by sample
    # stopping rules
    g2 = game_from_f(f, p)
    out = sa.sampled_shapley(g2, np.random.default_rng(3), max_pairs=10 ** 5, ci_tol=0.05)
    assert out["stop"] == "ci_tol" and np.max(1.96 * out["se"]) <= 0.05 + 1e-12
    g3 = game_from_f(f, p)
    out = sa.sampled_shapley(g3, np.random.default_rng(3), max_pairs=10 ** 5, budget_evals=200)
    assert out["stop"] == "budget" and g3.ev.n_evals <= 200
    # truncated semi-value = mean marginal damage over coalition sizes 0..K-1: check against brute force for K=4
    K = 4
    g4 = game_from_f(f, p)
    out = sa.sampled_shapley(g4, np.random.default_rng(5), max_pairs=600, max_failed=K)
    vals = sa.full_table(g4)
    ref = np.zeros(p)
    for i in range(p):
        per = []
        for s in range(K):
            ds = [vals[S | (1 << i)] - vals[S] for S in range(2 ** p) if not (S >> i) & 1 and bin(S).count("1") == s]
            per.append(np.mean(ds))
        ref[i] = np.mean(per)
    assert (np.abs(out["phi"] - ref) / np.maximum(out["se"], 1e-12)).max() < 4.5


def test_cache_and_vectorised():
    f = lambda F: float(F.sum() ** 1.5)  # noqa: E731
    a, b = game_from_f(f, 8), game_from_f(f, 8, vectorised=True)
    ra = sa.sampled_shapley(a, np.random.default_rng(0), max_pairs=20)
    rb = sa.sampled_shapley(b, np.random.default_rng(0), max_pairs=20)
    assert np.allclose(ra["phi"], rb["phi"]) and a.ev.n_evals == b.ev.n_evals
    assert b.ev.n_batches < a.ev.n_batches and a.ev.n_requests > a.ev.n_evals  # cache hits and batching happened
    n0 = a.ev.n_evals
    a.value(np.zeros(8, bool))
    assert a.ev.n_evals == n0


def test_worst_k():
    # A alone = 3, B+C only together = 5: for k=2 greedy picks A (+anything)=3, the optimum is {B,C}=5
    p = 6
    f = lambda F: 3.0 * F[0] + 5.0 * (F[1] and F[2]) + 0.1 * F[3]  # noqa: E731
    g = game_from_f(f, p)
    ex = sa.worst_k_exhaustive(g, [1, 2, 3])
    gr = sa.worst_k_beam(g, 3, 1)
    bm = sa.worst_k_beam(g, 3, p)
    assert ex[1]["set"] == [1 - 1] and abs(ex[1]["value"] - 3.0) < 1e-12
    assert sorted(ex[2]["set"]) == [1, 2] and abs(ex[2]["value"] - 5.0) < 1e-12
    assert abs(gr[2]["value"] - 3.1) < 1e-12 and gr[2]["value"] < ex[2]["value"]  # greedy is provably suboptimal here
    assert abs(bm[2]["value"] - ex[2]["value"]) < 1e-12 and abs(bm[3]["value"] - ex[3]["value"]) < 1e-12
    # additive: greedy == exhaustive == sum of the k largest
    w = np.array([0.3, 2.0, 1.0, 0.1, 5.0, 0.7])
    g = game_from_f(lambda F: float(w[F].sum()), 6)
    gr, ex = sa.worst_k_beam(g, 6, 1), sa.worst_k_exhaustive(g, range(1, 7))
    for k in range(1, 7):
        assert abs(gr[k]["value"] - np.sort(w)[::-1][:k].sum()) < 1e-12 and abs(ex[k]["value"] - gr[k]["value"]) < 1e-12


def test_min_set():
    w = np.array([0.1, 0.2, 5.0, 0.3, 0.05, 4.0, 0.4, 0.15])  # fail as many as possible with total damage <= tol
    tol = 1.0
    g = game_from_f(lambda F: float(w[F].sum()), 8)
    gb = sa.greedy_backward(g, tol)
    # optimum: fail the cheapest sensors while sum<=1: 0.05+0.1+0.15+0.2+0.3 = 0.8 (5 sensors), +0.4 = 1.2 too much -> 5
    assert gb["min_available"] == 3 and abs(gb["min_value"] - 0.8) < 1e-12
    rb = sa.ranked_backward(g, np.argsort(w), tol)
    assert rb["min_available"] == 3
    bb = sa.beam_backward(g, tol, width=3)
    assert bb["min_available"] <= gb["min_available"]
    assert gb["curve"][0]["n_available"] == 8 and len(gb["curve"]) == 9  # curve recorded past the tolerance too
    # a non-additive case where beam can beat greedy: sensors 0,1 look cheap singly but jointly expensive
    f = lambda F: 0.4 * F[0] + 0.4 * F[1] + 3.0 * (F[0] and F[1]) + 0.5 * F[2] + 0.5 * F[3] + 0.45 * F[4]  # noqa: E731
    g = game_from_f(f, 5)
    gb, bb = sa.greedy_backward(g, 1.0), sa.beam_backward(g, 1.0, width=10)
    assert bb["min_available"] <= gb["min_available"]
    # exhaustive group curve
    vals = sa.full_table(game_from_f(lambda F: float(w[F].sum()), 8))
    cur = sa.exhaustive_group_curve(vals, 8)
    assert abs(cur[5]["best_value"] - 0.8) < 1e-12 and abs(cur[1]["worst_value"] - 5.0) < 1e-12


def test_full_pipeline_json_and_summary():
    inv = sa.default_inventory()
    rng = np.random.default_rng(0)
    wg = np.array([0.2, 1.0, 0.5, 0.5, 0.0, 0.8, 1.5, 0.9])           # per-group weight
    ws = wg[inv.group_of] / np.maximum(np.array(inv.counts)[inv.group_of], 1) * rng.uniform(0.5, 1.5, inv.n_sensors)

    def vf(mask):  # damage grows with lost weight but saturates; clean error 0.8
        return 0.8 + 6.0 * (1 - math.exp(-ws[~mask].sum() / 2.0))

    res = sa.run_full_analysis(vf, inv, tol=2.0, sensor_budget_evals=1500, sensor_max_failed=15, worst_k_max=4,
                               beam_width=3)
    s = json.dumps(res)
    back = json.loads(s)
    assert back["n_sensors"] == 115 and abs(back["baseline"] - 0.8) < 1e-12
    assert back["cost"]["unique_evals"] > 256
    # group Shapley efficiency
    assert abs(sum(res["group"]["shapley"]) - (res["all_groups_failed"] - res["baseline"])) < 1e-9
    # ground-truth ordering of the heaviest group is recovered
    gsum = [ws[inv.indices(g)].sum() for g in range(8)]  # damage is monotone in lost weight -> same ordering
    assert [r["index"] for r in res["group"]["ranking_shapley"]] == list(np.argsort(-np.array(gsum), kind="stable"))
    # exhaustive group optimum >= greedy and beam
    for k in range(1, 8):
        ex = res["group"]["worst_k_exhaustive"][k]["worst_value"]
        assert res["group"]["worst_k_greedy"][str(k)]["value"] <= ex + 1e-9 and res["group"]["worst_k_beam"][str(k)]["value"] <= ex + 1e-9
    assert res["min_set"]["sensors_greedy"]["min_available"] is not None
    assert "unsupported" in res["min_set"]["note"]
    md = sa.summary_tables(res, top=5)
    tex = sa.summary_tables(res, top=5, fmt="tex")
    assert md.count("|") > 50 and "\\toprule" in tex


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok", name)
    print(f"{n} tests passed")
