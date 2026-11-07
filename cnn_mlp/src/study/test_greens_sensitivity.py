#!/usr/bin/env python
"""Assert-based tests for greens_sensitivity.py / sensor_geometry.py.  Run:  python test_greens_sensitivity.py"""
import math
import os
import sys

import numpy as np
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import greens_sensitivity as gs  # noqa: E402
import sensor_attribution as sa  # noqa: E402
import sensor_geometry as sg  # noqa: E402

MU0 = gs.MU0


def neumann_M(R, Z, Rc, Zc):
    """Mutual inductance of two coaxial circles by direct quadrature of the Neumann integral (independent of elliptic integrals)."""
    dz = Z - Zc
    f = lambda t: math.cos(t) / math.sqrt(R * R + Rc * Rc + dz * dz - 2 * R * Rc * math.cos(t))  # noqa: E731
    return MU0 * R * Rc * integrate.quad(f, 0, math.pi, limit=200)[0]


def biot_savart(R, Z, Rc, Zc):
    """(B_R, B_Z) per ampere of a ring at (Rc, Zc), field point (R, Z, phi=0): quadrature of the Biot-Savart law."""
    dz = Z - Zc

    def parts(t):  # source point (Rc cos t, Rc sin t, Zc), current along phi-hat(t); field point (R, 0, Z)
        rx, ry, rz = R - Rc * math.cos(t), -Rc * math.sin(t), dz
        r3 = (rx * rx + ry * ry + rz * rz) ** 1.5
        tx, ty = -math.sin(t), math.cos(t)
        # dl = Rc dt * (tx, ty, 0);  dB = mu0/(4 pi) dl x r / r^3
        bx = (ty * rz) * Rc / r3
        bz = (tx * ry - ty * rx) * Rc / r3
        return bx, bz

    bx = integrate.quad(lambda t: parts(t)[0], 0, 2 * math.pi, limit=200)[0]
    bz = integrate.quad(lambda t: parts(t)[1], 0, 2 * math.pi, limit=200)[0]
    return MU0 / (4 * math.pi) * bx, MU0 / (4 * math.pi) * bz


def test_on_axis_loop_closed_form():
    Rc, Zc = 0.7, 0.2
    for Z in (-1.0, 0.0, 0.2, 0.9):
        dz = Z - Zc
        ref = MU0 * Rc ** 2 / (2 * (Rc ** 2 + dz ** 2) ** 1.5)  # B_Z on the axis of a circular loop
        assert abs(gs.green_bz(1e-6, Z, Rc, Zc) - ref) / ref < 1e-9
        h = 1e-5  # paraxial: B_R = -(R/2) dB_Z/dz  (div B = 0)
        dBz = (gs.green_bz(1e-9, Z + h, Rc, Zc) - gs.green_bz(1e-9, Z - h, Rc, Zc)) / (2 * h)
        assert abs(gs.green_br(1e-4, Z, Rc, Zc) + 0.5e-4 * dBz) < 1e-6 * abs(0.5e-4 * dBz) + 1e-18
        # small pickup loop of radius a on axis: flux = pi a^2 B_Z
        a = 1e-3
        assert abs(2 * math.pi * gs.green_psi(a, Z, Rc, Zc) - math.pi * a * a * ref) / (math.pi * a * a * ref) < 1e-5
    assert gs.green_br(0.0, 0.3, Rc, Zc) == 0.0


def test_symmetry_and_neumann_quadrature():
    rng = np.random.default_rng(0)
    for _ in range(25):
        R, Rc = rng.uniform(0.05, 2.0, 2)
        Z, Zc = rng.uniform(-1.8, 1.8, 2)
        a, b = gs.green_psi(R, Z, Rc, Zc), gs.green_psi(Rc, Zc, R, Z)
        assert abs(a - b) <= 1e-12 * abs(a) + 1e-20  # G(a, b) = G(b, a)
        M = neumann_M(R, Z, Rc, Zc)
        assert abs(2 * math.pi * a - M) <= 1e-7 * abs(M) + 1e-18, (R, Z, Rc, Zc, 2 * math.pi * a, M)


def test_fields_vs_biot_savart_and_curl():
    rng = np.random.default_rng(1)
    for _ in range(15):
        R, Rc = rng.uniform(0.15, 1.9, 2)
        Z, Zc = rng.uniform(-1.8, 1.8, 2)
        if abs(R - Rc) < 0.05 and abs(Z - Zc) < 0.05:
            continue
        br, bz = biot_savart(R, Z, Rc, Zc)
        assert abs(gs.green_bz(R, Z, Rc, Zc) - bz) <= 1e-6 * (abs(bz) + abs(br)) + 1e-14
        assert abs(gs.green_br(R, Z, Rc, Zc) - br) <= 1e-6 * (abs(bz) + abs(br)) + 1e-14
        h = 1e-5
        dpsi_dR = (gs.green_psi(R + h, Z, Rc, Zc) - gs.green_psi(R - h, Z, Rc, Zc)) / (2 * h)
        dpsi_dZ = (gs.green_psi(R, Z + h, Rc, Zc) - gs.green_psi(R, Z - h, Rc, Zc)) / (2 * h)
        assert abs(dpsi_dR / R - gs.green_bz(R, Z, Rc, Zc)) < 1e-5 * (abs(bz) + abs(br)) + 1e-14   # B_Z = (1/R) dpsi/dR
        assert abs(-dpsi_dZ / R - gs.green_br(R, Z, Rc, Zc)) < 1e-5 * (abs(bz) + abs(br)) + 1e-14  # B_R = -(1/R) dpsi/dZ
    # vectorised broadcasting
    R = np.linspace(0.1, 1.9, 7)[None, :]
    Z = np.linspace(-1, 1, 5)[:, None]
    assert gs.green_psi(R, Z, 1.0, 0.0).shape == (5, 7) and np.isfinite(gs.green_br(R, Z, 1.0, 0.0)).all()


def test_geometry_real_data():
    geo = sg.load_geometry()
    inv = sa.default_inventory()
    assert geo.names[:3] == inv.sensor_names[:3] and len(geo.names) == 115
    assert all(geo.group[i] == inv.group_names[inv.group_of[i]] for i in range(115))
    cnt = {g: len(geo.group_indices(g)) for g in inv.group_names}
    assert [cnt[g] for g in inv.group_names] == [15, 40, 18, 18, 12, 10, 1, 1]
    pos = np.isfinite(geo.R)
    assert pos.sum() == 114 and not pos[inv.indices(7)[0]]            # only Ip has no position
    assert geo.R[pos].min() > 0.1 and geo.R[pos].max() < 2.0 and abs(geo.Z[pos]).max() < 2.0
    assert all("30421.zarr" in s["provenance"] or "global" in s["provenance"] for s in geo.raw["sensors"])
    # PF coil centroids vs the independent FreeGS MAST() positions: within 5 cm for P3-P5 (P2 inner/outer straddle the FreeGS P2)
    cc = sg.cross_check_coils(geo)
    assert all(cc[c] < 0.05 for c in ("P3L", "P3U", "P4L", "P4U", "P5L", "P5U")), cc
    assert all(cc[c] < 0.08 for c in ("P2IL", "P2IU", "P2OL", "P2OU")), cc
    # the hard-coded pipeline flux-loop pixels vs real loops: recorded, not asserted to agree
    cmp = sg.compare_pipeline_flux_loops(geo)
    assert len(cmp["rows"]) == 15 and cmp["median_d_same_index_m"] > cmp["grid_dR_m"]  # they do NOT coincide with real loops
    r, z = geo.pixel_positions()
    assert np.nanmax(r) <= 64 and np.nanmin(z) >= 0


def test_rank_agreement():
    rng = np.random.default_rng(0)
    a = rng.normal(size=40)
    out = gs.rank_agreement(a, a ** 3, n_perm=500)
    assert abs(out["spearman"] - 1) < 1e-12 and abs(out["kendall"] - 1) < 1e-12 and out["p_spearman"] < 0.01
    out = gs.rank_agreement(a, -a, n_perm=500)
    assert abs(out["spearman"] + 1) < 1e-12 and out["p_kendall"] < 0.01
    ps = [gs.rank_agreement(rng.normal(size=30), rng.normal(size=30), n_perm=200, seed=s)["p_spearman"] for s in range(40)]
    assert 0.02 < np.mean(ps) < 0.8 and np.mean(np.array(ps) < 0.05) < 0.25  # unrelated vectors: p roughly uniform
    # NaN pairs dropped
    b = a.copy()
    b[:5] = np.nan
    assert gs.rank_agreement(a, b, n_perm=100)["n"] == 35
    # known small case: tau for (1,2,3,4) vs (1,3,2,4) = 4/6
    assert abs(gs.rank_agreement([1, 2, 3, 4], [1, 3, 2, 4], n_perm=50)["kendall"] - 2 / 3) < 1e-12
    by = gs.rank_agreement_by_group(a, a, ["x"] * 20 + ["y"] * 20, n_perm=50)
    assert set(by) == {"all", "x", "y"} and abs(by["x"]["spearman"] - 1) < 1e-12


def test_error_map_vs_location():
    Ra, Za = sg.DEFAULT_R_AXIS, sg.DEFAULT_Z_AXIS
    RR, ZZ = np.meshgrid(Ra, Za)
    sR, sZ = 1.2, 0.4
    err = np.exp(-np.hypot(RR - sR, ZZ - sZ) / 0.3)             # error concentrated around the lost sensor
    out = gs.error_map_vs_location(err, sR, sZ, Ra, Za)
    assert out["spearman_distance"] < -0.99 and out["centroid_distance_m"] < 0.35
    rnd = gs.error_map_vs_location(np.random.default_rng(0).random(err.shape), sR, sZ, Ra, Za)
    assert abs(rnd["spearman_distance"]) < 0.1
    assert np.isnan(gs.error_map_vs_location(err, np.nan, np.nan, Ra, Za)["spearman_distance"])


def test_linear_gaussian_model_and_attribution_link():
    geo = sg.load_geometry()
    m = gs.build_model(geo, stride=4, ell=0.25)
    inv = sa.default_inventory()
    assert m.has_model.sum() == 103 and not m.has_model[inv.indices(4)].any()      # saddle voltages unmodelled
    allm = np.ones(115, bool)
    none = np.zeros(115, bool)
    assert abs(m.value_fn(none) - math.sqrt(m._prior)) < 1e-12
    # information never hurts: removing sensors can only increase the posterior variance (monotone along a random chain)
    rng = np.random.default_rng(0)
    chain, prev = allm.copy(), m.post_var(allm)
    for i in rng.permutation(115)[:40]:
        chain[i] = False
        cur = m.post_var(chain)
        assert cur >= prev - 1e-9 * (1 + prev)
        prev = cur
    assert m.value_fn(allm) < 0.5 * m.value_fn(none)
    s = m.scores()
    assert np.isfinite(s["loo_loss"][m.has_model]).all() and (s["loo_loss"][m.has_model] > -1e-9).all()
    assert np.isnan(s["loo_loss"][~m.has_model]).all()
    # plugs into sensor_attribution: group Shapley efficiency and a sampled sensor run
    ev = sa.Evaluator(m.value_fn)
    gg = sa.Game.groups(ev, inv)
    vals = sa.full_table(gg)
    phi = sa.exact_shapley(vals, 8)
    assert abs(phi.sum() - (vals[-1] - vals[0])) < 1e-9 and phi[4] >= -1e-12  # saddle group: zero effect
    assert abs(phi[4]) < 1e-9
    # the fully redundant duplicate sensor: identical row -> drop-one ~ small vs alone gain large, positive interaction
    A = np.vstack([m.A, m.A[0:1]])
    has = np.r_[m.has_model, True]
    noise = np.r_[m.noise_sd, m.noise_sd[0]]
    dup = gs.LinearGaussianModel(geo, has, A, m.T, m.Sigma_x, noise, m.n_pix, m.target_pts)
    base_all = dup.value_fn(np.ones(116, bool))
    only_a = np.ones(116, bool); only_a[115] = False
    only_b = np.ones(116, bool); only_b[0] = False
    both = np.ones(116, bool); both[[0, 115]] = False
    f = lambda F: dup.value_fn(~F)  # noqa: E731
    i2 = f(np.eye(116, dtype=bool)[0] | np.eye(116, dtype=bool)[115]) - f(np.eye(116, dtype=bool)[0]) \
        - f(np.eye(116, dtype=bool)[115]) + f(np.zeros(116, bool))
    assert i2 > 0, i2                                                    # backup pair -> positive interaction
    assert dup.value_fn(only_a) - base_all < 0.5 * (dup.value_fn(both) - base_all)  # losing one copy hurts much less than both


def test_saddle_option_duplicates():
    geo = sg.load_geometry()
    m = gs.build_model(geo, stride=4, include_saddle=True)
    inv = sa.default_inventory()
    sad = inv.indices(4)
    assert m.has_model.sum() == 115
    # 12 identical loops (same R, Z span, width) -> identical static rows, and they add information only once
    assert np.allclose(m.A[sad], m.A[sad[0]], rtol=1e-9)
    allm = np.ones(115, bool)
    one = allm.copy(); one[sad[1:]] = False
    assert m.post_var(one) <= m.post_var(np.where(np.isin(np.arange(115), sad), False, allm)) + 1e-12
    drop = allm.copy(); drop[sad[0]] = False
    assert m.value_fn(drop) - m.value_fn(allm) < 1e-3 * m.value_fn(allm)  # another identical loop covers for it


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok", name)
    print(f"{n} tests passed")
