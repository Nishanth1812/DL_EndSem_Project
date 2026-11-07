"""Regression tests for the sensor-failure corruptions in pipeline_robustness.py.

Layout of a collated task_1-3 batch (checked on a real batch, see HARNESS_V2.md):
  branches 0-5 (flux_loop, ccbv, obr, obv, saddle_v, coil_i): (B, 1, 1, T, n_sensors) -> time axis -2, sensors axis -1
  branches 6-7 (solenoid, ip):                              (B, 1, 1, T)             -> time axis -1, one sensor
The pre-fix harness treated the LAST axis as time everywhere, so on the profile branches it zeroed sensor columns for
temporal gaps, forward-filled across sensors, and zeroed a single time step for channel/category/group ablations.

Run:  python -m pytest cnn_mlp/src/study/test_corruptions.py -q       (needs torch + numpy; TM_* env is not required)
Real-batch tests use REAL_BATCH (env, default: the local axis_check/batch.npz) and are skipped when it is absent.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import pipeline_robustness as R  # noqa: E402
from aggregate_seeds import bootstrap_ci  # noqa: E402

SHAPES = [(15, 25), (40, 25), (18, 25), (18, 25), (12, 250), (10, 20), (None, 20), (None, 20)]  # (n_sensors, T)
B = 4
REAL = os.environ.get("REAL_BATCH", r"C:\Users\Devab\AppData\Local\Temp\claude\94cae6ef419a3dd0\9b3dc88f-3a2c-4815-bbad-6ee91700a5da\scratchpad\axis_check\batch.npz")


def synth(seed=0):
    """Same layout as a real batch; every value non-zero and distinct per (sample, sensor, time) so any axis mix-up is visible."""
    rng = np.random.default_rng(seed)
    out = []
    for n, T in SHAPES:
        shape = (B, 1, 1, T) if n is None else (B, 1, 1, T, n)
        out.append(torch.as_tensor(rng.uniform(1.0, 2.0, shape) + np.arange(int(np.prod(shape))).reshape(shape) * 1e-3, dtype=torch.float32))
    return out


def real_batch():
    if not os.path.exists(REAL):
        pytest.skip("no real batch (set REAL_BATCH)")
    d = np.load(REAL)
    return [torch.from_numpy(d[f"x{i}"]) for i in range(8)]


@pytest.fixture(params=["synthetic", "real"])
def xs(request):
    return synth() if request.param == "synthetic" else real_batch()


def tax(a):
    return -2 if a.ndim >= 5 else -1


def np_(t):
    return t.numpy().astype(np.float64)


def test_layout_matches_helpers(xs):
    assert [t.ndim for t in xs] == [5, 5, 5, 5, 5, 5, 4, 4]
    for t in xs:
        assert R._time_axis(np_(t)) == tax(np_(t))


# ----------------------------------------------------------------------------- whole-branch loss
@pytest.mark.parametrize("n", [1, 3, 6])
@pytest.mark.parametrize("imp", ["zero", "mean", "ffill"])
def test_channel_ablation_zeroes_whole_branches(xs, n, imp):
    mean = {i: 0.0 for i in range(8)}  # mean-impute of a whole lost sensor with mean 0 == zero
    out, info = R.apply_scenario(xs, "channel_ablation", n, imp, mean, np.random.default_rng(3))
    dead = info["dead_channels"]
    assert len(dead) == n and dead == sorted(dead)
    for i, (a, b) in enumerate(zip(xs, out)):
        if i in dead:
            assert (b == 0).all(), f"branch {i} not fully zero (zeroed only a slice?)"
        else:
            assert torch.equal(a, b), f"healthy branch {i} was modified"


@pytest.mark.parametrize("group", ["flux_loop", "pickup_fields", "saddle_voltage", "active_coils", "solenoid", "plasma_current"])
def test_category_ablation(xs, group):
    chans = R.CATEGORY_GROUPS[group]
    out, info = R.apply_group(xs, "category_abl", 1, "zero", {}, np.random.default_rng(0), group, chans)
    assert info["dead_channels"] == chans
    for i, (a, b) in enumerate(zip(xs, out)):
        assert (b == 0).all() if i in chans else torch.equal(a, b)


@pytest.mark.parametrize("group", list(R.CORRELATED_GROUPS))
def test_group_failure(xs, group):
    chans = R.CORRELATED_GROUPS[group]
    out, info = R.apply_group(xs, "group_failure", 1, "zero", {}, np.random.default_rng(0), group, chans)
    assert info["dead_channels"] == chans
    for i, (a, b) in enumerate(zip(xs, out)):
        assert (b == 0).all() if i in chans else torch.equal(a, b)


# ----------------------------------------------------------------------------- temporal gaps
@pytest.mark.parametrize("scen,where", [("temporal_gap", "random"), ("disruption_proximate", "end")])
@pytest.mark.parametrize("frac", [0.2, 0.4, 0.6])
def test_gap_is_a_time_span_over_all_sensors(xs, scen, where, frac):
    out, info = R.apply_scenario(xs, scen, frac, "zero", {}, np.random.default_rng(5))
    assert info["dead_channels"] == []  # partial damage: the failure-aware model is not told
    for a, b in zip(xs, out):
        a, b = np_(a), np_(b)
        ax = tax(a)
        T = a.shape[ax]
        g = max(1, int(round(frac * T)))
        am, bm = np.moveaxis(a, ax, -1), np.moveaxis(b, ax, -1)  # time last
        changed_t = np.flatnonzero((am != bm).reshape(-1, T).any(0))
        if changed_t.size:  # (on real data a span could land on already-zero padding, so only check what changed)
            assert changed_t.max() - changed_t.min() < g
        assert (bm.reshape(-1, T)[:, changed_t] == 0).all()
        # everything outside a single contiguous span of length g is untouched, for EVERY sensor
        eq = (am == bm).reshape(-1, T).all(0)  # per time step: all sensors/samples identical
        assert (~eq).sum() <= g
        if where == "end" and changed_t.size:
            assert changed_t.max() == T - 1 or (am.reshape(-1, T)[:, T - g:] == 0).all(0).any()
        if a.ndim >= 5:  # no sensor column may be zeroed at times outside the gap
            assert (bm.reshape(-1, T)[:, eq] == am.reshape(-1, T)[:, eq]).all()


def test_gap_span_exact_on_synthetic():
    xs = synth()
    for where_scen in ("temporal_gap", "disruption_proximate"):
        out, _ = R.apply_scenario(xs, where_scen, 0.4, "zero", {}, np.random.default_rng(1))
        for a, b in zip(xs, out):
            a, b = np_(a), np_(b)
            ax = tax(a)
            T = a.shape[ax]
            g = int(round(0.4 * T))
            zero_t = np.flatnonzero((np.moveaxis(b, ax, -1).reshape(-1, T) == 0).all(0))
            assert len(zero_t) == g and zero_t.max() - zero_t.min() == g - 1
            if where_scen == "disruption_proximate":
                assert zero_t.max() == T - 1


@pytest.mark.parametrize("scen", ["temporal_gap", "disruption_proximate"])
def test_ffill_runs_along_time_per_sensor(scen):
    """Hole values must equal the last pre-gap sample OF THE SAME SENSOR (the old code ffilled across sensors)."""
    xs = synth()
    z, _ = R.apply_scenario(xs, scen, 0.4, "zero", {}, np.random.default_rng(2))
    f, _ = R.apply_scenario(xs, scen, 0.4, "ffill", {}, np.random.default_rng(2))
    for a, zz, ff in zip(xs, z, f):
        a, zz, ff = np_(a), np_(zz), np_(ff)
        ax = tax(a)
        am, zm, fm = (np.moveaxis(v, ax, 0) for v in (a, zz, ff))  # time first
        hole_t = np.flatnonzero((zm.reshape(zm.shape[0], -1) == 0).all(1))
        s, e = hole_t.min(), hole_t.max()
        for t in range(s, e + 1):
            if s == 0:
                assert (fm[t] == 0).all()  # nothing before the gap: stays 0
            else:
                assert np.array_equal(fm[t], am[s - 1])
        assert np.array_equal(fm[:s], am[:s]) and np.array_equal(fm[e + 1:], am[e + 1:])


def test_mean_impute_fills_only_holes():
    xs = synth()
    means = {i: 7.5 + i for i in range(8)}
    z, _ = R.apply_scenario(xs, "temporal_gap", 0.4, "zero", means, np.random.default_rng(2))
    m, _ = R.apply_scenario(xs, "temporal_gap", 0.4, "mean", means, np.random.default_rng(2))
    for i, (zz, mm) in enumerate(zip(z, m)):
        hole = (zz == 0)
        assert hole.any() and (mm[hole] == means[i]).all() and torch.equal(mm[~hole], zz[~hole])


# ----------------------------------------------------------------------------- random dropout
@pytest.mark.parametrize("sev", [0.1, 0.25, 0.5])
def test_random_dropout_fraction(sev):
    xs = synth()
    out, info = R.apply_scenario(xs, "random_dropout", sev, "zero", {}, np.random.default_rng(4))
    assert info["dead_channels"] == []
    for a, b in zip(xs, out):
        n = a.numel()
        assert (b == 0).sum().item() == int(round(sev * n))
        keep = b != 0
        assert torch.equal(a[keep], b[keep])


def test_random_dropout_ffill_along_time():
    xs = synth()
    z, _ = R.apply_scenario(xs, "random_dropout", 0.25, "zero", {}, np.random.default_rng(4))
    f, _ = R.apply_scenario(xs, "random_dropout", 0.25, "ffill", {}, np.random.default_rng(4))
    for a, zz, ff in zip(xs, z, f):
        a, zz, ff = np_(a), np_(zz), np_(ff)
        ax = tax(a)
        am, zm, fm = (np.moveaxis(v, ax, 0) for v in (a, zz, ff))
        exp = zm.copy()
        for t in range(am.shape[0]):
            h = zm[t] == 0
            exp[t][h] = exp[t - 1][h] if t > 0 else 0.0
        assert np.allclose(fm, exp)


# ----------------------------------------------------------------------------- failure-aware vs plain path
class _Spy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.health, self.n_args = [], []

    def forward(self, *a):
        self.n_args.append(len(a))
        if len(a) == 9:
            self.health.append(a[8].clone())
        return torch.zeros(a[0].shape[0], 1, 1, 65, 65)


def _loader():
    xs = synth()
    y = [torch.zeros(B, 1, 65, 65)]
    return [(["1"] * B, torch.arange(B), xs, y)]


def _run(fa, corrupt_fn):
    R.CFG["failure_aware"] = fa
    spy = _Spy()
    R.run_pass(_loader(), spy, torch.device("cpu"), 0.0, 1.0, corrupt_fn=corrupt_fn)
    return spy


@pytest.fixture(autouse=True)
def _restore_cfg():
    old = R.CFG["failure_aware"]
    yield
    R.CFG["failure_aware"] = old


def test_failure_aware_gets_health_mask_for_lost_sensors_only():
    train_means = {i: 0.0 for i in range(8)}
    ch = R.make_corrupt_fn(dict(scenario="channel_ablation", severity=3, impute="zero", group=None, chans=None), 7, train_means)
    spy = _run(True, ch)
    h = spy.health[0]
    assert h.shape == (B, 8) and int((h == 0).sum().item()) == 3 * B and (h.sum(1) == 5).all()
    grp = R.make_corrupt_fn(dict(scenario="group_failure", severity=1, impute="zero", group="magnetics_all", chans=[0, 1, 2, 3, 4]), 7, train_means)
    h = _run(True, grp).health[0]
    assert (h[:, :5] == 0).all() and (h[:, 5:] == 1).all()
    for scen in ("random_dropout", "temporal_gap", "disruption_proximate"):
        fn = R.make_corrupt_fn(dict(scenario=scen, severity=0.4, impute="zero", group=None, chans=None), 7, train_means)
        assert (_run(True, fn).health[0] == 1).all(), scen


def test_transformer_gets_per_sensor_mask(monkeypatch):
    monkeypatch.setitem(R.CFG, "model", "transformer")
    fn = R.make_corrupt_fn(dict(scenario="group_failure", severity=1, impute="zero", group="power_supply", chans=[5, 6]), 7, {})
    h = _run(True, fn).health[0]
    n_ch = [15, 40, 18, 18, 12, 10, 1, 1]
    assert h.shape == (B, sum(n_ch))  # 115 sensors
    assert (h[:, 103:114] == 0).all() and h.sum(1).eq(sum(n_ch) - 11).all()  # coil_i (10) + solenoid (1) lost; ip alive


def test_plain_model_gets_no_health_argument():
    fn = R.make_corrupt_fn(dict(scenario="channel_ablation", severity=3, impute="zero", group=None, chans=None), 7, {})
    spy = _run(False, fn)
    assert spy.n_args == [8] and spy.health == []
    assert _run(True, None).n_args == [9]  # clean pass of a failure-aware model: all-ones mask


# ----------------------------------------------------------------------------- misc harness helpers
def test_phys_window_selection_is_order_independent_and_about_1_in_stride():
    pairs = [(s, w) for s in range(12000, 12060) for w in range(100)]
    sel = [R.phys_selected(s, w, 4) for s, w in pairs]
    assert abs(np.mean(sel) - 0.25) < 0.03
    assert sel == [R.phys_selected(s, w, 4) for s, w in pairs]
    assert not any(R.phys_selected(s, w, 0) for s, w in pairs[:50])


def test_bootstrap_ci_brackets_the_mean_and_resamples_shots():
    rng = np.random.default_rng(0)
    shots = np.repeat(np.arange(30), 20)
    vals = rng.normal(2.0, 0.3, 30)[shots] + rng.normal(0, 0.05, shots.size)
    ci = bootstrap_ci(shots, vals)
    assert ci["lo"] < ci["mean"] < ci["hi"] and ci["n_shots"] == 30 and ci["n"] == 600
    assert bootstrap_ci([1, 1], [np.nan, np.nan])["n"] == 0


def test_rob_paths_honours_run_tag(monkeypatch):
    monkeypatch.delenv("TM_ROB_DIR", raising=False)
    old = R.CFG["run_tag"]
    try:
        R.CFG["run_tag"] = ""
        legacy = R.rob_paths("random")
        R.CFG["run_tag"] = "mlp_mask_pinn_s1"
        tagged = R.rob_paths("random")
    finally:
        R.CFG["run_tag"] = old
    assert "mlp_mask_pinn_s1" in tagged["dir"] and "mlp_mask_pinn_s1" in tagged["model_dir"]
    assert "mlp_mask_pinn_s1" not in legacy["dir"]
