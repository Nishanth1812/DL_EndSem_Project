"""CPU smoke tests for the models / variants / seeds / tuning in pipeline.py (synthetic tensors with the real shapes).

Run:  python cnn_mlp/src/test_pipeline_models.py      (or pytest; ~1 min on CPU, no data, no GPU)

pipeline.py needs the TokaMark checkout only lazily (setup_imports), so we point TM_TOKAMARK / TM_BASELINE at the copy in
runs/lstm_fa and stub `torchinfo` (imported by the baseline model files, only used for summaries).
"""
import os, sys, json, types, subprocess, tempfile, warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
os.environ.setdefault("TM_TOKAMARK", os.path.join(REPO, "runs", "lstm_fa", "tokamark"))
os.environ.setdefault("TM_BASELINE", os.path.join(REPO, "runs", "lstm_fa", "tokamark_baseline"))
for _k in ("TM_VARIANT", "TM_RUN_TAG", "TM_TUNED", "TM_MODEL", "TM_FAILURE_AWARE", "TM_PINN_LAMBDA", "TM_SEED"):
    os.environ.pop(_k, None)
if "torchinfo" not in sys.modules:
    try:
        import torchinfo  # noqa: F401
    except ImportError:
        sys.modules["torchinfo"] = types.SimpleNamespace(summary=lambda *a, **k: None)
sys.path.insert(0, HERE)

import numpy as np
import torch
import pipeline as P

CWD = os.getcwd()
CFG = P.CFG
# per-sample shapes of the 8 input groups (flux loops ... Ip), 115 sensors; target psi 65x65 (from a real Kaggle log)
IN_SHAPES = [(1, 1, 25, 15), (1, 1, 25, 40), (1, 1, 25, 18), (1, 1, 25, 18), (1, 1, 250, 12), (1, 1, 20, 10), (1, 1, 20), (1, 1, 20)]
N_SENSORS = 115
STATS = dict(mu_f=-0.25546626077270007, sd_f=0.382853499608823, mu_p=-0.026102678542290144, sd_p=0.054930705723548384)  # random split yaml
META = {"output": {"equilibrium-psi": {"mean": STATS["mu_p"], "std": STATS["sd_p"]}},
        "input": {"magnetics-flux_loop_flux": {"mean": STATS["mu_f"], "std": STATS["sd_f"]}},
        "task_window_segmenter": {"output_length": 0.005}}
SHAPES = (IN_SHAPES, [], [(1, 65, 65)])
FAKE_LOADER = types.SimpleNamespace(dataset=[{"shot_id": 0, "input": [np.zeros(s, np.float32) for s in IN_SHAPES],
                                              "exogenous": [], "y": [np.zeros((1, 65, 65), np.float32)]}])
B = 4


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = [torch.randn((B,) + s, generator=g) for s in IN_SHAPES]
    y = [torch.randn(B, 1, 65, 65, generator=g)]
    return x, y


def configure(model, variant, **kw):
    fa = variant != "plain"
    CFG.update(model=model, failure_aware=fa, variant=variant, pinn_lambda=0.1 if variant == "mask_pinn" else 0.0,
               fail_aug_p=1.0, dropout=0.0, d_model=64, tx_layers=3, tx_heads=4, seed=23, run_tag="", pinn_mode="measured")
    CFG.update(kw)
    P.set_flux_loop_stats(META)


def build(model, variant, **kw):
    configure(model, variant, **kw)
    torch.manual_seed(CFG["seed"])
    m = P.make_model(FAKE_LOADER, META, shapes=SHAPES)
    os.chdir(CWD)   # setup_imports() chdir()s into the baseline checkout
    return m


def crit():
    from src.trainer import MultiOutputMSELoss
    return MultiOutputMSELoss()


def test_all_models_forward_backward():
    counts = {}
    for model in ("mlp", "lstm", "transformer"):
        for variant in ("plain", "mask", "mask_pinn"):
            m = build(model, variant)
            m.train()
            x, y = batch()
            loss, mse, pinn = P.train_loss(m, crit(), x, y)
            assert torch.isfinite(loss), (model, variant)
            assert (pinn is None) == (variant != "mask_pinn"), (model, variant)
            loss.backward()
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters()), (model, variant)
            m.eval()
            with torch.no_grad():
                out = m(*x)
            assert out[0].shape == (B, 1, 65, 65), out[0].shape
            counts[(model, variant)] = P.count_params(m)
    for model in ("mlp", "lstm", "transformer"):
        print(f"  params {model:12s} plain={counts[(model, 'plain')]:>9,}  failure-aware={counts[(model, 'mask')]:>9,}")
    r = counts[("transformer", "plain")] / counts[("mlp", "plain")]
    assert 0.5 < r < 3.0, f"transformer is not parameter-comparable to the CNN+MLP (ratio {r:.2f})"


def test_transformer_drops_masked_sensors():
    m = build("transformer", "mask").eval()
    x, _ = batch()
    offs = np.cumsum([0] + [int(np.prod(s[3:])) if len(s) == 4 else 1 for s in IN_SHAPES])
    assert offs[-1] == N_SENSORS
    mask = torch.ones(B, N_SENSORS)
    dead = [0, 3, 14, 15 + 7, 100, 114]           # scattered sensors in several groups, incl. both scalar groups
    mask[:, dead] = 0.0
    with torch.no_grad():
        ref = m(*x, mask)[0]
        # perturb every masked sensor wildly: output must not move
        x2 = [t.clone() for t in x]
        for c in dead:
            g = int(np.searchsorted(offs, c, side="right") - 1)
            n = c - offs[g]
            if x2[g].dim() == 5:
                x2[g][..., n] = 1e3 * torch.randn_like(x2[g][..., n])
            else:
                x2[g] = 1e3 * torch.randn_like(x2[g])
        d_dead = (m(*x2, mask)[0] - ref).abs().max().item()
        # control: perturbing a LIVE sensor (group 1, sensor 5) changes the output by a much larger amount
        x3 = [t.clone() for t in x]
        x3[1][..., 5] += 30.0
        d_live = (m(*x3, mask)[0] - ref).abs().max().item()
        print(f"  max |dy|: masked-sensor perturbation {d_dead:.2e}, live-sensor perturbation {d_live:.2e}")
        assert d_dead < 1e-6 and d_live > 1e3 * max(d_dead, 1e-9), (d_dead, d_live)
        # the group-level (B, 8) mask equals the expanded per-sensor mask
        gm = torch.ones(B, 8); gm[:, 4] = 0.0
        cm = torch.ones(B, N_SENSORS); cm[:, offs[4]:offs[5]] = 0.0
        assert torch.allclose(m(*x, gm)[0], m(*x, cm)[0], atol=1e-6)
        # no mask -> exactly-zero sensors are inferred as missing (same rule as the failure-aware CNNs)
        xz = [t.clone() for t in x]; xz[4].zero_()
        assert torch.allclose(m(*xz)[0], m(*x, gm)[0], atol=1e-5)
        # everything lost -> still finite (always-valid CLS token)
        assert torch.isfinite(m(*x, torch.zeros(B, N_SENSORS))[0]).all()


def test_pinn_exactly_zero_and_absent_for_mask_variant():
    m = build("transformer", "mask")
    x, y = batch()
    real = P.flux_loop_pinn_loss
    P.flux_loop_pinn_loss = lambda *a, **k: (_ for _ in ()).throw(AssertionError("PINN evaluated for variant mask"))
    try:
        loss, mse, pinn = P.train_loss(m.train(), crit(), x, y)
    finally:
        P.flux_loop_pinn_loss = real
    assert pinn is None and loss is mse
    m2 = build("transformer", "mask_pinn")
    loss, mse, pinn = P.train_loss(m2.train(), crit(), x, y)
    assert pinn is not None and pinn.item() > 0 and abs(loss.item() - (mse.item() + 0.1 * pinn.item())) < 1e-5
    assert P.CFG["pinn_lambda"] == 0.1 and P.CFG["pinn_mode"] == "measured"


def test_mask_pinn_uses_measured_and_legacy_still_available():
    x, y = batch()
    real_l, real_m = P.flux_loop_pinn_loss, P.flux_loop_pinn_loss_measured
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong PINN mode evaluated"))  # noqa: E731
    try:
        for model in ("transformer", "mlp"):
            m = build(model, "mask_pinn").train()
            P.flux_loop_pinn_loss = boom                                    # default measured: legacy must not run
            loss, mse, pinn = P.train_loss(m, crit(), x, y)
            assert pinn is not None and torch.isfinite(pinn)
            P.flux_loop_pinn_loss = real_l
            m = build(model, "mask_pinn", pinn_mode="legacy").train()
            P.flux_loop_pinn_loss_measured = boom                           # legacy: measured must not run
            loss, mse, pinn = P.train_loss(m, crit(), x, y)
            assert pinn is not None and torch.isfinite(pinn)
            P.flux_loop_pinn_loss_measured = real_m
    finally:
        P.flux_loop_pinn_loss, P.flux_loop_pinn_loss_measured = real_l, real_m
        CFG["pinn_mode"] = "measured"


def test_flux_loop_bilinear_and_standardisation_synthetic():
    # bilinear sampling is exact for a field linear in (R, Z) and differentiable
    R = 0.06 + 0.03 * torch.arange(65.0)
    Z = -2.0 + 0.0625 * torch.arange(65.0)
    psi = 0.3 * R[None, :] - 0.1 * Z[:, None] + 0.02                         # [z_row, r_col]
    got = P.flux_loop_sample(psi[None, None].requires_grad_())
    want = torch.tensor([0.3 * r - 0.1 * z + 0.02 for r, z in P.FLUX_LOOP_RZ], dtype=torch.float32)
    assert torch.allclose(got[0], want, atol=1e-5), (got[0] - want).abs().max()
    got.sum().backward()
    # a window whose loops measure exactly 2*pi*psi gives target == standardised psi at the loops, so the loss is 0
    S = P._FL_STATS
    psi_raw = psi * 0.05                                                       # some Wb/rad field
    psi_std = ((psi_raw - S["mu_p"]) / S["sd_p"])[None, None]
    raw_loops = 2 * np.pi * P.flux_loop_sample(psi_raw[None, None])             # (1, 15) Wb
    x0 = ((raw_loops - S["mu_f"]) / S["sd_f"]).reshape(1, 1, 1, 1, 15).repeat(1, 1, 1, 25, 1)
    tgt, alive = P.flux_loop_measured_target(x0)
    assert alive.all() and torch.allclose(tgt, P.flux_loop_sample(psi_std), atol=1e-4)
    assert P.flux_loop_pinn_loss_measured(psi_std, x0, torch.ones(1, 15)).item() < 1e-8
    # availability: failure-masked loop, NaN-filled (all-zero) loop and ~0-reading dead loop carry no term
    x1 = x0.clone(); x1[..., 2] = 0.0                                          # NaN-filled
    x1[..., 4] = (0.0 - S["mu_f"]) / S["sd_f"]                                 # reads 0 Wb: dead
    _, al = P.flux_loop_measured_target(x1)
    assert al[0].tolist() == [i not in (2, 4) for i in range(15)]
    bad = psi_std + 1.0
    full = P.flux_loop_pinn_loss_measured(bad, x0, torch.ones(1, 15)).item()
    h = torch.ones(1, 15); h[0, :5] = 0.0
    part = P.flux_loop_pinn_loss_measured(bad, x0, h).item()
    assert full > part > 0 and P.flux_loop_pinn_loss_measured(bad, x0, torch.zeros(1, 15)).item() == 0.0


def _real_zarr_dir():
    d = os.environ.get("TM_TEST_ZARR_DIR", os.path.join(os.environ.get("TEMP", ""), "claude", "94cae6ef419a3dd0",
                       "9b3dc88f-3a2c-4815-bbad-6ee91700a5da", "scratchpad", "data", "random"))
    return d if os.path.isdir(d) else None


def test_measured_flux_vs_efit_on_real_shot():
    """Real shot (skipped when the local zarr data or zarr is absent): measured loop flux ~ 2*pi*EFIT psi at the loop."""
    d = _real_zarr_dir()
    if d is None:
        print("  SKIP real-shot check (no local zarr data; set TM_TEST_ZARR_DIR)"); return
    sd = os.path.join(HERE, "study")
    sys.path.insert(0, sd)
    extra = os.path.join(d, "..", "..", "pylib")          # zarr lives next to the data in the scratch layout
    if os.path.isdir(extra):
        sys.path.insert(0, extra)
    try:
        import efit_truth as E
        import zarr  # noqa: F401
    except Exception as ex:  # noqa
        print("  SKIP real-shot check (", repr(ex)[:80], ")"); return
    shots = [f for f in ("12811.zarr", "13222.zarr", "12848.zarr") if os.path.isdir(os.path.join(d, f))]
    if not shots:
        print("  SKIP real-shot check (no 12811/13222/12848 zarr)"); return
    S = P._FL_STATS
    errs = []
    for f in shots:
        path = os.path.join(d, f)
        tr = E.load_efit_truth(path)
        g = E.open_shot(path)
        tt, vv = E._time_of(g, "magnetics", "flux_loop_flux")
        for w in E.enumerate_windows(path, truth=tr)[:25]:
            end = int(round((w["t_cut"] - tt[0]) / 0.0002))
            if end - 25 < 0 or end > vv.shape[1]:
                continue
            seg = np.nan_to_num(vv[:, end - 25:end].T, nan=0.0)                  # (25, 15) raw Wb, like the collate fn
            x0 = torch.tensor(((seg - S["mu_f"]) / S["sd_f"])[None, None, None], dtype=torch.float32)
            psi_raw = tr["psi"][w["slice_index"]]
            if np.isnan(psi_raw).any():
                continue
            psi_std = torch.tensor((psi_raw - S["mu_p"]) / S["sd_p"], dtype=torch.float32)[None, None]
            tgt, alive = P.flux_loop_measured_target(x0)
            e = (tgt - P.flux_loop_sample(psi_std))[0, :5][alive[0, :5]]         # inboard CC loops: tight
            errs.extend(e.tolist())
    errs = np.array(errs)
    print(f"  real shots {shots}: inboard loops converted-measured minus EFIT psi, standardised units: "
          f"mean {errs.mean():+.4f} rms {np.sqrt((errs ** 2).mean()):.4f} (n={len(errs)})")
    assert len(errs) > 20 and abs(errs.mean()) < 0.05 and np.sqrt((errs ** 2).mean()) < 0.08


def test_augmentation_masks():
    torch.manual_seed(0)
    x, _ = batch()
    xa, gm = P.apply_structured_failure(x, p_fail=1.0)                  # legacy group-level contract
    assert gm.shape == (B, 8) and [t.shape for t in xa] == [t.shape for t in x]
    xc, cm = P.apply_structured_failure(x, p_fail=1.0, per_channel=True)
    assert cm.shape == (B, N_SENSORS) and (cm.sum(1) < N_SENSORS).all()
    assert P.channel_layout(x) == [15, 40, 18, 18, 12, 10, 1, 1]
    for b in range(B):                                                   # every masked sensor really lost data
        i = 0
        for g, t in enumerate(xc):
            for n in range(P.channel_layout(x)[g]):
                if cm[b, i] == 0:
                    v = t[b, ..., n] if t.dim() == 5 else t[b]
                    assert (v == 0).any()
                else:
                    assert torch.equal(v := (t[b, ..., n] if t.dim() == 5 else t[b]), (x[g][b, ..., n] if t.dim() == 5 else x[g][b]))
                i += 1
    _, cm0 = P.apply_structured_failure(x, p_fail=0.0, per_channel=True)
    assert (cm0 == 1).all()


def test_seeds_change_weights_not_shots():
    a = build("transformer", "mask", seed=1); b = build("transformer", "mask", seed=2)
    assert not torch.equal(a.sensor_emb.weight, b.sensor_emb.weight)
    c = build("transformer", "mask", seed=1)
    assert torch.equal(a.sensor_emb.weight, c.sensor_emb.weight)
    # shot selection depends on TM_SELECT_SEED only, and every command reads the one shared selection.json
    def sel_seed(**env):
        e = {**os.environ, **env}
        r = subprocess.run([sys.executable, "-W", "ignore", "-c", "import pipeline as p;print(p.CFG['select_seed'], p.CFG['seed'])"],
                           cwd=HERE, env=e, capture_output=True, text=True, check=True)
        return r.stdout.split()
    assert sel_seed(TM_SEED="1")[0] == sel_seed(TM_SEED="2")[0] == "23"
    assert sel_seed(TM_SEED="5")[1] == "5"
    sels = set()
    for seed in (1, 2):
        for tag in ("", "auto", "foo"):
            CFG.update(seed=seed, run_tag=("transformer_mask_s%d" % seed) if tag == "auto" else tag)
            sels.add(P.paths("random")["sel"])
    assert len(sels) == 1, sels
    CFG.update(run_tag="")
    assert P.paths("random")["out"] == os.path.join(CFG["work"], "out", "random")        # legacy layout untouched
    CFG.update(run_tag="transformer_mask_s1")
    assert P.paths("random")["model_dir"].replace("\\", "/").endswith("random/transformer_mask_s1/model/")
    CFG.update(run_tag="")


def test_variant_env_and_auto_tag():
    def run(**env):
        e = {k: v for k, v in os.environ.items() if not k.startswith("TM_VARIANT")}
        e.update(env)
        code = "import pipeline as p;c=p.CFG;print(c['variant'],c['failure_aware'],c['pinn_lambda'],c['run_tag'])"
        return subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=HERE, env=e, capture_output=True, text=True, check=True).stdout.split()
    assert run(TM_VARIANT="plain", TM_MODEL="mlp", TM_RUN_TAG="auto") == ["plain", "False", "0.0", "mlp_plain_s23"]
    assert run(TM_VARIANT="mask", TM_MODEL="transformer", TM_SEED="3", TM_RUN_TAG="auto") == ["mask", "True", "0.0", "transformer_mask_s3"]
    assert run(TM_VARIANT="mask_pinn", TM_RUN_TAG="auto")[:3] == ["mask_pinn", "True", "0.1"]
    assert run(TM_VARIANT="mask_pinn", TM_RUN_TAG="auto", TM_PINN_MODE="legacy")[3] == "mlp_mask_pinn_legacy_s23"
    assert run(TM_VARIANT="mask_pinn", TM_RUN_TAG="auto")[3] == "mlp_mask_pinn_s23"
    assert run(TM_FAILURE_AWARE="1")[:3] == ["mask_pinn", "True", "0.1"]          # legacy pair: unchanged behaviour
    assert run()[:2] == ["plain", "False"]


def test_tune_orchestration_and_tuned_file():
    work = tempfile.mkdtemp()
    seen = {}
    real = P.cmd_train

    def fake_train(split):
        key = CFG["model"]
        seen.setdefault(key, []).append((CFG["lr"], CFG["weight_decay"], CFG["dropout"], CFG["d_model"], CFG["max_steps"], CFG["run_tag"]))
        n = len(seen[key])
        return dict(best_val=1.0 - 0.01 * (n if n != 4 else 0), steps=CFG["max_steps"], params=1)
    P.cmd_train = fake_train
    keep = dict(CFG)
    try:
        CFG.update(work=work, tune_trials=6, tune_steps=11)
        outs = {}
        for model in ("mlp", "transformer"):
            configure(model, "plain", work=work, tune_trials=6, tune_steps=11)
            outs[model] = P.cmd_tune("random")
            assert CFG["lr"] == 1e-4 and CFG["seed"] == 23 and CFG["run_tag"] == "" and CFG["max_steps"] == keep["max_steps"]
    finally:
        P.cmd_train = real
        CFG.update(keep)
    # same sampled (lr, wd, dropout) list for every model; d_model only searched for the transformer
    assert [t[:3] for t in seen["mlp"]] == [t[:3] for t in seen["transformer"]] and len(seen["mlp"]) == 6
    assert "d_model" not in outs["mlp"]["chosen"] and outs["transformer"]["chosen"]["d_model"] in (64, 128)
    assert outs["mlp"]["chosen_trial"] == 5          # fake val losses decrease with the trial index
    path = os.path.join(work, "out", "random", "tune", "transformer_plain", "tuning.json")
    t = json.load(open(path))
    assert t["chosen"] == outs["transformer"]["chosen"] and len(t["trials"]) == 6 and "validation" in t["selection"]
    # TM_TUNED is consumed at import: explicit env > tuned file > default
    e = {**os.environ, "TM_TUNED": path}
    code = "import pipeline as p;print(p.CFG['lr'],p.CFG['weight_decay'],p.CFG['dropout'],p.CFG['d_model'])"
    got = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=HERE, env=e, capture_output=True, text=True, check=True).stdout.split()
    c = t["chosen"]
    assert [float(g) for g in got] == [c["lr"], c["weight_decay"], c["dropout"], c["d_model"]]
    e["TM_LR"] = "0.5"
    got = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=HERE, env=e, capture_output=True, text=True, check=True).stdout.split()
    assert float(got[0]) == 0.5


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as ex:  # noqa
                import traceback; traceback.print_exc()
                print("FAIL", name, repr(ex)[:300]); fails += 1
    sys.exit(1 if fails else 0)
