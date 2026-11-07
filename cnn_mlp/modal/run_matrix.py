#!/usr/bin/env python
"""Generate RUN_MATRIX.md: job list, GPU-hour totals and Kaggle-A / Kaggle-B / Modal assignment.

    python run_matrix.py            # writes RUN_MATRIX.md next to this file
    python run_matrix.py --check    # self-check only (asserts), writes nothing

Pure stdlib. Every number that is not measured is tagged (assumed)/(extrapolated) in RUN_MATRIX.md.
Edit the constants below, re-run, and the plan updates.
"""
import math
import os
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- durations (hours, one T4)
# MEASURED (user brief, Kaggle 2xT4; the run summary shows the pipeline chose a single GPU, so ~= 1 T4):
#   plain CNN+MLP 155 min; failure-aware(+PINN) CNN+MLP ~190 min (194.4 in train_summary.json);
#   failure-aware(+PINN) CNN+LSTM ~190 min; 4,200 steps / 53 epochs with early stopping.
# EXTRAPOLATED / ASSUMED: everything else (see NOTES).
TRAIN_H = {
    ("cnn_mlp", "plain"): 155 / 60,
    ("cnn_mlp", "mask"): 3.0,          # assumed: between plain and mask+PINN (mask costs more than plain; PINN term small)
    ("cnn_mlp", "mask_pinn"): 190 / 60,
    ("cnn_lstm", "plain"): 2.6,        # extrapolated: LSTM mask+PINN 190 min x (MLP plain/MLP mask+PINN = 155/190)
    ("cnn_lstm", "mask"): 3.0,         # assumed
    ("cnn_lstm", "mask_pinn"): 190 / 60,
    ("transformer", "plain"): 3.4,     # extrapolated: 1.3 x CNN+LSTM (no measurement exists; +/-50 %)
    ("transformer", "mask"): 3.9,
    ("transformer", "mask_pinn"): 4.1,
}
POST_H = 0.25          # assumed: test-set eval + plots + summary after training, per run
ROB_H = 1.5            # user brief: robustness study (63 passes) ~1-2 h; midpoint. + physics_study not included
TUNE_TRIALS = 6        # pipeline.py CFG default TM_TUNE_TRIALS=6 (in progress by another engineer) ...
TUNE_TRIAL_H = 0.25    # ... x TM_TUNE_STEPS=300 steps at ~2.7 s/step (190 min / 4,200 steps) = 14 min + data/model build: derived, assumed
TUNE_MIN_FRAC = 0.5    # minimal scenario halves the tuning budget

# ----------------------------------------------------------------------------- budgets
KAGGLE_WEEKLY_H = 30.0     # per account (user brief); counts session wall-clock, not GPUs
SESSION_H = 9.0            # conservative end of "~9-12 h"
SESSION_OVERHEAD_DATA = 0.4   # assumed: pip + clone + untar a saved dataset onto local disk
SESSION_OVERHEAD_NODATA = 2.0  # measured-ish: notebook default TM_PREP_DEADLINE_H=2.0 (re-download ~40 GB from HF)
MODAL_CREDIT_USD = 20.0    # interpretation A: "20 hrs" = $20 of credit (Modal sells dollars, not hours)
MODAL_RESERVE = 0.15       # keep 15 % of credit unspent: retries after preemption, image builds, estimate error
MODAL_CREDIT_HOURS_LITERAL = 20.0  # interpretation B: 20 literal T4-hours of everything
# Modal list prices, USD per second (https://modal.com/pricing, fetched 2026-11-04)
PRICE_S = dict(T4=0.000164, L4=0.000222, A10G=0.000306, L40S=0.000542, A100_40=0.000583, A100_80=0.000694, H100=0.001097)
CPU_CORE_S = 0.0000131     # per physical core (2 vCPU)
MEM_GIB_S = 0.00000222
MODAL_CPU, MODAL_MEM_GIB = 4, 16   # what app.py requests per GPU job
PREP_H_PER_SPLIT = 2.5     # assumed: HF range-request download of ~40 GB + tar to volume; CPU-only
SMOKE_H = 0.3              # assumed single smoke test on a T4

MODELS = ["cnn_mlp", "cnn_lstm", "transformer"]
VARIANTS = ["plain", "mask", "mask_pinn"]


def rate(gpu="T4", cpu=MODAL_CPU, mem=MODAL_MEM_GIB):
    """USD per hour, all-in (GPU + CPU + memory)."""
    return 3600 * (PRICE_S[gpu] + cpu * CPU_CORE_S + mem * MEM_GIB_S)


CPU_ONLY_RATE = 3600 * (MODAL_CPU * CPU_CORE_S + MODAL_MEM_GIB * MEM_GIB_S)


@dataclass
class Job:
    id: str
    kind: str                 # prep | smoke | tune | train | rob
    model: str = ""
    variant: str = ""
    seed: int = 0
    split: str = "random"
    hours: float = 0.0
    tier: int = 0
    after: str = ""           # rob jobs run where their train job ran (needs best_model.pt)
    prefer_modal: bool = False
    bin: str = ""
    note: str = ""
    extra: list = field(default_factory=list)


def train_job(model, variant, seed, split, tier):
    return Job(f"train {split} {model}/{variant}/s{seed}", "train", model, variant, seed, split,
               TRAIN_H[(model, variant)] + POST_H, tier, prefer_modal=(split == "temporal"))


def rob_job(tj, tier):
    return Job(f"rob {tj.split} {tj.model}/{tj.variant}/s{tj.seed}", "rob", tj.model, tj.variant, tj.seed, tj.split,
               ROB_H, tier, after=tj.id, prefer_modal=tj.prefer_modal)


def tune_jobs(frac=1.0):
    return [Job(f"tune {m}", "tune", m, "plain", 0, "random", TUNE_TRIALS * TUNE_TRIAL_H * frac, 1, prefer_modal=True,
                note="tune on `plain`; the chosen hyper-parameters are reused for mask and mask_pinn (equal tuning)")
            for m in MODELS]


def scenario(name):
    """Return the ordered job list (priority order = list order, tier is the cut group)."""
    tj = {}
    jobs = []

    def T(model, variant, seed, split, tier):
        j = train_job(model, variant, seed, split, tier)
        tj[(model, variant, seed, split)] = j
        jobs.append(j)
        return j

    def R(model, variant, seed, split, tier):
        jobs.append(rob_job(tj[(model, variant, seed, split)], tier))

    jobs += tune_jobs(TUNE_MIN_FRAC if name == "minimal" else 1.0)
    # tier 2: the ablation table, seed 1, random split, headline model first
    for m in MODELS:
        for v in VARIANTS:
            T(m, v, 1, "random", 2)
    # tier 3: robustness on the two ends of the ablation (plain vs full) -> the robustness figure
    for m in MODELS:
        R(m, "mask_pinn", 1, "random", 3)
    if name == "minimal":
        for v in ("plain", "mask_pinn"):          # headline pair gets mean +/- std
            for s in (2, 3):
                T("cnn_mlp", v, s, "random", 4)
        return jobs
    for m in MODELS:
        R(m, "plain", 1, "random", 3)
    # tier 4: seeds 2,3 -> mean +/- std for every cell
    for s in (2, 3):
        for m in MODELS:
            for v in VARIANTS:
                T(m, v, s, "random", 4)
    # tier 5: temporal replica of plain vs full per model, + robustness of the remaining mask runs
    for m in MODELS:
        for v in ("plain", "mask_pinn"):
            T(m, v, 1, "temporal", 5)
    for m in MODELS:
        R(m, "mask", 1, "random", 5)
    if name == "target":
        return jobs
    # stretch: seeds 4,5 everywhere, temporal mask rows, temporal robustness
    for s in (4, 5):
        for m in MODELS:
            for v in VARIANTS:
                T(m, v, s, "random", 6)
    for m in MODELS:
        T(m, "mask", 1, "temporal", 6)
    for m in MODELS:
        R(m, "mask_pinn", 1, "temporal", 6)
        R(m, "plain", 1, "temporal", 6)
    return jobs


def modal_cap_h(jobs_prep_splits):
    """Usable Modal GPU-hours after prep (CPU-only) and one smoke test; interpretation A ($)."""
    fixed = jobs_prep_splits * PREP_H_PER_SPLIT * CPU_ONLY_RATE + SMOKE_H * rate()
    return max(0.0, (MODAL_CREDIT_USD * (1 - MODAL_RESERVE) - fixed) / rate()), fixed


def pack(job_hours, overhead):
    """First-fit pack bundles (priority order) into sessions of SESSION_H incl. overhead. Returns session wall list."""
    sess = []
    for h in job_hours:
        assert h + overhead <= SESSION_H + 1e-9, f"bundle {h:.2f} h cannot fit a {SESSION_H} h session"
        for i, used in enumerate(sess):
            if used + h <= SESSION_H:
                sess[i] += h
                break
        else:
            sess.append(overhead + h)
    return sess


def assign(jobs, cap):
    """Greedy: bundle = train job + its rob jobs. Modal-preferred first, then least-loaded Kaggle account,
    then spill the lowest-priority Kaggle bundles to Modal while credit remains."""
    robs = {}
    for j in jobs:
        if j.kind == "rob":
            robs.setdefault(j.after, []).append(j)
    bundles = []
    for j in jobs:
        if j.kind in ("train", "tune"):
            bundles.append([j] + robs.get(j.id, []))
    used = 0.0
    load = {"Kaggle A": 0.0, "Kaggle B": 0.0}
    hrs = lambda b: sum(x.hours for x in b)
    for b in bundles:
        if b[0].prefer_modal and used + hrs(b) <= cap:
            used += hrs(b)
            for x in b:
                x.bin = "Modal"
    for b in bundles:
        if not b[0].bin:
            k = min(load, key=load.get)
            load[k] += hrs(b)
            for x in b:
                x.bin = k
    for b in sorted([b for b in bundles if b[0].bin != "Modal"], key=lambda b: -b[0].tier):
        if used + hrs(b) <= cap:
            used += hrs(b)
            for x in b:
                x.bin = "Modal"
    # preferred-modal bundles that did not fit fall back to Kaggle
    for b in bundles:
        if not b[0].bin:
            k = min(load, key=load.get)
            for x in b:
                x.bin = k
    return used


def summarize(name):
    jobs = scenario(name)
    splits = {j.split for j in jobs if j.kind != "tune"}
    cap, fixed = modal_cap_h(len(splits))
    used = assign(jobs, cap)
    res = dict(name=name, jobs=jobs, cap=cap, fixed=fixed, modal_used=used,
               n_train=sum(j.kind == "train" for j in jobs), n_rob=sum(j.kind == "rob" for j in jobs),
               total=sum(j.hours for j in jobs), by_bin={}, weeks={}, weeks_nodata={}, sessions={})
    for b in ("Kaggle A", "Kaggle B", "Modal"):
        res["by_bin"][b] = sum(j.hours for j in jobs if j.bin == b)
    for b in ("Kaggle A", "Kaggle B"):
        bundles = []
        for j in jobs:
            if j.bin == b and j.kind in ("train", "tune"):
                bundles.append(j.hours + sum(r.hours for r in jobs if r.after == j.id))
        for key, oh in (("weeks", SESSION_OVERHEAD_DATA), ("weeks_nodata", SESSION_OVERHEAD_NODATA)):
            s = pack(bundles, oh)
            res[key][b] = math.ceil(sum(s) / KAGGLE_WEEKLY_H - 1e-9) if s else 0
            if key == "weeks":
                res["sessions"][b] = len(s)
    res["modal_cost"] = used * rate() + fixed
    return res


# ----------------------------------------------------------------------------- rendering
def launch(j):
    if j.kind == "tune":
        return f"modal run app.py::tune --model {j.model}"
    base = f"--model {j.model} --variant {j.variant} --seed {j.seed} --split {j.split}"
    if j.bin == "Modal":
        return f"modal run app.py::{'train' if j.kind == 'train' else 'robustness'} {base}"
    env = (f"TM_MODEL={ {'cnn_mlp': 'mlp', 'cnn_lstm': 'lstm', 'transformer': 'transformer'}[j.model] } "
           f"TM_VARIANT={j.variant} TM_SEED={j.seed} TM_RUN_TAG={j.model}_{j.variant}_s{j.seed}")
    return f"notebook cell: {env}; split={j.split}; " + ("train+eval" if j.kind == "train" else "pipeline_robustness.py")


def render():
    out = []
    w = out.append
    w("# RUN_MATRIX (generated by `run_matrix.py` - do not hand-edit)\n")
    w("Budget plan for the TokaMark `task_1-3` study: 3 models x 3 variants (plain / mask / mask+PINN) x seeds, "
      "random split main table, temporal replica, robustness passes, one `tune` stage per model.\n")
    w("## 1. Inputs and how trustworthy each is\n")
    w("| Quantity | Value | Source |\n|---|---|---|")
    w("| CNN+MLP plain, 1 run | 2.58 h | **measured** (user brief, 155 min) |")
    w("| CNN+MLP mask+PINN, 1 run | 3.17 h | **measured** (190 min; `train_summary.json` says 194.4 min, 4,200 steps, 53 epochs) |")
    w("| CNN+LSTM mask+PINN, 1 run | 3.17 h | **measured** (user brief, ~190 min) |")
    w("| CNN+LSTM plain | 2.6 h | extrapolated (LSTM x 155/190) |")
    w("| mask-only (MLP, LSTM) | 3.0 h | assumed |")
    w("| Transformer plain / mask / mask+PINN | 3.4 / 3.9 / 4.1 h | **extrapolated 1.3 x CNN+LSTM, no measurement; +/-50 %** |")
    w(f"| post-train eval + plots per run | {POST_H} h | assumed |")
    w(f"| robustness study, 63 passes | {ROB_H} h | user brief (1-2 h), midpoint; an older log in `results/robustness_lstm` shows 8.7 min for a smaller harness, so this is conservative |")
    w(f"| `tune` per model | {TUNE_TRIALS} trials x {TUNE_TRIAL_H} h = {TUNE_TRIALS * TUNE_TRIAL_H} h | derived from the in-progress `tune` defaults (6 trials x 300 steps) at the measured step time; confirm with the first tune run |")
    w("| 1 or 2 GPUs | 1 T4 == 2xT4 | the 190-min run logged `single GPU (DataParallel not clearly faster)`, so durations are 1-T4 durations |")
    w(f"| Kaggle | 2 accounts x {KAGGLE_WEEKLY_H:.0f} h/week, sessions packed to {SESSION_H:.0f} h | user brief; session limit 9-12 h unverified, 9 used |")
    w(f"| Kaggle per-session overhead | {SESSION_OVERHEAD_DATA} h with a saved dataset / {SESSION_OVERHEAD_NODATA} h re-downloading ~40 GB | assumed / notebook default `TM_PREP_DEADLINE_H=2.0` |")
    w("")
    w("### Modal price (list, USD) - https://modal.com/pricing, fetched 2026-11-04\n")
    w("| GPU | $/s | $/h GPU only | $/h all-in (4 cores + 16 GiB) |\n|---|---|---|---|")
    for g, p in PRICE_S.items():
        w(f"| {g} | {p} | {p * 3600:.3f} | {rate(g):.3f} |")
    w(f"\nCPU ${CPU_CORE_S}/physical core/s, memory ${MEM_GIB_S}/GiB/s. CPU-only prep container: ${CPU_ONLY_RATE:.3f}/h. "
      "Starter plan advertises $30/month free credit.\n")
    w("**What does \"20 hrs\" mean?** Modal bills dollars, not hours.\n")
    w(f"- Interpretation A, $20 credit: {MODAL_CREDIT_USD / rate():.1f} T4-hours all-in ({MODAL_CREDIT_USD / (PRICE_S['T4'] * 3600):.1f} h if only the GPU were billed; CPU and memory are billed on top), "
      f"{MODAL_CREDIT_USD / rate('L4'):.1f} L4-h, {MODAL_CREDIT_USD / rate('A10G'):.1f} A10G-h, {MODAL_CREDIT_USD / rate('A100_40'):.1f} A100-40GB-h.")
    w(f"- Interpretation B, 20 literal hours: ~{MODAL_CREDIT_HOURS_LITERAL:.0f} T4-h (about ${MODAL_CREDIT_HOURS_LITERAL * rate():.0f}). Same order of magnitude, so the plan below is insensitive to the reading.")
    w("- Check the real balance in the Modal dashboard (Settings -> Billing) before the first real job; this repo cannot read it.\n")
    w("**Is a faster Modal GPU worth it?** L4 is cheaper per run than T4 only if it is "
      f"> {rate('L4') / rate('T4'):.2f}x faster (all-in), A10G if > {rate('A10G') / rate('T4'):.2f}x. The job may be input-bound "
      "(DataParallel did not help), in which case a bigger GPU buys nothing: measure steps/s in the smoke test on T4 and L4 and pick by that rule. "
      "Plan assumes T4.\n")

    res = {n: summarize(n) for n in ("minimal", "target", "stretch")}
    w("## 2. Scenarios\n")
    w("| Scenario | train runs | rob passes | total GPU-h | +10 % early-stop contingency | Kaggle A h / weeks | Kaggle B h / weeks | Modal h / cost | weeks if data NOT saved as dataset |\n|---|---|---|---|---|---|---|---|---|")
    for n, r in res.items():
        b = r["by_bin"]
        w(f"| **{n}** | {r['n_train']} | {r['n_rob']} | {r['total']:.1f} | {r['total'] * 1.1:.1f} | "
          f"{b['Kaggle A']:.1f} / {r['weeks']['Kaggle A']} | {b['Kaggle B']:.1f} / {r['weeks']['Kaggle B']} | "
          f"{b['Modal']:.1f} / ${r['modal_cost']:.1f} | A {r['weeks_nodata']['Kaggle A']}, B {r['weeks_nodata']['Kaggle B']} |")
    w("")
    w("Weeks = ceil(session wall-hours / 30) per account, sessions first-fit packed to 9 h. If 2 independent processes (one per T4) "
      "fit in one 2xT4 session without slowing down (untested; DataParallel not helping hints at an input bottleneck, so test it "
      "with the first real session), Kaggle weeks roughly halve. Modal capacity above is "
      f"{res['target']['cap']:.1f} T4-h after CPU-only prep, one smoke test and a {MODAL_RESERVE:.0%} credit reserve (interpretation A).\n")
    w("Scenario content:\n")
    w("- **minimal**: half-size tune; the 9 ablation cells x seed 1 (random); seeds 2-3 for CNN+MLP plain and mask+PINN only (a mean +/- std on the headline pair); robustness for the three mask+PINN models.")
    w("- **target**: full tune; all 9 cells x seeds 1-3 (27 runs); robustness for all 9 cells (seed 1); temporal replica of plain and mask+PINN for each model (6 runs).")
    w("- **stretch**: target + seeds 4-5 for every cell (18 runs), temporal mask rows (3), temporal robustness (6).")
    w("")
    w("## 3. Priority order and cut order\n")
    w("Run in tier order; the ablation table is fillable (one seed, every cell) as soon as tier 2 finishes. Cut from the bottom when hours run out.\n")
    w("| Tier | Content | If hours run out |\n|---|---|---|")
    w("| 0 | `prep` random (once), smoke test | never cut |")
    w("| 1 | `tune` per model | shrink trials, never skip: equal tuning is a stated requirement |")
    w("| 2 | 9 cells x seed 1 (random) | last to cut; order cnn_mlp, cnn_lstm, transformer |")
    w("| 3 | robustness: mask+PINN and plain, seed 1 | cut plain before mask+PINN |")
    w("| 4 | seeds 2-3, all cells | keep headline pair (cnn_mlp plain / mask+PINN) longest; drop transformer seeds first (most expensive, least certain) |")
    w("| 5 | temporal replica + robustness of mask rows | drop mask robustness first, then temporal transformer |")
    w("| 6 | seeds 4-5, temporal mask, temporal robustness | cut first |")
    w("")
    w("## 4. Data: keeping every seed and variant on the SAME shots\n")
    w("`out/<split>/selection.json` (plus `out/tar_index.json`, which makes `prep` skip re-indexing) is the contract. "
      "Existing local copy: `cnn_mlp/results/run_40gb_CNN_MLP_failure_aware/out/random/selection.json` (606/74/60 shots, 40 GB). "
      "**Reuse it instead of re-selecting**: upload it to the Modal data volume (`RUNBOOK_MODAL.md` step 3), so Modal downloads exactly those shots, "
      "and copy the same file into every Kaggle notebook (it is 0.3 MB; `prep` reuses it when it has >=5 shots per subset and `TM_FORCE_RESELECT` is unset).\n")
    w("| Option | Per-session cost | Notes |\n|---|---|---|")
    ns = res['target']['sessions']['Kaggle A'] + res['target']['sessions']['Kaggle B']
    w(f"| Kaggle re-downloads from HF every session | ~{SESSION_OVERHEAD_NODATA} h GPU quota each (notebook default prep deadline 2 h) | target plan = {ns} sessions -> ~{ns * SESSION_OVERHEAD_NODATA:.0f} quota-hours burned on downloading: avoid |")
    w(f"| **Tar each split once on Modal, download, upload as a Kaggle dataset (recommended for Kaggle)** | ~{SESSION_OVERHEAD_DATA} h (untar to local disk) | one-time: `modal volume get tokamark-data tars/random.tar .` then upload (40 GB; Kaggle private dataset limits not verified - check before relying; a tar is 1 file, so file-count limits are not an issue). Put the same selection.json inside. |")
    w("| Run everything on Modal | 0 h (volume -> local disk untar ~minutes) | limited by ~20 h credit, so Modal alone covers only ~6 full runs |")
    w("Recommendation: prep once on Modal (CPU-only, ~$1), keep Modal as the source of truth, and ship the tar + selection.json to a Kaggle dataset "
      "if the upload bandwidth allows; otherwise accept one HF re-download per Kaggle session and use long (11-12 h) sessions to amortise it. "
      "Temporal split is a separate 40 GB (different official shot lists) - only needed in tier 5+; keep it on Modal.\n")

    for n, r in res.items():
        w(f"## 5{'abc'[list(res).index(n)]}. Assignment - {n}\n")
        w(f"Modal: {r['modal_used']:.1f} h GPU of {r['cap']:.1f} h usable, estimated cost ${r['modal_cost']:.2f} "
          f"(includes CPU-only prep ${len({j.split for j in r['jobs'] if j.kind != 'tune'}) * PREP_H_PER_SPLIT * CPU_ONLY_RATE:.2f} + smoke).\n")
        w("| # | Tier | Job | Hours | Bin | Launch |\n|---|---|---|---|---|---|")
        for i, j in enumerate(sorted(r["jobs"], key=lambda x: (x.tier, jobs_idx(r["jobs"], x))), 1):
            w(f"| {i} | {j.tier} | {j.id} | {j.hours:.2f} | {j.bin} | `{launch(j)}` |")
        w("")
    w("## 6. Notes\n")
    w("- Robustness jobs run in the same bin as their training job because they need that run's `best_model.pt` (Kaggle: same session or saved output; Modal: same results volume).")
    w("- Transformer figures are the weakest numbers here. If the first transformer smoke test shows step time k x the LSTM's, scale `TRAIN_H` in `run_matrix.py` and re-run.")
    w("- Kaggle bins are balanced by hours, not by model, so any model can land on either account; `selection.json` identity matters, not the account.")
    return "\n".join(out) + "\n"


def jobs_idx(jobs, j):
    return jobs.index(j)


def self_check():
    for n in ("minimal", "target", "stretch"):
        r = summarize(n)
        assert all(j.bin for j in r["jobs"]), f"{n}: unassigned job"
        assert all(next(t for t in r["jobs"] if t.id == j.after).bin == j.bin for j in r["jobs"] if j.kind == "rob")
        assert r["modal_used"] <= r["cap"] + 1e-9
        assert abs(sum(r["by_bin"].values()) - r["total"]) < 1e-6
    assert summarize("target")["n_train"] == 27 + 6
    assert summarize("stretch")["n_train"] == 27 + 6 + 18 + 3
    assert abs(rate() - 0.907) < 0.01


if __name__ == "__main__":
    self_check()
    if "--check" not in sys.argv:
        with open(os.path.join(HERE, "RUN_MATRIX.md"), "w", encoding="utf-8") as f:
            f.write(render())
        print("wrote", os.path.join(HERE, "RUN_MATRIX.md"))
