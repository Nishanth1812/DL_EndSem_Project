"""Build the figures and LaTeX tables of the CNN+MLP vs CNN+LSTM report from the downloaded Kaggle runs.

Run from the repo root:  python cnn_mlp/report/build_report_assets.py
Reads  runs/{mlp_fa,lstm_fa,mlp_fa_robust_v3,lstm_fa_robust}  (gitignored Kaggle outputs) and
       cnn_mlp/results/run_40gb_CNN_MLP_failure_aware  (committed old failure-aware MLP run).
Writes cnn_mlp/report/figures/*.png and cnn_mlp/report/generated_tables.tex. Every number in the report tables comes from here.
"""
import csv, json, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(ROOT, "runs")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

P = {
    "MLP": dict(train=f"{RUNS}/mlp_fa/out/random", rob=f"{RUNS}/mlp_fa_robust_v3/out/random/robustness"),
    "LSTM": dict(train=f"{RUNS}/lstm_fa/out/random", rob=f"{RUNS}/lstm_fa_robust/out/random/robustness"),
}
OLD = f"{ROOT}/cnn_mlp/results/run_40gb_CNN_MLP_failure_aware/out/random"
COL = {"MLP": "#1F4E79", "LSTM": "#C55A11"}
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight"})


def jl(path):
    return json.load(open(path, encoding="utf-8"))


def tex(s):
    return str(s).replace("_", r"\_")


ROB = {k: jl(v["rob"] + "/robust_metrics.json") for k, v in P.items()}
CRIT = {k: jl(v["rob"] + "/physics/criticality.json") for k, v in P.items()}
MIT = {k: jl(v["rob"] + "/physics/mitigation.json") for k, v in P.items()}
TS = {k: jl(v["train"] + "/train_summary.json") for k, v in P.items()}
HIST = {k: jl(v["train"] + "/model/training_history.json") for k, v in P.items()}
HIST["MLP (old)"] = jl(OLD + "/model/training_history.json")


def task_metrics(base):
    for r in csv.DictReader(open(base + "/results/task_1-3/task_metrics.csv")):
        if r["feature_name"] == "equilibrium-psi":
            return {k: float(r[k]) for k in ("NRMSE_mean", "NMAE_mean", "RMSE_mean", "MAE_mean", "n_shots")}


TM = {"MLP": task_metrics(P["MLP"]["train"]), "LSTM": task_metrics(P["LSTM"]["train"]), "MLP (old)": task_metrics(OLD)}

# --- copy the pipeline's own figures ------------------------------------------------------------------------
for m, base in (("lstm", P["LSTM"]["train"]), ("mlp", P["MLP"]["train"])):
    for f in os.listdir(base + "/figures"):
        shutil.copy(f"{base}/figures/{f}", f"{FIG}/{m}_{f}")
for m in ("MLP", "LSTM"):
    for sub, names in (("figures", ("signal_importance.png", "degradation_curves.png")),
                       ("physics", ("minimal_sensor_set_curve.png", "spatial_error_propagation_heatmaps.png"))):
        for f in names:
            shutil.copy(f"{P[m]['rob']}/{sub}/{f}", f"{FIG}/{m.lower()}_{f}")

# --- helpers ------------------------------------------------------------------------------------------------
SCEN = ["random_dropout", "temporal_gap", "disruption_proximate", "channel_ablation", "group_failure", "category_abl"]
SCEN_LABEL = {"random_dropout": "Random dropout", "temporal_gap": "Temporal gap", "disruption_proximate": "Disruption-proximate",
              "channel_ablation": "Channel ablation", "group_failure": "Group failure", "category_abl": "Category ablation"}


def zero_passes(m, scen):
    out = []
    for name, v in ROB[m]["corrupted"].items():
        s, sev, imp = name.split("|")
        if s == scen and imp == "zero":
            out.append((sev, v))
    return out


def clean_nrmse(m):
    return ROB[m]["clean"]["NRMSE"]


def worst(m, scen):
    return max(v["NRMSE"] for _, v in zero_passes(m, scen))


# --- Fig: training curves of the three failure-aware runs ----------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
sty = {"MLP": (COL["MLP"], "-"), "MLP (old)": ("#7FA7CF", "--"), "LSTM": (COL["LSTM"], "-")}
lab = {"MLP": "CNN+MLP (epoch loop)", "MLP (old)": "CNN+MLP (step loop)", "LSTM": "CNN+LSTM (step loop)"}
for m, h in HIST.items():
    c, ls = sty[m]
    ax[0].plot(h["step"], h["val_loss"], color=c, ls=ls, lw=1.8, label=lab[m])
    ax[1].plot(h["step"], h["train_loss"], color=c, ls=ls, lw=1.8, label=lab[m])
ax[0].set_title("Validation loss"); ax[1].set_title("Training loss (running mean)")
for a in ax:
    a.set_xlabel("optimiser step"); a.set_ylabel("MultiOutputMSELoss (standardised)")
ax[1].set_yscale("log"); ax[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/cmp_training.png"); plt.close(fig)

# --- Fig: robustness scores ----------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 3.6))
x = np.arange(len(SCEN)); w = 0.38
for i, m in enumerate(("MLP", "LSTM")):
    rs = [clean_nrmse(m) / worst(m, s) for s in SCEN]
    b = ax.bar(x + (i - 0.5) * w, rs, w, color=COL[m], label=f"CNN+{m}")
    ax.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
ax.set_xticks(x); ax.set_xticklabels([SCEN_LABEL[s].replace(" ", "\n", 1) for s in SCEN], fontsize=9)
ax.set_ylabel("Robustness score (clean / worst NRMSE)"); ax.set_ylim(0, 1.1); ax.legend(frameon=False, ncol=2)
ax.set_title("Worst-case robustness score per failure scenario (zero-fill, higher is better)")
fig.tight_layout(); fig.savefig(f"{FIG}/cmp_robustness_score.png"); plt.close(fig)

# --- Fig: degradation curves ---------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(12.5, 3.3), sharey=False)
for a, s in zip(ax, SCEN[:4]):
    for m in ("MLP", "LSTM"):
        pts = sorted((float(sev), v["NRMSE"]) for sev, v in zero_passes(m, s))
        a.plot(*zip(*pts), "o-", color=COL[m], lw=1.8, label=f"CNN+{m}")
        a.axhline(clean_nrmse(m), color=COL[m], ls=":", lw=1)
    a.set_title(SCEN_LABEL[s], fontsize=10)
    a.set_xlabel("channels removed" if s == "channel_ablation" else "severity (fraction)")
ax[0].set_ylabel("NRMSE (test)"); ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/cmp_degradation.png"); plt.close(fig)

# --- Fig: Module 1 boundary error ----------------------------------------------------------------------------
PASSES = [("clean", "Clean"), ("channel_ablation|6|zero", "6 channels\nlost"), ("group_failure|magnetics_all|zero", "Magnetics\ngroup lost"),
          ("temporal_gap|0.4|zero", "Temporal\ngap 40%"), ("disruption_proximate|0.25|zero", "Pre-disruption\n25%"),
          ("random_dropout|0.1|zero", "Random\ndropout 10%")]


def phys(m, name):
    d = ROB[m]
    return d["clean"]["PHYS"] if name == "clean" else d["corrupted"][name]["PHYS"]


fig, ax = plt.subplots(figsize=(9, 3.7))
x = np.arange(len(PASSES))
for i, m in enumerate(("MLP", "LSTM")):
    v = [phys(m, n)["boundary_mean_cm"] for n, _ in PASSES]
    b = ax.bar(x + (i - 0.5) * 0.38, v, 0.38, color=COL[m], label=f"CNN+{m}")
    ax.bar_label(b, fmt="%.1f", fontsize=8, padding=2)
ax.axhline(2.0, color="k", ls="--", lw=1, label="2 cm tolerance (Module 3)")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in PASSES], fontsize=9)
ax.set_ylabel("Mean LCFS boundary error (cm)"); ax.legend(frameon=False, ncol=3, fontsize=8)
ax.set_title("Module 1: plasma boundary error in centimetres (35 test windows per pass)")
fig.tight_layout(); fig.savefig(f"{FIG}/cmp_boundary_cm.png"); plt.close(fig)

# --- Fig: Module 3 criticality ---------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 3.8))
order = [s["short"] for s in CRIT["MLP"]["signals"]]
yy = np.arange(len(order))
for i, m in enumerate(("MLP", "LSTM")):
    sh = {s["short"]: s["shift_cm"] for s in CRIT[m]["signals"]}
    ax.barh(yy + (i - 0.5) * 0.38, [sh[o] for o in order], 0.38, color=COL[m], label=f"CNN+{m}")
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(yy); ax.set_yticklabels([o.replace("_", r"\_") if False else o for o in order]); ax.invert_yaxis()
ax.set_xlabel("Boundary-error change when the branch is lost (cm)")
ax.set_title("Module 3: sensor-branch criticality"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/cmp_criticality.png"); plt.close(fig)

# --- Fig: Module 4 mitigation ----------------------------------------------------------------------------------
SC4 = ["clean", "transient_spike", "redundant_probe_dead", "critical_branch_dead", "group_loss", "channel_loss_6",
       "temporal_gap_40", "pre_disruption_25"]
SC4_LABEL = ["clean", "spike", "redundant\nprobe dead", "critical\nbranch dead", "group\nloss", "6 channels\nlost", "gap 40%", "pre-disr.\n25%"]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for a, m in zip(ax, ("MLP", "LSTM")):
    x = np.arange(len(SC4))
    a.bar(x - 0.2, [MIT[m]["scenarios"][s]["naive"]["boundary_mean_cm"] for s in SC4], 0.4, color="#9E9E9E", label="bare model (zero-fill)")
    a.bar(x + 0.2, [MIT[m]["scenarios"][s]["brain"]["boundary_mean_cm"] for s in SC4], 0.4, color=COL[m], label="with Module 4 brain")
    a.set_xticks(x); a.set_xticklabels([l.replace("\n", " ") for l in SC4_LABEL], fontsize=8, rotation=30, ha="right")
    a.set_title(f"CNN+{m}")
ax[0].set_ylabel("Mean LCFS boundary error (cm)"); ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/cmp_mitigation.png"); plt.close(fig)

# --- LaTeX tables ----------------------------------------------------------------------------------------------
T = []


def table(name, body):
    T.append(f"\\newcommand{{\\{name}}}{{%\n{body}}}\n")


# training summary
rows = []
plain = dict(nrmse=0.14186, mae=0.00389, val=0.8723, ep=44, steps=3500, mins=154.9)
rows.append(f"CNN+MLP, plain baseline$^{{a}}$ & 3\\,500 / 44 & 154.9 & 0.8723 & 0.1419 & 0.00389 \\\\")
for key, label, fn in (("MLP (old)", "CNN+MLP, failure-aware (step loop)", OLD), ("MLP", "CNN+MLP, failure-aware (epoch loop)$^{b}$", None),
                       ("LSTM", "CNN+LSTM, failure-aware (step loop)", None)):
    ts = jl(fn + "/train_summary.json") if fn else TS[key]
    tm = TM[key]
    st = f"{ts['steps']:,}".replace(",", "\\,")
    rows.append(f"{label} & {st} / {ts['epochs']} & {ts['duration_min']:.1f} & {ts['best_val']:.4f} & "
                f"{tm['NRMSE_mean']:.4f} & {tm['MAE_mean']:.5f} \\\\")
table("TrainTable", "\n".join(rows) + "\n")

# clean metrics (from robustness engine, identical protocol)
rows = []
for m in ("MLP", "LSTM"):
    c = ROB[m]["clean"]
    rows.append(f"CNN+{m} & {c['NRMSE']:.4f} $\\pm$ {c['NRMSE_std']:.4f} & {c['NMAE']:.4f} & {c['RMSE']:.5f} & {c['MAE']:.5f} & {c['n_windows']:,} \\\\".replace("3,450", "3\\,450"))
table("CleanTable", "\n".join(rows) + "\n")

# robustness scores
rows = []
for s in SCEN:
    cells = []
    for m in ("MLP", "LSTM"):
        wst = worst(m, s)
        cells += [f"{wst:.3f}", f"{clean_nrmse(m) / wst:.2f}"]
    rows.append(f"{SCEN_LABEL[s]} & " + " & ".join(cells) + " \\\\")
table("RSTable", "\n".join(rows) + "\n")

# full zero-fill sweep (appendix)
names = sorted({n.rsplit("|", 1)[0] for n in ROB["MLP"]["corrupted"]})
rows = []
for n in names:
    cells = [f"{ROB[m]['corrupted'][n + '|zero']['NRMSE']:.4f}" for m in ("MLP", "LSTM")]
    cells += [f"{ROB[m]['corrupted'][n + '|zero']['RS_vs_clean']:.3f}" for m in ("MLP", "LSTM")]
    rows.append(f"\\texttt{{{tex(n)}}} & " + " & ".join(cells) + " \\\\")
table("SweepTable", "\n".join(rows) + "\n")

# worst-case NRMSE per scenario and imputation (whole-sensor scenarios are mask-handled, so imputation cannot matter there)
rows, inv = [], {}
for s in SCEN:
    for m in ("MLP", "LSTM"):
        w3 = [max(v["NRMSE"] for n, v in ROB[m]["corrupted"].items() if n.split("|")[0] == s and n.endswith("|" + i))
              for i in ("zero", "mean", "ffill")]
        inv[(s, m)] = w3
        rows.append(f"{SCEN_LABEL[s] if m == 'MLP' else ''} & CNN+{m} & " + " & ".join(f"{x:.3f}" for x in w3) + " \\\\")
    rows.append("\\midrule")
table("ImputeTable", "\n".join(rows[:-1]) + "\n")
inv = {f"{k[0]}/{k[1]}": v for k, v in inv.items()}

# Module 1
rows = []
for n, lbl in PASSES:
    for m in ("MLP", "LSTM"):
        p = phys(m, n)
        rows.append(f"{lbl.replace(chr(10), ' ').replace('%', chr(92) + '%') if m == 'MLP' else ''} & CNN+{m} & {p['boundary_mean_cm']:.2f} & {p['boundary_hausdorff_cm']:.1f} & "
                    f"{p['axis_shift_cm']:.1f} & {p['xpoint_err_cm']:.1f} & {p['false_alarm_rate_pct']:.1f} & {p['missed_alarm_rate_pct']:.1f} \\\\")
    rows.append("\\midrule")
table("PhysTable", "\n".join(rows[:-1]) + "\n")

# Module 3
rows = []
for s_m, s_l in zip(CRIT["MLP"]["signals"], sorted(CRIT["LSTM"]["signals"], key=lambda s: [x["short"] for x in CRIT["MLP"]["signals"]].index(s["short"]))):
    rows.append(f"\\texttt{{{tex(s_m['short'])}}} & {s_m['n_probes']} & {s_m['boundary_cm']:.2f} & {s_m['shift_cm']:+.2f} & "
                f"{s_l['boundary_cm']:.2f} & {s_l['shift_cm']:+.2f} \\\\")
table("CritTable", "\n".join(rows) + "\n")
T.append("\\newcommand{\\CleanBndMLP}{%.2f}\n\\newcommand{\\CleanBndLSTM}{%.2f}\n" % (CRIT["MLP"]["clean"]["boundary_cm"], CRIT["LSTM"]["clean"]["boundary_cm"]))

# Module 4
rows = []
for m in ("MLP", "LSTM"):
    for s, l in zip(SC4, SC4_LABEL):
        d = MIT[m]["scenarios"][s]
        rows.append(f"{('CNN+' + m) if s == 'clean' else ''} & {l.replace(chr(10), ' ').replace('%', chr(92) + '%')} & {d['naive']['boundary_mean_cm']:.2f} & {d['brain']['boundary_mean_cm']:.2f} & "
                    f"{d['alarms']['naive']['false_alarm_rate_pct']:.0f} / {d['alarms']['brain']['false_alarm_rate_pct']:.0f} / {d['alarms']['conservative']['false_alarm_rate_pct']:.0f} & "
                    f"{d['alarms']['naive']['missed_alarm_rate_pct']:.0f} / {d['alarms']['brain']['missed_alarm_rate_pct']:.0f} / "
                    f"{d['alarms']['conservative']['missed_alarm_rate_pct']:.0f} & "
                    f"{d['latency_ms']['total']['mean']:.1f} \\\\")
    rows.append("\\midrule")
table("MitTable", "\n".join(rows[:-1]) + "\n")
for m in ("MLP", "LSTM"):
    d = MIT[m]
    T.append("\\newcommand{\\Delta%s}{%.1f}\n" % (m, d["delta_cm"]["module2"]))
    T.append("\\newcommand{\\Lat%s}{%.1f}\n" % (m, np.mean([d["scenarios"][s]["latency_ms"]["total"]["mean"] for s in SC4])))
    T.append("\\newcommand{\\Ncal%s}{%d}\n" % (m, d["n_calibration"]))
# hazard prevalence (clean): shows the wall-clearance proxy flags most windows
d = MIT["MLP"]["scenarios"]["clean"]["alarms"]["naive"]
T.append("\\newcommand{\\NHazard}{%d}\n\\newcommand{\\NSafe}{%d}\n" % (d["n_hazard"], d["n_safe"]))

# LSTM vs MLP deltas quoted in the text
for k, v in (("NRMSEmlp", TM["MLP"]["NRMSE_mean"]), ("NRMSElstm", TM["LSTM"]["NRMSE_mean"]), ("NRMSEmlpold", TM["MLP (old)"]["NRMSE_mean"])):
    T.append("\\newcommand{\\%s}{%.4f}\n" % (k, v))
open(HERE + "/generated_tables.tex", "w", encoding="utf-8").write("".join(T))
print("figures:", sorted(os.listdir(FIG)))
print("tables written; imputation spread", inv)
