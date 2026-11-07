"""Build the Kaggle kernel that runs the full robustness study on the trained failure-aware CNN+LSTM.

Run from anywhere:  python scripts/build_lstm_robust_kernel.py   (re-run after editing src/pipeline.py or src/study/*)

The kernel takes the training kernel's output (kernel_sources) as the frozen model, then runs the 63-pass sensor-failure
sweep on both T4s, the Module 1 physics metrics per pass, and the Module 3 / Module 4 studies (src/study/physics_study.py).
Output: kernels/tokamark-robust-lstm-fa/{tokamark_robustness_kaggle.ipynb, kernel-metadata.json}
"""
import io, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "src", "study", "tokamark_robustness_study_template.ipynb")
OUT = os.path.join(ROOT, "kernels", "tokamark-robust-lstm-fa")
NB_NAME = "tokamark_robustness_kaggle.ipynb"
TRAIN_KERNEL = "nishanthdevabathini/tokamark-train-lstm-fa"
FILES = {  # name inside /kaggle/working -> repo path
    "pipeline.py": "src/pipeline.py",
    "robustness.py": "src/study/pipeline_robustness.py",
    "physics_evaluation.py": "src/study/physics_evaluation.py",
    "efit_truth.py": "src/study/efit_truth.py",
    "aggregate_seeds.py": "src/study/aggregate_seeds.py",
    "adaptive_mitigation.py": "src/study/adaptive_mitigation.py",
    "sensor_criticality.py": "src/study/sensor_criticality.py",
    "physics_study.py": "src/study/physics_study.py",
}

nb = json.load(io.open(TEMPLATE, encoding="utf-8"))
cells = nb["cells"]
by_marker = {"".join(c["source"]).split("\n", 1)[0]: c for c in cells if c["cell_type"] == "code"}
for name, rel in FILES.items():
    marker = f"%%writefile /kaggle/working/{name}"
    body = io.open(os.path.join(ROOT, rel), encoding="utf-8").read().replace("\r\n", "\n")
    lines = (marker + "\n" + body).splitlines(keepends=True)
    if not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    by_marker[marker]["source"] = lines

cfg_cell = next(c for c in cells if c["cell_type"] == "code" and "# ================= CONFIG" in "".join(c["source"]))
cfg = "".join(cfg_cell["source"])
old = 'MODEL_TYPE    = "mlp"     # "mlp" (CNN+MLP champion) or "lstm" (CNN+LSTM benchmark)'
assert cfg.count(old) == 1
cfg = cfg.replace(old, 'MODEL_TYPE    = "lstm"    # trained failure-aware CNN+LSTM')
cfg_cell["source"] = cfg.splitlines(keepends=True)
# dual-GPU inference is on by default (DataParallel when 2 GPUs are visible); TM_USE_DP=0 would turn it off.

os.makedirs(OUT, exist_ok=True)
json.dump(nb, io.open(os.path.join(OUT, NB_NAME), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
meta = {"id": "nishanthdevabathini/tokamark-robust-lstm-fa", "title": "tokamark-robust-lstm-fa", "code_file": NB_NAME,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [TRAIN_KERNEL], "model_sources": []}
json.dump(meta, io.open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
print("built", OUT, "| cells:", len(cells), "| kernel_sources:", TRAIN_KERNEL)
