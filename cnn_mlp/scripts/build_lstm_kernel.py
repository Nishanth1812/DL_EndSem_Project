"""Build the Kaggle kernel for the failure-aware CNN+LSTM run from the shared notebook.

Run from anywhere:  python scripts/build_lstm_kernel.py      (re-run after every edit of src/pipeline.py)

The LSTM run is the CNN+MLP run with ONE change, MODEL_TYPE = "lstm". Everything else (data subset, batch, lr, weight
decay, patience, validation interval, failure augmentation, PINN weight, seed) comes from the same notebook and the same
pipeline.py. Two things are added for the LSTM kernel only:
  * the CNN+MLP run's exact shot selection (selection.json + tar_index.json) is embedded, so both models train, validate
    and test on the same shots;
  * USE_DP = "1": DataParallel over both T4s is forced instead of the 1-vs-2 GPU auto-benchmark (which picked one GPU for
    the CNN+MLP run).
Outputs: kernels/tokamark-train-lstm-fa/{tokamark_baseline_kaggle.ipynb, kernel-metadata.json}
"""
import base64, gzip, io, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_NB = os.path.join(ROOT, "src", "tokamark_baseline_kaggle.ipynb")
RUN = os.path.join(ROOT, "results", "run_40gb_CNN_MLP_failure_aware", "out")
OUT = os.path.join(ROOT, "kernels", "tokamark-train-lstm-fa")
NB_NAME = "tokamark_baseline_kaggle.ipynb"


def sub(src, old, new):
    assert src.count(old) == 1, old
    return src.replace(old, new)


nb = json.load(io.open(SRC_NB, encoding="utf-8"))
cfg = "".join(nb["cells"][1]["source"])
cfg = sub(cfg, 'MODEL_TYPE     = "mlp"', 'MODEL_TYPE     = "lstm"')
cfg = sub(cfg, 'USE_DP         = "auto"', 'USE_DP         = "1"   ')
nb["cells"][1]["source"] = cfg.splitlines(keepends=True)

emb = {}
for rel, path in (("out/tar_index.json", os.path.join(RUN, "tar_index.json")),
                  ("out/random/selection.json", os.path.join(RUN, "random", "selection.json"))):
    emb[rel] = base64.b64encode(gzip.compress(io.open(path, "rb").read(), 9)).decode()
code = (
    "# Same data as the CNN+MLP run: reuse its exact shot selection (the pipeline reuses an existing selection.json).\n"
    "import base64, gzip\n"
    f"_EMB = {json.dumps(emb)}\n"
    "for rel, b64 in _EMB.items():\n"
    "    p = os.path.join(W, rel); os.makedirs(os.path.dirname(p), exist_ok=True)\n"
    "    open(p, 'wb').write(gzip.decompress(base64.b64decode(b64)))\n"
    "sel = json.load(open(f'{W}/out/random/selection.json'))\n"
    "print('CNN+MLP shot selection restored:', {k: len(v) for k, v in sel['shots'].items()})\n"
)
nb["cells"].insert(2, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                       "source": code.splitlines(keepends=True)})
nb["cells"][0]["source"] = (
    "# TokaMark `task_1-3` failure-aware CNN+LSTM on Kaggle (dual T4)\n"
    "Identical setup to the CNN+MLP run: same shots, batch 512, Adam lr 1e-4 / wd 1e-4, patience 10, validate every 100 "
    "steps, failure augmentation p=0.5, flux-loop PINN weight 0.1, seed 23. Only the backbone differs (LSTM instead of MLP) "
    "and `DataParallel` is forced over both T4s.\n"
    "Settings: **Internet ON**, **GPU T4 x2**, *Save & Run All*.\n").splitlines(keepends=True)

os.makedirs(OUT, exist_ok=True)
json.dump(nb, io.open(os.path.join(OUT, NB_NAME), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
meta = {"id": "nishanthdevabathini/tokamark-train-lstm-fa", "title": "tokamark-train-lstm-fa", "code_file": NB_NAME,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        "model_sources": []}
json.dump(meta, io.open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)

# the pipeline cell must still be the shared pipeline.py
chk = json.load(io.open(os.path.join(OUT, NB_NAME), encoding="utf-8"))
body = "".join(chk["cells"][3]["source"])
assert body.startswith("%%writefile /kaggle/working/pipeline.py\n")
disk = io.open(os.path.join(ROOT, "src", "pipeline.py"), encoding="utf-8").read().replace("\r\n", "\n")
assert body[len("%%writefile /kaggle/working/pipeline.py\n"):] == disk, "run scripts/sync_nb.py first"
print("built", OUT, "| pipeline cell == src/pipeline.py | cells:", len(chk["cells"]))
