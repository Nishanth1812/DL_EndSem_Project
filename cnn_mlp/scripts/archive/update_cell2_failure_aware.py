import json

nb_path = "src/tokamark_baseline_kaggle.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell2_src = nb["cells"][1]["source"]

new_src = []
for line in cell2_src:
    if "TRAIN_FROM_SCRATCH = True" in line:
        new_src.append(line)
        new_src.append("FAILURE_AWARE  = 1        # 0 = baseline CNN; 1 = Module 2 Failure-Aware + PINN model\n")
        new_src.append("FAIL_AUG_P     = 0.5      # failure augmentation probability during training\n")
        new_src.append("PINN_LAMBDA    = 0.1      # flux-loop consistency loss weight\n")
    elif "TM_TRAIN_FROM_SCRATCH" in line:
        line_mod = line.replace(
            'TM_TRAIN_FROM_SCRATCH=("1" if TRAIN_FROM_SCRATCH else "0"))',
            'TM_TRAIN_FROM_SCRATCH=("1" if TRAIN_FROM_SCRATCH else "0"), TM_FAILURE_AWARE=("1" if FAILURE_AWARE else "0"), TM_FAIL_AUG_P=FAIL_AUG_P, TM_PINN_LAMBDA=PINN_LAMBDA)\n'
        )
        new_src.append(line_mod)
    else:
        new_src.append(line)

nb["cells"][1]["source"] = new_src
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated Cell 2 successfully")
