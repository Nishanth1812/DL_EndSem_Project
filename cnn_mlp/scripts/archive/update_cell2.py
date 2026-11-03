import json

NB_PATH = "src/tokamark_baseline_kaggle.ipynb"
nb = json.load(open(NB_PATH, encoding="utf-8"))

src = nb["cells"][1]["source"]
full_src = "".join(src)

# Add MODEL_TYPE to config
old_cfg = 'WEIGHT_DECAY   = 1e-3     # L2 regularization: 1e-3 reduces overfitting on the 606 shots (official default: 1e-4)\nTRAIN_FROM_SCRATCH = True # True = wipe old checkpoints and train fresh from step 0; False = resume\n'
new_cfg = ('MODEL_TYPE     = "mlp"    # "mlp" (spatial feedforward, best 0.1419 NRMSE) or "lstm" (friend\'s recurrent model)\n'
           'WEIGHT_DECAY   = 1e-4     # L2 regularization: 1e-4 is optimal balance (1e-3 slightly over-penalizes)\n'
           'TRAIN_FROM_SCRATCH = True # True = wipe old checkpoints and train fresh from step 0; False = resume\n')

assert old_cfg in full_src, "old_cfg not found in Cell 2!"
full_src = full_src.replace(old_cfg, new_cfg)

# Add TM_MODEL to env dict
old_env = 'TM_WEIGHT_DECAY=WEIGHT_DECAY, TM_TRAIN_FROM_SCRATCH=("1" if TRAIN_FROM_SCRATCH else "0"))'
new_env = 'TM_MODEL=MODEL_TYPE, TM_WEIGHT_DECAY=WEIGHT_DECAY, TM_TRAIN_FROM_SCRATCH=("1" if TRAIN_FROM_SCRATCH else "0"))'

assert old_env in full_src, "old_env not found in Cell 2!"
full_src = full_src.replace(old_env, new_env)

nb["cells"][1]["source"] = [l + "\n" for l in full_src.splitlines()]

json.dump(nb, open(NB_PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("Updated Cell 2 in tokamark_baseline_kaggle.ipynb successfully.")
