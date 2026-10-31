import json, csv, zipfile

def load_metrics(path):
    with open(path) as f:
        r = list(csv.DictReader(f))
        return r[0]

def load_metrics_zip(zpath, entry):
    with zipfile.ZipFile(zpath) as z:
        lines = z.read(entry).decode('utf-8').splitlines()
        r = list(csv.DictReader(lines))
        return r[0]

m1 = load_metrics('results/run_40gb_CNN_MLP/extracted/out/random/results/task_1-3/task_metrics.csv')
m2 = load_metrics('results/run_40gb_CNN_MLP/extracted_run2/out/random/results/task_1-3/task_metrics.csv')
mL = load_metrics_zip('results/run_40gb_CNN_LSTM/results.zip', 'results/out/task_metrics.csv')

print('=' * 85)
print(f"{'Metric':<18} | {'Run 1 (wd=1e-4)':<20} | {'Run 2 (wd=1e-3)':<20} | {'CNN+LSTM':<16}")
print('=' * 85)
for k in ['NRMSE_mean', 'NMAE_mean', 'RMSE_mean', 'MAE_mean']:
    v1 = float(m1[k])
    v2 = float(m2[k])
    vL = float(mL[k])
    b1 = " (BEST)" if v1 < v2 else ""
    b2 = " (BEST)" if v2 < v1 else ""
    print(f"{k:<18} | {v1:.5f}{b1:<13} | {v2:.5f}{b2:<13} | {vL:.5f}")
print('=' * 85)

t1 = json.load(open('results/run_40gb_CNN_MLP/extracted/out/random/train_summary.json'))
t2 = json.load(open('results/run_40gb_CNN_MLP/extracted_run2/out/random/train_summary.json'))

print(f"{'Total Steps':<18} | {t1['steps']:<20} | {t2['steps']:<20} | {'~1500':<16}")
print(f"{'Epochs':<18} | {t1['epochs']:<20} | {t2['epochs']:<20} | {'~20':<16}")
print(f"{'Duration':<18} | {t1['duration_min']:.1f} min{'':<13} | {t2['duration_min']:.1f} min{'':<13} | {'~120 min':<16}")
print(f"{'Best Val Loss':<18} | {t1['best_val']:.5f}{'':<13} | {t2['best_val']:.5f}{'':<13} | {'1.12000':<16}")
print(f"{'Weight Decay':<18} | {t1['weight_decay']:<20} | {t2['weight_decay']:<20} | {'0.0001':<16}")
print('=' * 85)
