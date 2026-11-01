import zipfile

z = zipfile.ZipFile('results/run_40gb_CNN_MLP/results-2.zip')
code = z.read('tokamark_baseline/src/trainer.py').decode('utf-8', errors='replace')
lines = code.splitlines()

start = False
for i, l in enumerate(lines):
    if "class BatchStepTrainer" in l:
        start = True
    if start:
        print(f"{i+1:3d}: {l}")
        if "def train(" in l:
            # print next 80 lines
            for j in range(i+1, min(i+120, len(lines))):
                print(f"{j+1:3d}: {lines[j]}")
            break
