import zipfile

z = zipfile.ZipFile('results/run_40gb_CNN_MLP/results-2.zip')
for f in z.namelist():
    if f.endswith('.py'):
        txt = z.read(f).decode('utf-8', errors='replace')
        if 'prediction.png' in txt or 'loss_curve.png' in txt:
            print(f"Found in {f}")
