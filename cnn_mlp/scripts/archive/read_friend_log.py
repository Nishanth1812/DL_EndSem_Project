import zipfile

z = zipfile.ZipFile('results/run_40gb_CNN_LSTM/results.zip')
txt = z.read('results/out/final.log').decode('utf-8', errors='replace')
lines = txt.splitlines()
for i in range(min(120, len(lines))):
    clean = lines[i].encode('ascii', errors='replace').decode('ascii')
    print(f"{i+1:3d}: {clean}")
