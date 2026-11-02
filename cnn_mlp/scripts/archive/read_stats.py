import zipfile

z = zipfile.ZipFile('results/run_40gb_CNN_LSTM/results.zip')
log_txt = z.read('results/out/final.log').decode('utf-8', errors='replace')
for l in log_txt.splitlines():
    if 'First epoch:' in l or 'shots' in l.lower() or 'batch' in l.lower():
        print(l[:90])
