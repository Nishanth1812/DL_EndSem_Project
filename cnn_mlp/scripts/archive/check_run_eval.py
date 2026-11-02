import zipfile

z = zipfile.ZipFile('results/run_40gb_CNN_MLP/results-2.zip')
code = z.read('tokamark_baseline/run_evaluation.py').decode('utf-8', errors='replace')
for l in code.splitlines():
    if 'prediction.png' in l or 'plot' in l or 'loss' in l:
        print(l[:90])
