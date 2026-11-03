import zipfile

z = zipfile.ZipFile('results/run_40gb_CNN_MLP/results-2.zip')
code = z.read('tokamark_baseline/run_training.py').decode('utf-8', errors='replace')
for l in code.splitlines()[:60]:
    print(l)
