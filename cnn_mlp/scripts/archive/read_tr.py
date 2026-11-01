import zipfile

z = zipfile.ZipFile('results/run_40gb_CNN_MLP/results-2.zip')
tr_code = z.read('tokamark_baseline/src/trainer.py').decode('utf-8', errors='replace')
lines = tr_code.splitlines()
for i in range(65, 145):
    if i < len(lines):
        # replace non-ascii
        clean_l = lines[i].encode('ascii', errors='replace').decode('ascii')
        print(f"{i+1:3d}: {clean_l}")
