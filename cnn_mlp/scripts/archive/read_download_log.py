import zipfile

z = zipfile.ZipFile('results/run_40gb_CNN_LSTM/results.zip')
txt = z.read('results/download_test.log').decode('utf-8', errors='replace')
print(txt)
