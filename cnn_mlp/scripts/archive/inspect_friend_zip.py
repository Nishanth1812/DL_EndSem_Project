import zipfile
import matplotlib.pyplot as plt

z = zipfile.ZipFile('results/run_40gb_CNN_LSTM/results.zip')
# check if there are other files or text
print("Files in friend results:")
for f in z.namelist():
    if 'log' in f or 'yaml' in f or 'csv' in f:
        print(f, z.getinfo(f).file_size)
