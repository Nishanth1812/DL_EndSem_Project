import zipfile

z = zipfile.ZipFile('results/run_40gb_CNN_MLP/results-2.zip')
mlp_code = z.read('tokamark_baseline/src/multi_conv_mlp_model.py').decode('utf-8', errors='replace')
lstm_code = z.read('tokamark_baseline/src/multi_conv_lstm_model.py').decode('utf-8', errors='replace')

print("=== MLP Model Head ===")
for l in mlp_code.splitlines()[:40]:
    print(l)

print("\n=== LSTM Model Head ===")
for l in lstm_code.splitlines()[:40]:
    print(l)
