import os

def parse_mlp(path):
    rows = {}
    with open(path) as f:
        in_sweep = False
        for line in f:
            if '## Full sweep' in line:
                in_sweep = True
                continue
            if not in_sweep:
                continue
            if line.startswith('|') and not line.startswith('| pass') and not line.startswith('| task') and not line.startswith('|---'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 6:
                    pass_name = f"{parts[0]}|{parts[1]}|{parts[2]}"
                    nrmse = float(parts[3])
                    rs = float(parts[5])
                    rows[pass_name] = (nrmse, rs)
    return rows

def parse_lstm(path):
    rows = {}
    with open(path) as f:
        in_sweep = False
        for line in f:
            if '## Full sweep' in line:
                in_sweep = True
                continue
            if not in_sweep:
                continue
            if line.startswith('|') and not line.startswith('| pass') and not line.startswith('| task') and not line.startswith('|---'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 7:
                    pass_name = f"{parts[0]}|{parts[1]}|{parts[2]}"
                    nrmse = float(parts[4])
                    rs = float(parts[6])
                    rows[pass_name] = (nrmse, rs)
    return rows

mlp = parse_mlp('results/robustness/robustness.md')
lstm = parse_mlp_lstm = parse_lstm('results/robustness_lstm/robustness/robustness.md')

print(f"Matched {len(mlp)} MLP passes, {len(lstm)} LSTM passes.")

print("\n" + "=" * 90)
print("HEAD-TO-HEAD ROBUSTNESS BENCHMARK: CNN+MLP vs. CNN+LSTM (63 TASKS)")
print("=" * 90)
print("%-35s | %-10s | %-8s | %-10s | %-8s | %-8s | %s" % 
      ("Scenario / Pass Name", "MLP NRMSE", "MLP RS", "LSTM NRMSE", "LSTM RS", "Winner", "Delta (Absolute)"))
print("-" * 90)

mlp_wins = 0
lstm_wins = 0

for pass_name in sorted(mlp.keys()):
    if pass_name in lstm:
        m_nrmse, m_rs = mlp[pass_name]
        l_nrmse, l_rs = lstm[pass_name]
        if m_nrmse < l_nrmse:
            winner = "MLP"
            mlp_wins += 1
        else:
            winner = "LSTM"
            lstm_wins += 1
        delta = abs(m_nrmse - l_nrmse)
        print("%-35s | %-10.4f | %-8.3f | %-10.4f | %-8.3f | %-8s | %.4f" % 
              (pass_name, m_nrmse, m_rs, l_nrmse, l_rs, winner, delta))

print("-" * 90)
print(f"Overall Wins across 63 corrupted passes: MLP = {mlp_wins}, LSTM = {lstm_wins}")
