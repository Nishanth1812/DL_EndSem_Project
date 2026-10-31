import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# -------------------------------------------------------------
# 1. Diagnostic Criticality Ranking
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.8), layout='constrained')

categories = [
    'Pickup Coils (BR, BZ, Bphi)',
    'Flux Loops (15)',
    'PF Shaping Coils (10)',
    'Central Solenoid (20)',
    'Saddle Flux Loops (12)',
    'Plasma Current (Ip)'
]
nrmse_vals = [0.148, 0.162, 0.169, 0.187, 0.224, 0.341]
degradation = [(v - 0.1419)/0.1419 * 100 for v in nrmse_vals]

colors = ['#2b5c8f', '#2b5c8f', '#4682b4', '#d97706', '#dc2626', '#991b1b']

bars = ax.barh(categories, degradation, color=colors, height=0.6, edgecolor='none')

ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('NRMSE Error Surge Above Clean Baseline (%)', fontsize=11, fontweight='bold', labelpad=6)
ax.set_title('Diagnostic Criticality: Test NRMSE Surge When Sensor Groups Fail', fontsize=12, fontweight='bold', pad=10)
ax.set_xlim(0, 160)

for bar, deg, nrmse in zip(bars, degradation, nrmse_vals):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
            f'+{deg:.1f}% (NRMSE {nrmse:.3f})', 
            va='center', ha='left', fontsize=9, fontweight='semibold', color='#1e293b')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cbd5e1')
ax.spines['bottom'].set_color('#cbd5e1')
ax.tick_params(colors='#1e293b', labelsize=9.5)

plt.savefig('PPT/figures/robustness_criticality.png', dpi=300)
plt.close()
print("Saved PPT/figures/robustness_criticality.png")

# -------------------------------------------------------------
# 2. Online Imputation Comparison (Zero vs Mean vs Forward)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.8), layout='constrained')

dropout_rates = ['Clean (p=0)', 'p = 0.10', 'p = 0.25', 'p = 0.50']
zero_fill = [0.1419, 0.1895, 0.2480, 0.3520]
mean_fill = [0.1419, 0.1620, 0.1985, 0.2610]
forward_fill = [0.1419, 0.1492, 0.1588, 0.1670]

x = np.arange(len(dropout_rates))
width = 0.24

b1 = ax.bar(x - width, zero_fill, width, label='Zero-Fill (Default Baseline)', color='#dc2626', edgecolor='none')
b2 = ax.bar(x, mean_fill, width, label='Mean-Fill (Dataset Mean)', color='#f59e0b', edgecolor='none')
b3 = ax.bar(x + width, forward_fill, width, label='Forward-Fill (Physics-Informed)', color='#10b981', edgecolor='none')

ax.axhline(0.1419, color='#64748b', linestyle='--', linewidth=1.0, label='Clean Baseline (0.1419)')
ax.set_ylabel('Test NRMSE (Lower is Better)', fontsize=11, fontweight='bold', labelpad=6)
ax.set_title('Online Fault Mitigation: Imputation Performance Under Random Dropouts', fontsize=12, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(dropout_rates, fontsize=10, fontweight='semibold')
ax.set_ylim(0, 0.40)
ax.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=9, loc='upper left')

# annotate forward-fill advantage
ax.annotate('Forward-Fill limits\ndegradation to <18%', 
            xy=(3 + width, 0.1670), xytext=(2.6, 0.26),
            arrowprops=dict(facecolor='#10b981', edgecolor='none', shrink=0.08, width=1.5, headwidth=6),
            fontsize=8.5, fontweight='bold', color='#047857',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecfdf5', edgecolor='#a7f3d0'))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cbd5e1')
ax.spines['bottom'].set_color('#cbd5e1')
ax.tick_params(colors='#1e293b', labelsize=9.5)

plt.savefig('PPT/figures/robustness_mitigation.png', dpi=300)
plt.close()
print("Saved PPT/figures/robustness_mitigation.png")
