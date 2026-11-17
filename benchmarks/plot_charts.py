import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Set Matplotlib Thai Font & Style
plt.rcParams['font.family'] = ['Leelawadee UI', 'Tahoma', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Ensure output directory exists
out_dir = 'Thai-Temporal-Normalizer/docs/assets'
os.makedirs(out_dir, exist_ok=True)

# 1. Overall Accuracy & Latency Data
systems = [
    'dateparser\n(v1.5.0)',
    'PyThaiNLP\n(v5.1.2)',
    'Simple Regex\n(Baseline)',
    'Typhoon LLM\n(v2.5-30B)',
    'Thai-Temporal\nNormalizer (Ours)'
]
accuracy = [6.00, 18.67, 26.67, 47.33, 100.00]
latency_p95 = [12.4, 1.8, 0.05, 2753.9, 0.20]

# --- CHART 1: Overall Accuracy & Latency (Dual View) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Colors
colors = ['#e74c3c', '#e67e22', '#95a5a6', '#3498db', '#2ecc71']

# Accuracy Plot
bars1 = ax1.bar(systems, accuracy, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
ax1.set_title('Overall Accuracy (%) on Thai Baseline (N=150)', fontsize=13, fontweight='bold', pad=12)
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_ylim(0, 115)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Latency Plot (Log Scale)
bars2 = ax2.bar(systems, latency_p95, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
ax2.set_yscale('log')
ax2.set_title('p95 Latency (ms) - Log Scale (Lower is Better)', fontsize=13, fontweight='bold', pad=12)
ax2.set_ylabel('Latency p95 (ms) [Log Scale]', fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.axhline(5.0, color='red', linestyle=':', linewidth=1.5, label='SLA Target (5.0 ms)')
ax2.legend(loc='upper left', frameon=True)

for bar in bars2:
    yval = bar.get_height()
    if yval < 1:
        label = f'{yval:.2f} ms'
    elif yval < 100:
        label = f'{yval:.1f} ms'
    else:
        label = f'{yval:.0f} ms'
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval * 1.3, label, ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Thai-Temporal-Normalizer vs Existing Libraries & SOTA LLM (Benchmark 2026)', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
chart1_path = os.path.join(out_dir, 'overall_benchmark_comparison.png')
plt.savefig(chart1_path, bbox_inches='tight')
plt.close()
print(f"Chart 1 saved to: {chart1_path}")

# --- CHART 2: Per-Category Accuracy Breakdown ---
categories = [
    'Relative Day/Week\n(พรุ่งนี้, พุธหน้า)',
    'Thai 6-Hour Clock\n(บ่ายสาม, ตีสอง)',
    'Coarse Time\n(สายๆ, หัวค่ำ)',
    'Short Buddhist Era\n(ปี 69, มี.ค. 67)',
    'Night Rollover\n(คืนวันศุกร์ ตีสอง)'
]

acc_dateparser = [30.0, 0.0, 0.0, 0.0, 0.0]
acc_pythainlp  = [23.3, 70.0, 0.0, 0.0, 0.0]
acc_regex      = [33.3, 83.3, 0.0, 0.0, 16.7]
acc_typhoon    = [60.0, 100.0, 33.3, 20.0, 23.3]
acc_ours       = [100.0, 100.0, 100.0, 100.0, 100.0]

x = np.arange(len(categories))
width = 0.16

fig, ax = plt.subplots(figsize=(15, 6.5), dpi=300)

r1 = ax.bar(x - 2*width, acc_dateparser, width, label='dateparser (v1.5.0)', color='#e74c3c', edgecolor='black', linewidth=0.6)
r2 = ax.bar(x - width,   acc_pythainlp,  width, label='PyThaiNLP (v5.1.2)', color='#e67e22', edgecolor='black', linewidth=0.6)
r3 = ax.bar(x,           acc_regex,      width, label='Simple Regex', color='#95a5a6', edgecolor='black', linewidth=0.6)
r4 = ax.bar(x + width,   acc_typhoon,    width, label='Typhoon LLM (30B)', color='#3498db', edgecolor='black', linewidth=0.6)
r5 = ax.bar(x + 2*width, acc_ours,       width, label='Thai-Temporal-Normalizer (Ours)', color='#2ecc71', edgecolor='black', linewidth=1.2)

ax.set_title('Per-Category Accuracy Breakdown Across 5 Critical Thai Linguistic Nuances (N=30 each)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10.5, fontweight='bold')
ax.set_ylim(0, 120)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='upper right', frameon=True, fontsize=10.5)

# Highlight Zero Failure Zones
ax.axvspan(1.5, 4.5, color='#fff3cd', alpha=0.35, label='High Failure Zone for Existing Tools')

for r in [r4, r5]:
    for bar in r:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2.0, h + 2, f'{h:.0f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
chart2_path = os.path.join(out_dir, 'per_category_breakdown.png')
plt.savefig(chart2_path, bbox_inches='tight')
plt.close()
print(f"Chart 2 saved to: {chart2_path}")
