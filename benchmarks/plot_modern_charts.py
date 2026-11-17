import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Set high-quality styling
plt.rcParams['font.family'] = ['Leelawadee UI', 'Tahoma', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

out_dir = 'Thai-Temporal-Normalizer/docs/assets'
os.makedirs(out_dir, exist_ok=True)

# Data Summary for all 6 systems
models = [
    'Thai-Temporal-Normalizer (Ours)',
    'Gemini 2.5 Flash (Cloud LLM)',
    'Typhoon 2.5 30B (Thai LLM)',
    'Simple Regex Baseline',
    'PyThaiNLP (v5.1.2)',
    'dateparser (v1.5.0)'
]
accuracy = [100.00, 73.33, 47.33, 26.67, 18.67, 6.00]
latency_p95 = [0.20, 1181.5, 2753.9, 0.05, 1.8, 12.4]

# -------------------------------------------------------------
# CHART 1: Modern Executive Dashboard (Horizontal Hero Card Style)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=300)
fig.patch.set_facecolor('#FAFAFB')
ax1.set_facecolor('#FAFAFB')
ax2.set_facecolor('#FAFAFB')

# Modern Palette
# Emerald green for winner, crisp slate/coral/blue for others
colors = ['#10B981', '#3B82F6', '#8B5CF6', '#94A3B8', '#F59E0B', '#EF4444']

y_pos = np.arange(len(models))

# Left Subplot: Accuracy
bars1 = ax1.barh(y_pos, accuracy, color=colors, height=0.55, edgecolor='none')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(models, fontsize=11, fontweight='bold', color='#1E293B')
ax1.invert_yaxis()  # Top-down order
ax1.set_xlim(0, 118)
ax1.set_xlabel('Accuracy on Thai Temporal Baseline (%)', fontsize=11, fontweight='bold', color='#475569')
ax1.set_title('Parsing Accuracy (Higher is Better)', fontsize=13, fontweight='bold', color='#0F172A', pad=12)

# Value annotations
for bar in bars1:
    w = bar.get_width()
    ax1.text(w + 2, bar.get_y() + bar.get_height()/2.0, f'{w:.1f}%', 
             ha='left', va='center', fontsize=11, fontweight='bold', color='#0F172A')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#CBD5E1')
ax1.spines['bottom'].set_color('#CBD5E1')
ax1.grid(axis='x', linestyle='--', alpha=0.4, color='#94A3B8')

# Right Subplot: Latency (Log Scale)
bars2 = ax2.barh(y_pos, latency_p95, color=colors, height=0.55, edgecolor='none')
ax2.set_yticks(y_pos)
ax2.set_yticklabels([])  # Hide duplicate labels
ax2.invert_yaxis()
ax2.set_xscale('log')
ax2.set_xlabel('p95 Latency in milliseconds (Log Scale - Lower is Better)', fontsize=11, fontweight='bold', color='#475569')
ax2.set_title('Response Speed (p95)', fontsize=13, fontweight='bold', color='#0F172A', pad=12)

# SLA Target Line
ax2.axvline(5.0, color='#DC2626', linestyle=':', linewidth=2, label='SLA Budget: 5.0 ms')
ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=10)

for bar in bars2:
    w = bar.get_width()
    if w < 1:
        txt = f'{w:.2f} ms (Lightning)'
    elif w < 100:
        txt = f'{w:.1f} ms'
    else:
        txt = f'{w:.0f} ms (~{w/1000:.1f}s)'
    ax2.text(w * 1.3, bar.get_y() + bar.get_height()/2.0, txt, 
             ha='left', va='center', fontsize=10, fontweight='bold', color='#0F172A')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#CBD5E1')
ax2.spines['bottom'].set_color('#CBD5E1')
ax2.grid(axis='x', linestyle='--', alpha=0.4, color='#94A3B8')

plt.suptitle('Thai-Temporal-Normalizer vs LLMs & Traditional NLP Libraries', 
             fontsize=16, fontweight='bold', color='#0F172A', y=0.98)
plt.tight_layout()
p1 = os.path.join(out_dir, 'benchmark_executive_summary.png')
plt.savefig(p1, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {p1}")

# -------------------------------------------------------------
# CHART 2: Efficiency Frontier (Accuracy vs Speed Scatter Plot)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
fig.patch.set_facecolor('#FAFAFB')
ax.set_facecolor('#FAFAFB')

# Scatter Points
scatter_colors = ['#10B981', '#3B82F6', '#8B5CF6', '#94A3B8', '#F59E0B', '#EF4444']
sizes = [400, 280, 280, 180, 180, 180]

ax.set_xscale('log')
ax.set_xlim(0.02, 8000)
ax.set_ylim(-5, 115)

for i in range(len(models)):
    ax.scatter(latency_p95[i], accuracy[i], color=scatter_colors[i], s=sizes[i], 
               edgecolor='white', linewidth=2.5, zorder=5)

# Annotate each point
offsets = [
    (0.24, 98, 'Thai-Temporal-Normalizer (Ours)\n100% Acc | 0.20 ms | 0฿ (Offline)', '#065F46', 'bottom'),
    (1500, 71, 'Gemini 2.5 Flash\n73.3% Acc | 1.18s | Paid API', '#1E40AF', 'bottom'),
    (3100, 45, 'Typhoon 2.5 30B\n47.3% Acc | 2.75s | Paid API', '#5B21B6', 'bottom'),
    (0.06, 23, 'Simple Regex\n26.7% | 0.05ms', '#475569', 'top'),
    (2.2, 16, 'PyThaiNLP\n18.7% | 1.8ms', '#92400E', 'top'),
    (15.0, 3, 'dateparser\n6.0% | 12.4ms', '#991B1B', 'top')
]

for x_pt, y_pt, label, col, va_align in offsets:
    ax.text(x_pt, y_pt, label, fontsize=10, fontweight='bold', color=col, va=va_align)

# Optimal Zone Highlight Box (Green Quadrant)
ax.axvspan(0.02, 5.0, ymin=0.75, ymax=1.0, color='#D1FAE5', alpha=0.45, zorder=1)
ax.text(0.03, 107, 'IDEAL PRODUCTION ZONE (Sub-5ms SLA & >90% Acc)', 
        fontsize=10.5, fontweight='bold', color='#047857')

# SLA Marker Line
ax.axvline(5.0, color='#EF4444', linestyle='--', linewidth=1.5, zorder=2)
ax.text(5.5, 15, 'SLA Threshold (5.0 ms)', color='#EF4444', fontsize=9.5, fontweight='bold')

ax.set_xlabel('Latency p95 (ms) [Log Scale - Faster to the Left] ◄──', fontsize=11, fontweight='bold', color='#334155')
ax.set_ylabel('Accuracy on Thai Baseline (%) ──► Higher is Better', fontsize=11, fontweight='bold', color='#334155')
ax.set_title('Trade-Off Matrix: Accuracy vs Execution Speed (2026)', fontsize=14, fontweight='bold', color='#0F172A', pad=12)

ax.grid(True, linestyle='--', alpha=0.35, color='#94A3B8')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CBD5E1')
ax.spines['bottom'].set_color('#CBD5E1')

plt.tight_layout()
p2 = os.path.join(out_dir, 'accuracy_vs_latency_frontier.png')
plt.savefig(p2, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {p2}")

# -------------------------------------------------------------
# CHART 3: Clean Category Breakdown (Grouped Bar Chart)
# -------------------------------------------------------------
cats = [
    'Relative Day/Week\n(พรุ่งนี้, พุธหน้า)',
    'Thai 6-Hour Clock\n(บ่ายสาม, ตีสอง)',
    'Coarse Time\n(สายๆ, หัวค่ำ)',
    'Short BE Year\n(ปี 69, มี.ค. 67)',
    'Night Rollover\n(คืนวันศุกร์ ตีสอง)'
]

acc_ours_cat       = [100.0, 100.0, 100.0, 100.0, 100.0]
acc_gemini_cat     = [73.3,  93.3,  86.7,  70.0,  43.3]
acc_typhoon_cat    = [60.0,  100.0, 33.3,  20.0,  23.3]
acc_pythainlp_cat  = [23.3,  70.0,  0.0,   0.0,   0.0]
acc_dateparser_cat = [30.0,  0.0,   0.0,   0.0,   0.0]

fig, ax = plt.subplots(figsize=(14, 6.2), dpi=300)
fig.patch.set_facecolor('#FAFAFB')
ax.set_facecolor('#FAFAFB')

x = np.arange(len(cats))
w = 0.15

rects1 = ax.bar(x - 2*w, acc_ours_cat, w, label='Thai-Temporal-Normalizer (Ours)', color='#10B981', edgecolor='none')
rects2 = ax.bar(x - w,   acc_gemini_cat, w, label='Gemini 2.5 Flash', color='#3B82F6', edgecolor='none')
rects3 = ax.bar(x,       acc_typhoon_cat, w, label='Typhoon 2.5 (30B)', color='#8B5CF6', edgecolor='none')
rects4 = ax.bar(x + w,   acc_pythainlp_cat, w, label='PyThaiNLP (v5.1.2)', color='#F59E0B', edgecolor='none')
rects5 = ax.bar(x + 2*w, acc_dateparser_cat, w, label='dateparser (v1.5.0)', color='#EF4444', edgecolor='none')

ax.set_title('Linguistic Nuance Accuracy Breakdown (N=30 cases per category)', fontsize=14, fontweight='bold', color='#0F172A', pad=14)
ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color='#334155')
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=10.5, fontweight='bold', color='#1E293B')
ax.set_ylim(0, 118)
ax.grid(axis='y', linestyle='--', alpha=0.4, color='#94A3B8')
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=10)

# Add 100% tags for Ours
for rect in rects1:
    ax.text(rect.get_x() + rect.get_width()/2.0, rect.get_height() + 1.5, '100%', 
            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#065F46')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CBD5E1')
ax.spines['bottom'].set_color('#CBD5E1')

plt.tight_layout()
p3 = os.path.join(out_dir, 'category_nuance_breakdown.png')
plt.savefig(p3, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {p3}")
