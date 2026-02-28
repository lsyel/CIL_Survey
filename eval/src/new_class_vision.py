import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
# Data Extraction
stages = ['Task 1', 'Task 2', 'Task 3', 'Task 4']
x = np.arange(len(stages))
width = 0.25

# Accuracy on NEW classes only (Plasticity)
wa_moe_new = [99.20, 98.35, 94.85, 96.35]
wa_new = [99.20, 98.25, 81.35, 88.10]
memo_new = [99.15, 98.35, 90.80, 95.75]

# Plotting
plt.figure(figsize=(9, 5.5)) 

# Bars
bar1 = plt.bar(x - width, wa_moe_new, width, label='WA-MoE (Ours)', color='#d62728', edgecolor='black', alpha=0.8) 
bar2 = plt.bar(x, wa_new, width, label='WA', color='#1f77b4', edgecolor='black', alpha=0.8)       
bar3 = plt.bar(x + width, memo_new, width, label='MEMO', color='#2ca02c', edgecolor='black', alpha=0.8) 

# Labels and Title
plt.ylabel('对新学习类别任务准确率(%)', fontsize=12)
plt.xticks(x, stages, fontsize=11)

# Set y-limit strictly to 100
plt.ylim(75, 100) 

# Legend: Moved back to 'upper right', reduced font size significantly so it fits in the corner
plt.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        # Adjusted text height slightly so it fits below the 100 line
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=8.5, rotation=0)

add_labels(bar1)
add_labels(bar2)
add_labels(bar3)

plt.tight_layout()
plt.savefig('plasticity_comparison_upper_right.png', dpi=300)