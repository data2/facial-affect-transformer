import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

def parse_report(file_path):
    """从 classification_report 文件中提取各类别的 F1-score"""
    f1_scores = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[2:9]:  # 提取 0-6 类别的数据行
            parts = re.split(r'\s+', line.strip())
            f1_scores.append(float(parts[3])) # F1-score 在第4列
    return f1_scores

# 1. 准备数据
reports = {
    'Exp1: Vanilla': './exp1_vanilla/best_report.txt',
    'Exp3: LLRD+EMA': './exp3_vit_llrd/best_report.txt',
    'Exp5-r1: SpatialHead': './exp5_refined_final/best_report.txt',
    'Exp5-r2: Refined-V2': './exp5_refined_v2/best_report.txt'
}

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
data = {name: parse_report(path) for name, path in reports.items()}
df = pd.DataFrame(data, index=class_names)

# 2. 绘图设置
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
x = np.arange(len(class_names))
width = 0.2

# 绘制柱状图
for i, (label, scores) in enumerate(df.items()):
    ax.bar(x + i*width - width*1.5, scores, width, label=label, alpha=0.9)

# 3. 美化图表
ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Class-wise F1-Score Evolution Across Experiments', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(class_names, fontsize=11)
ax.set_ylim(0.5, 1.0) # 表情识别通常在0.5以上，这样对比更明显
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True)

# 标注 Accuracy 增益
plt.annotate('Accuracy Boost:\n72.05% → 74.19%', xy=(3, 0.92), xytext=(4, 0.95),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1),
             fontsize=10, fontweight='bold', color='darkred')

plt.tight_layout()
plt.savefig('ablation_study_f1.png')
print("✅ 图像已生成：ablation_study_f1.png")
plt.show()
