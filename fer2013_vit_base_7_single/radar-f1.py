import numpy as np
import matplotlib.pyplot as plt

# 1. 准备真实 F1-score 数据 (来自你的 best_report.txt)
# 顺序：生气, 厌恶, 恐惧, 开心, 悲伤, 惊讶, 中性
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
exp1_f1 = [0.6455, 0.7273, 0.5828, 0.8860, 0.6807, 0.6122, 0.8263]
exp5_f1 = [0.6694, 0.7596, 0.6027, 0.9082, 0.7155, 0.6328, 0.8311]

# 闭合雷达图（首尾相连）
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
exp1_f1 += exp1_f1[:1]
exp5_f1 += exp5_f1[:1]
angles += angles[:1]

# 2. 绘图设置
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 绘制 Exp 1 (基准)
ax.plot(angles, exp1_f1, color='#1a9641', linewidth=2, label='Vanilla ViT', marker='o')
ax.fill(angles, exp1_f1, color='#1a9641', alpha=0.1)

# 绘制 Exp 5-v3 (改进)
ax.plot(angles, exp5_f1, color='#2b83ba', linewidth=2, label='Ours', marker='s')
ax.fill(angles, exp5_f1, color='#2b83ba', alpha=0.2)

# 3. 优化坐标轴与标注
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 设置类标
plt.xticks(angles[:-1], labels, fontsize=12)

# 设置刻度 (F1 通常在 0.5 到 1.0 之间波动)
ax.set_rlabel_position(0)
plt.yticks([0.6, 0.7, 0.8, 0.9], ["0.6", "0.7", "0.8", "0.9"], color="grey", size=10)
plt.ylim(0.5, 0.95)

# 图例与标题
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=11)
plt.title('F1-Score Comparison per Class', size=15, pad=20)

# 4. 保存为 SVG 矢量格式
plt.tight_layout()
plt.savefig('Radar_F1_Comparison.svg', format='svg', bbox_inches='tight')
plt.show()
