import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 真实 F1-score 数据
# ===============================
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

exp1_f1 = [0.6455, 0.7273, 0.5828, 0.8860, 0.6807, 0.6122, 0.8263]
exp5_f1 = [0.6694, 0.7596, 0.6027, 0.9082, 0.7155, 0.6328, 0.8311]

# 闭合雷达图
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

exp1_f1 += exp1_f1[:1]
exp5_f1 += exp5_f1[:1]
angles += angles[:1]

# ===============================
# 全局论文风格
# ===============================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 14

# ===============================
# 绘图
# ===============================
fig, ax = plt.subplots(
    figsize=(8, 8),
    subplot_kw=dict(polar=True)
)

# 基准模型（深灰）
ax.plot(
    angles,
    exp1_f1,
    color='dimgray',
    linewidth=2,
    linestyle='--',
    marker='o',
    markersize=6,
    label='Vanilla ViT'
)

ax.fill(
    angles,
    exp1_f1,
    color='lightgray',
    alpha=0.25
)

# 改进模型（黑色）
ax.plot(
    angles,
    exp5_f1,
    color='black',
    linewidth=2,
    linestyle='-',
    marker='s',
    markersize=6,
    label='Ours'
)

ax.fill(
    angles,
    exp5_f1,
    color='gray',
    alpha=0.15
)

# ===============================
# 坐标轴
# ===============================
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

plt.xticks(
    angles[:-1],
    labels,
    fontsize=14
)

ax.set_rlabel_position(0)

plt.yticks(
    [0.6, 0.7, 0.8, 0.9],
    ["0.6", "0.7", "0.8", "0.9"],
    color="black",
    size=12
)

plt.ylim(0.5, 0.95)

# 网格改成灰色
ax.grid(
    color='gray',
    linestyle=':',
    linewidth=0.8,
    alpha=0.7
)

# 极坐标外圈
ax.spines['polar'].set_color('black')
ax.spines['polar'].set_linewidth(1.2)

# 图例
plt.legend(
    loc='upper right',
    bbox_to_anchor=(1.2, 1.1),
    fontsize=14,
    frameon=True
)

# 标题
plt.title(
    'F1-Score Comparison per Class',
    fontsize=18,
    pad=20
)

# ===============================
# 保存
# ===============================
plt.tight_layout()

plt.savefig(
    'Radar_F1_Comparison.svg',
    format='svg',
    bbox_inches='tight'
)

plt.show()