import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ================= 全局论文风格 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# 全局字体（放大，适合论文）
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# ================= 真实数据区 =================
folds = np.arange(1, 11)

# 1. 十折准确率
accs = np.array([
    0.9758, 0.9758, 0.9919, 1.0000, 0.9597,
    0.9756, 0.9675, 0.9837, 0.9756, 1.0000
])
mean_acc = np.mean(accs)

# 2. 类别标签
classes = [
    'Anger',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

# 3. 汇总混淆矩阵
cm_data = np.array([
    [121, 0, 0, 0, 11, 2, 0],
    [0, 172, 0, 0, 2, 0, 0],
    [1, 0, 74, 0, 0, 0, 0],
    [0, 0, 0, 206, 0, 0, 0],
    [3, 1, 0, 1, 303, 0, 0],
    [0, 0, 0, 0, 3, 81, 0],
    [0, 0, 0, 0, 1, 0, 246]
])

# ==========================================================
# 图4 十折交叉验证准确率
# ==========================================================
def draw_fig4_accuracy():

    plt.figure(figsize=(9, 5))

    # 黑白配色
    colors = ['lightgray'] * len(folds)

    bars = plt.bar(
        folds,
        accs * 100,
        color=colors,
        edgecolor='black',
        linewidth=1.0
    )

    plt.axhline(
        y=mean_acc * 100,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label=f'Mean: {mean_acc*100:.2f}%'
    )

    plt.ylim(90, 103)
    plt.xticks(folds)

    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.xlabel('Fold Index', fontsize=16)
    plt.title(
        '10-Fold Cross-Validation Accuracy on CK+',
        fontsize=18,
        pad=18
    )

    # 数值标签
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.2,
            f'{h:.2f}',
            ha='center',
            fontsize=12
        )

    plt.legend(loc='lower right')

    plt.grid(
        axis='y',
        linestyle=':',
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        'fig4_CK_Accuracy.svg',
        format='svg'
    )

    plt.close()

    print("图4已生成：fig4_CK_Accuracy.svg")


# ==========================================================
# 图5 Softmax概率分布
# ==========================================================
def draw_fig5_softmax():

    probs = [
        0.38,
        0.02,
        0.03,
        0.01,
        0.45,
        0.08,
        0.03
    ]

    plt.figure(figsize=(9, 5))

    x = np.arange(len(classes))

    # 黑白配色
    colors = ['lightgray'] * len(classes)

    # Ground Truth
    colors[0] = 'dimgray'

    # Predicted
    colors[4] = 'black'

    plt.bar(
        x,
        probs,
        color=colors,
        edgecolor='black',
        linewidth=1.0
    )

    plt.title(
        'Softmax Probability Distribution (Bimodal Characteristic)',
        fontsize=18,
        pad=18
    )

    plt.xticks(
        x,
        classes,
        rotation=20,
        fontsize=14
    )

    plt.ylabel(
        'Probability',
        fontsize=16
    )

    plt.ylim(0, 0.6)

    plt.annotate(
        'Predicted (Wrong)',
        xy=(4, 0.45),
        xytext=(4.5, 0.53),
        fontsize=13,
        arrowprops=dict(
            facecolor='black',
            shrink=0.05,
            width=1,
            headwidth=5
        )
    )

    plt.annotate(
        'Ground Truth',
        xy=(0, 0.38),
        xytext=(0.5, 0.49),
        fontsize=13,
        arrowprops=dict(
            facecolor='black',
            shrink=0.05,
            width=1,
            headwidth=5
        )
    )

    plt.grid(
        axis='y',
        linestyle='--',
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        'fig5_Softmax_Bimodal.svg',
        format='svg'
    )

    plt.close()

    print("图5已生成：fig5_Softmax_Bimodal.svg")


# ==========================================================
# 图6 混淆矩阵
# ==========================================================
def draw_fig6_confusion():

    plt.figure(figsize=(9, 7))

    cm_percent = cm_data.astype(float)
    cm_percent = cm_percent / cm_percent.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.2%',
        cmap='Greys',
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"size": 12},
        cbar=True
    )

    plt.title(
        'Aggregated Confusion Matrix on CK+',
        fontsize=18,
        pad=18
    )

    plt.xlabel(
        'Predicted Label',
        fontsize=16
    )

    plt.ylabel(
        'True Label',
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        'fig6_CK_Confusion.svg',
        format='svg'
    )

    plt.close()

    print("图6已生成：fig6_CK_Confusion.svg")


# ==========================================================
# 执行
# ==========================================================
draw_fig4_accuracy()
draw_fig5_softmax()
draw_fig6_confusion()