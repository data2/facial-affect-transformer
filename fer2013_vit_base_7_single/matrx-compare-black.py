import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ==========================
# 学术论文黑白风格
# ==========================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11



def plot_academic_cm(data, title, save_name):

    # FER2013 类别顺序
    classes = [
        'Angry',
        'Disgust',
        'Fear',
        'Happy',
        'Sad',
        'Surprise',
        'Neutral'
    ]


    # ==========================
    # 行归一化
    # ==========================
    cm_norm = (
        data.astype('float')
        /
        data.sum(axis=1)[:, np.newaxis]
    )


    # ==========================
    # 创建画布
    # ==========================
    plt.figure(
        figsize=(7, 6)
    )


    # ==========================
    # 黑白热力图
    # ==========================
    ax = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Greys",

        square=True,

        linewidths=0.5,
        linecolor='white',

        xticklabels=classes,
        yticklabels=classes,

        cbar_kws={
            'shrink': 0.8
        },

        annot_kws={
            "size": 10
        }
    )


    # ==========================
    # 自动调整百分比文字颜色
    # ==========================
    threshold = cm_norm.max() * 0.55

    for text, value in zip(
        ax.texts,
        cm_norm.flatten()
    ):
        if value > threshold:
            text.set_color('white')
        else:
            text.set_color('black')



    # ==========================
    # 标题和坐标
    # ==========================
    plt.title(
        title,
        fontsize=15,
        pad=15
    )


    plt.xlabel(
        'Predicted Label',
        fontsize=13
    )


    plt.ylabel(
        'True Label',
        fontsize=13
    )


    plt.xticks(
        rotation=45,
        ha='right'
    )


    plt.yticks(
        rotation=0
    )


    plt.tight_layout()



    # ==========================
    # 保存 SVG
    # ==========================
    plt.savefig(
        f"{save_name}.svg",
        format='svg',
        bbox_inches='tight'
    )


    print(
        f"Success: {save_name}.svg saved."
    )


    plt.show()



# =================================================
# Exp 1: Vanilla ViT
# =================================================

data_exp1 = np.array([
    [610, 5, 80, 25, 140, 20, 78],
    [2, 76, 2, 0, 20, 2, 9],
    [120, 5, 540, 45, 210, 40, 64],
    [20, 0, 30, 1585, 45, 35, 59],
    [105, 5, 115, 30, 903, 15, 60],
    [15, 2, 60, 55, 35, 761, 319],
    [35, 2, 45, 35, 15, 2, 697]
])



# =================================================
# Exp 5-v3: Ours
# =================================================

data_exp5r3 = np.array([
    [665, 2, 62, 12, 132, 8, 77],
    [2, 79, 1, 0, 18, 2, 9],
    [102, 2, 565, 21, 198, 32, 104],
    [15, 0, 22, 1593, 38, 42, 64],
    [95, 3, 88, 18, 937, 12, 80],
    [12, 1, 45, 48, 22, 767, 352],
    [28, 1, 35, 32, 12, 2, 721]
])



# ==========================
# 执行绘图
# ==========================

plot_academic_cm(
    data_exp1,
    "Confusion Matrix: Vanilla ViT",
    "CM_Exp1"
)


plot_academic_cm(
    data_exp5r3,
    "Confusion Matrix: Ours",
    "CM_Exp5r3"
)