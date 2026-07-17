import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# =======================
# 学术论文黑白风格
# =======================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11



def plot_fer_vector_graph(csv_path):

    # =======================
    # 读取日志
    # =======================
    df = pd.read_csv(csv_path)

    epochs = df['epoch'].values
    train_loss = df['train_loss'].values
    val_loss = df['val_loss'].values
    val_acc = df['val_acc'].values


    # EMA
    ema_val_acc = pd.Series(
        val_acc
    ).ewm(span=8).mean().values



    # =======================
    # 创建画布
    # =======================
    fig, ax1 = plt.subplots(
        figsize=(8.5, 5.2)
    )



    # =======================
    # Loss 左轴
    # =======================
    ax1.set_xlabel(
        'Epoch',
        fontsize=14
    )

    ax1.set_ylabel(
        'Loss',
        fontsize=14
    )


    # Train Loss
    ax1.plot(
        epochs,
        train_loss,
        color='0.65',
        linewidth=1.3,
        label='Train Loss'
    )


    # Val Loss
    ax1.plot(
        epochs,
        val_loss,
        color='black',
        linestyle='--',
        linewidth=1.2,
        label='Val Loss'
    )


    ax1.set_ylim(
        min(train_loss)*0.8,
        2.0
    )



    # =======================
    # Accuracy 右轴
    # =======================
    ax2 = ax1.twinx()


    ax2.set_ylabel(
        'Accuracy',
        fontsize=14
    )


    # Val Acc
    ax2.plot(
        epochs,
        val_acc,
        color='0.35',
        linewidth=1.2,
        label='Val Acc'
    )


    # EMA
    ax2.plot(
        epochs,
        ema_val_acc,
        color='black',
        linewidth=2.5,
        label='EMA-Val Acc'
    )


    ax2.set_ylim(
        0.4,
        0.8
    )



    # =======================
    # Warmup阶段
    # =======================
    ax1.axvline(
        x=5,
        color='gray',
        linestyle=':',
        linewidth=1
    )


    ax1.text(
        3,
        1.85,
        'Warmup',
        ha='center',
        fontsize=11,
        color='0.35',
        fontweight='bold'
    )



    # =======================
    # 最优点
    # =======================
    best_ep = 21
    best_val = 0.7421


    ax2.scatter(
        best_ep,
        best_val,
        color='black',
        s=45,
        zorder=5
    )


    ax2.annotate(
        f'SOTA: {best_val:.2%}',
        xy=(
            best_ep,
            best_val
        ),
        xytext=(
            best_ep+10,
            best_val-0.05
        ),
        fontsize=11,
        fontweight='bold',
        arrowprops=dict(
            arrowstyle='->',
            color='black',
            connectionstyle="arc3,rad=.2"
        )
    )



    # =======================
    # 图例
    # =======================
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()


    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='lower right',
        frameon=True,
        fontsize=11
    )



    # =======================
    # 标题
    # =======================
    plt.title(
        'Ours Training Dynamics\n'
        'Convergence and EMA Smoothing (FER2013)',
        fontsize=15,
        pad=12
    )



    # =======================
    # 网格
    # =======================
    ax1.grid(
        True,
        linestyle=':',
        linewidth=0.7,
        color='0.75',
        alpha=0.8
    )


    fig.tight_layout(
        rect=[0,0,1,0.93]
    )



    # =======================
    # 保存 SVG
    # =======================
    save_name = 'FER2013_Results.svg'


    plt.savefig(
        save_name,
        format='svg',
        bbox_inches='tight'
    )


    print(
        ">>> 矢量图已生成: FER2013_Results.svg"
    )


    plt.show()



# =======================
# 执行
# =======================
plot_fer_vector_graph(
    './exp5_refined_v3/run_8/train_log.csv'
)