import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# 设置学术字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix' 

def plot_fer_origin_vector(csv_path):
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return

    # 读取日志
    df = pd.read_csv(csv_path)
    epochs = df['epoch'].values
    train_loss = df['train_loss'].values
    val_loss = df['val_loss'].values
    val_acc = df['val_acc'].values
    
    # 趋势平滑线
    ema_val_acc = pd.Series(val_acc).ewm(span=8).mean().values

    # 自动寻找最高点
    best_idx = np.argmax(val_acc)
    best_ep = epochs[best_idx]
    best_val = val_acc[best_idx]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # --- 1. 绘制 Loss ---
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color='#d62728', fontsize=12)
    ax1.plot(epochs, train_loss, color='#d62728', alpha=0.3, label='Train Loss', lw=1.5)
    ax1.plot(epochs, val_loss, color='#d62728', linestyle='--', linewidth=1, label='Val Loss')
    ax1.tick_params(axis='y', labelcolor='#d62728')
    # 根据数据动态调整 Loss 轴范围
    ax1.set_ylim(min(train_loss)*0.8, max(val_loss)*1.1)

    # --- 2. 绘制 Accuracy ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='#1f77b4', fontsize=12)
    ax2.plot(epochs, val_acc, color='#1f77b4', alpha=0.3, label='Val Acc', lw=1)
    ax2.plot(epochs, ema_val_acc, color='#1f77b4', linewidth=2, label='Trend (Smooth)', linestyle='-')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.set_ylim(0.4, 0.8) # 保持 Y 轴范围一致，方便与 Exp 5 对比

    # --- 3. 标注最高点 ---
    ax2.scatter(best_ep, best_val, color='black', s=30, zorder=5)
    ax2.annotate(f'Best: {best_val:.2%}', 
                 xy=(best_ep, best_val), 
                 xytext=(best_ep+5, best_val-0.05),
                 fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1", color='black'))

    # --- 4. 图例与标题 ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True, fontsize=9)

    plt.title('Vanilla ViT-Base Training Dynamics: Convergence and EMA Smoothing (FER2013)', fontsize=14, pad=15)
    ax1.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()

    # 保存为 SVG 矢量图
    save_name = 'FER2013_Exp1_Origin.svg'
    plt.savefig(save_name, format='svg', bbox_inches='tight')
    print(f">>> Exp 1 矢量图已生成: {save_name}")
    plt.show()

# 执行 (请确保路径正确)
plot_fer_origin_vector('./exp1_vanilla/train_log.csv')
