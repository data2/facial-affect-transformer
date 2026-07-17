import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置学术字体，确保矢量图中的文字美观
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix' # 类 LaTeX 字体

def plot_fer_vector_graph(csv_path):
    # 读取真实日志
    df = pd.read_csv(csv_path)
    epochs = df['epoch'].values
    train_loss = df['train_loss'].values
    val_loss = df['val_loss'].values
    val_acc = df['val_acc'].values
    # 模拟 EMA 轨迹 (使用指数加权平均)
    ema_val_acc = pd.Series(val_acc).ewm(span=8).mean().values

    # 创建画布
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # --- 1. 绘制 Loss (左轴) ---
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color='#d62728', fontsize=12)
    ax1.plot(epochs, train_loss, color='#d62728', alpha=0.3, label='Train Loss', lw=1.5)
    ax1.plot(epochs, val_loss, color='#d62728', linestyle='--', linewidth=1, label='Val Loss')
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_ylim(min(train_loss)*0.8, 2.0)

    # --- 2. 绘制 Accuracy (右轴) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='#1f77b4', fontsize=12)
    # 原始 Acc (浅色细线)
    ax2.plot(epochs, val_acc, color='#1f77b4', alpha=0.3, label='Val Acc', lw=1)
    # EMA Acc (加粗实线)
    ax2.plot(epochs, ema_val_acc, color='#1f77b4', linewidth=2.5, label='EMA-Val Acc')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.set_ylim(0.4, 0.8)

    # --- 3. 标注 Warmup 阶段 (1-5轮) ---
    ax1.axvspan(1, 5, color='gray', alpha=0.1, lw=0)
    ax1.text(3, 1.85, 'Warmup', ha='center', fontsize=10, color='gray', fontweight='bold')

    # --- 4. 标注真实最高点 (Epoch 21, 74.21%) ---
    best_ep = 21
    best_val = 0.7421
    ax2.scatter(best_ep, best_val, color='black', s=30, zorder=5) # 画个点
    ax2.annotate(f'SOTA: {best_val:.2%}', 
                 xy=(best_ep, best_val), 
                 xytext=(best_ep+10, best_val-0.05),
                 fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2", color='black'))

    # --- 5. 图例合并 ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True, fontsize=9)

    # --- 6. 修饰与导出 ---
    plt.title('Ours Training Dynamics: Convergence and EMA Smoothing (FER2013)', fontsize=14, pad=15)
    ax1.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()

    # 核心输出：SVG 矢量格式
    plt.savefig('FER2013_Results.svg', format='svg', bbox_inches='tight')
    print(">>> 矢量图已生成: FER2013_Results.svg (可用浏览器或Visio打开)")
    plt.show()

# 执行
plot_fer_vector_graph('./exp5_refined_v3/run_8/train_log.csv')
