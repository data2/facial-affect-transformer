import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_fer_comparison():
    # 1. 实验配置：{ 标签名: 文件夹路径 }
    experiments = {
        "Exp1: Vanilla": "exp1_vanilla",
        "Exp3: ViT+LLRD": "exp3_vit_llrd",
        "Exp5: Refined V1": "exp5_refined_final",
        "Exp5: Refined V2 (Ours)": "exp5_refined_v2"
    }

    # 2. 设置绘图风格
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), dpi=300)
    
    # 颜色配置：V2 用最醒目的深红色
    colors = ['#7f8c8d', '#3498db', '#f39c12', '#d35400']
    
    for i, (label, folder) in enumerate(experiments.items()):
        csv_path = os.path.join(folder, "train_log.csv")
        if not os.path.exists(csv_path):
            print(f"跳过：找不到 {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        
        # 绘制 Accuracy (左图)
        linewidth = 2.5 if "V2" in label else 1.2
        ax1.plot(df['epoch'], df['val_acc'], label=label, color=colors[i], linewidth=linewidth)
        
        # 绘制 F1 Score (右图)
        ax2.plot(df['epoch'], df['f1'], label=label, color=colors[i], linewidth=linewidth)

    # 3. 细节优化：Accuracy 图
    ax1.set_title("Validation Accuracy Comparison", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.axhline(y=0.73, color='gray', linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # 4. 细节优化：F1 Score 图
    ax2.set_title("F1 Score Comparison", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("F1 Score", fontsize=12)
    ax2.axhline(y=0.72, color='gray', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("exp_comparison_report.png")
    print("✅ 对比图已生成：exp_comparison_report.png")
    plt.show()

if __name__ == "__main__":
    plot_fer_comparison()
