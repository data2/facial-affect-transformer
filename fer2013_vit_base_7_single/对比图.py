import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_exp_comparison():
    # 1. 定义实验路径和对应的标签名
    experiments = {
        "Exp1: Vanilla": "exp1_vanilla/train_log.csv",
        "Exp3: ViT+LLRD": "exp3_vit_llrd/train_log.csv",
        "Exp5: Refined V1": "exp5_refined_final/train_log.csv",
        "Exp5: Refined V2 (Ours)": "exp5_refined_v2/train_log.csv"
    }

    # 2. 设置绘图风格（学术常用清爽风格）
    plt.figure(figsize=(12, 7), dpi=300)
    sns.set_theme(style="whitegrid")
    
    # 颜色组合，突出 V2
    colors = ['#95a5a6', '#3498db', '#e67e22', '#e74c3c'] 
    
    found_data = False

    # 3. 循环读取数据
    for i, (label, path) in enumerate(experiments.items()):
        if not os.path.exists(path):
            print(f"⚠️ 跳过：找不到路径 {path}")
            continue
        
        try:
            df = pd.read_csv(path)
            
            # 自动识别列名（兼容不同版本的 csv 列名）
            col_map = {c.lower(): c for c in df.columns}
            x_col = col_map.get('epoch')
            # 优先寻找 EMA-Val Acc 或 Val Acc，最后找 Acc
            y_col = col_map.get('ema-val acc') or col_map.get('val acc') or col_map.get('acc')

            if x_col and y_col:
                # 绘制折线图，V2 使用加粗线
                linewidth = 3 if "V2" in label else 1.5
                plt.plot(df[x_col], df[y_col], label=label, color=colors[i], linewidth=linewidth, alpha=0.9)
                found_data = True
            else:
                print(f"⚠️ 警告：{path} 中未找到 epoch 或 acc 列")
        except Exception as e:
            print(f"❌ 读取 {path} 失败: {e}")

    if not found_data:
        print("❌ 未能绘制任何数据，请检查文件路径及列名")
        return

    # 4. 图表细节优化
    plt.title("FER2013 Accuracy Convergence Comparison", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Epochs", fontsize=13)
    plt.ylabel("Validation Accuracy (EMA)", fontsize=13)
    
    # 设置 74% 目标参考线
    plt.axhline(y=0.74, color='#2c3e50', linestyle='--', alpha=0.5, label='SOTA Target (0.74)')
    
    plt.legend(fontsize=11, frameon=True, loc='lower right')
    plt.tight_layout()

    # 5. 保存并展示
    save_path = "exp_comparison_curves.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 对比图已生成并保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_exp_comparison()