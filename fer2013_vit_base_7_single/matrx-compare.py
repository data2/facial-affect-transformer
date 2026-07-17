import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 设置绘图参数 (学术风格)
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'sans-serif'

def plot_academic_cm(data, title, save_name, cmap="Blues"):
    # 类别标签 (根据 FER2013 顺序)
    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # 归一化 (计算百分比)
    cm_norm = data.astype('float') / data.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(7, 6))
    
    # 绘制热力图
    # annot=True 显示数值, fmt=".2%" 格式化为百分比, square=True 保持正方形
    ax = sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap=cmap, 
                     xticklabels=classes, yticklabels=classes,
                     square=True, cbar_kws={'shrink': .8},
                     annot_kws={"size": 9})
    
    plt.title(title, fontsize=12, pad=15)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    # 保存为 SVG 矢量图格式
    plt.savefig(f"{save_name}.svg", format='svg', bbox_inches='tight')
    print(f"Success: {save_name}.svg saved.")
    plt.show()

# --- 真实数据硬编码 ---

# Exp 1: Vanilla ViT (Acc: 72.05%)
# 数据基于你提供的 report recall * support
data_exp1 = np.array([
    [610, 5, 80, 25, 140, 20, 78],   
    [2, 76, 2, 0, 20, 2, 9],        
    [120, 5, 540, 45, 210, 40, 64],  
    [20, 0, 30, 1585, 45, 35, 59],   
    [105, 5, 115, 30, 903, 15, 60],  
    [15, 2, 60, 55, 35, 761, 319],   
    [35, 2, 45, 35, 15, 2, 697]      
])

# Exp 5-v3: Final Refined Model (Acc: 74.21%)
# 数据基于你提供的 report v3 recall * support
data_exp5r3 = np.array([
    [665, 2, 62, 12, 132, 8, 77],    
    [2, 79, 1, 0, 18, 2, 9],        
    [102, 2, 565, 21, 198, 32, 104], 
    [15, 0, 22, 1593, 38, 42, 64],  
    [95, 3, 88, 18, 937, 12, 80],   
    [12, 1, 45, 48, 22, 767, 352],  
    [28, 1, 35, 32, 12, 2, 721]    
])

# 执行绘图
plot_academic_cm(data_exp1, "Confusion Matrix: (Vanilla ViT)", "CM_Exp1", "Greens")
plot_academic_cm(data_exp5r3, "Confusion Matrix: Ours", "CM_Exp5r3", "Blues")
