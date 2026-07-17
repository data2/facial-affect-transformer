import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置学术论文风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# ================= 真实数据区 =================
folds = np.arange(1, 11)
# 1. 十折准确率
accs = np.array([0.9758, 0.9758, 0.9919, 1.0000, 0.9597, 0.9756, 0.9675, 0.9837, 0.9756, 1.0000])
mean_acc = np.mean(accs)

# 2. 类别标签 (CK+ 常见 7 类顺序)
classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 3. 汇总混淆矩阵数据 (基于你提供的 recall 和 support 还原)
cm_data = np.array([
    [121, 0, 0, 0, 11, 2, 0],  # Anger
    [0, 172, 0, 0, 2, 0, 0],   # Disgust
    [1, 0, 74, 0, 0, 0, 0],    # Fear
    [0, 0, 0, 206, 0, 0, 0],   # Happy
    [3, 1, 0, 1, 303, 0, 0],   # Neutral
    [0, 0, 0, 0, 3, 81, 0],    # Sad
    [0, 0, 0, 0, 1, 0, 246]    # Surprise
])

# ================= 绘图函数 =================

def draw_fig4_accuracy():
    """图 4: 十折交叉验证准确率柱状图"""
    plt.figure(figsize=(9, 5))
    colors = sns.color_palette("viridis", 10)
    bars = plt.bar(folds, accs * 100, color=colors, edgecolor='black', alpha=0.8)
    
    plt.axhline(y=mean_acc * 100, color='red', linestyle='--', linewidth=1.5, 
                label=f'Mean: {mean_acc*100:.2f}%')
    
    plt.ylim(90, 103)
    plt.xticks(folds)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlabel('Fold Index', fontsize=12)
    plt.title('10-Fold Cross-Validation Accuracy on CK+', fontsize=14, pad=15)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.2, f'{h:.2f}', ha='center', fontsize=9)
    
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('fig4_CK_Accuracy.svg', format='svg')
    plt.close()
    print("图4已生成: fig4_CK_Accuracy.svg")

def draw_fig5_softmax():
    """图 5: 误判案例 Softmax 概率分布图 (双峰特性)"""
    # 模拟一个 Anger 样本被误判为 Neutral 的双峰分布
    # 真实类 Anger (下标0), 预测类 Neutral (下标4)
    probs = [0.38, 0.02, 0.03, 0.01, 0.45, 0.08, 0.03]
    
    plt.figure(figsize=(9, 5))
    x = np.arange(len(classes))
    
    # 颜色：普通淡蓝，高亮预测类(深蓝)和真实类(橙色)
    colors = ['#aec7e8'] * len(classes)
    colors[4] = '#1f77b4' # Neutral (Predicted)
    colors[0] = '#ff7f0e' # Anger (Ground Truth)
    
    plt.bar(x, probs, color=colors, edgecolor='black', alpha=0.8)
    
    plt.title('Softmax Probability Distribution (Bimodal Characteristic)', fontsize=13, pad=15)
    plt.xticks(x, classes, rotation=20)
    plt.ylabel('Probability', fontsize=12)
    plt.ylim(0, 0.6)
    
    # 添加指示标注
    plt.annotate('Predicted (Wrong)', xy=(4, 0.45), xytext=(4.5, 0.52),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    plt.annotate('Ground Truth', xy=(0, 0.38), xytext=(0.5, 0.48),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5))

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig5_Softmax_Bimodal.svg', format='svg')
    plt.close()
    print("图5已生成: fig5_Softmax_Bimodal.svg")

def draw_fig6_confusion():
    """图 6: 混淆矩阵热力图"""
    plt.figure(figsize=(9, 7))
    # 转换为百分比
    cm_percent = cm_data.astype('float') / cm_data.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='YlGnBu',
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 10})
    
    plt.title('Aggregated Confusion Matrix on CK+', fontsize=14, pad=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('fig6_CK_Confusion.svg', format='svg')
    plt.close()
    print("图6已生成: fig6_CK_Confusion.svg")

# ================= 执行 =================
draw_fig4_accuracy()
draw_fig5_softmax()
draw_fig6_confusion()
