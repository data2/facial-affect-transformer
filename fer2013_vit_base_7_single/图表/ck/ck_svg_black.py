import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# ================= 数据 =================
folds = np.arange(1, 11)
accs = np.array([0.9758, 0.9758, 0.9919, 1.0000, 0.9597, 0.9756, 0.9675, 0.9837, 0.9756, 1.0000])
mean_acc = np.mean(accs)

classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cm_data = np.array([
    [121, 0, 0, 0, 11, 2, 0],
    [0, 172, 0, 0, 2, 0, 0],
    [1, 0, 74, 0, 0, 0, 0],
    [0, 0, 0, 206, 0, 0, 0],
    [3, 1, 0, 1, 303, 0, 0],
    [0, 0, 0, 0, 3, 81, 0],
    [0, 0, 0, 0, 1, 0, 246]
])

# ================= 图4 =================
def draw_fig4_accuracy():
    plt.figure(figsize=(9, 5))
    gray = ['#444444', '#555555', '#666666', '#777777', '#888888', 
            '#999999', '#777777', '#666666', '#555555', '#444444']
    bars = plt.bar(folds, accs * 100, color=gray, edgecolor='black', linewidth=1)
    plt.axhline(y=mean_acc * 100, color='black', linestyle='--', linewidth=1.5, 
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
    plt.savefig('fig4_CK_Accuracy.png', dpi=300)
    plt.close()

# ================= 图5 =================
def draw_fig5_softmax():
    probs = [0.38, 0.02, 0.03, 0.01, 0.45, 0.08, 0.03]
    plt.figure(figsize=(9, 5))
    x = np.arange(len(classes))
    colors = ['#cccccc'] * len(classes)
    colors[4] = '#333333'
    colors[0] = '#666666'
    plt.bar(x, probs, color=colors, edgecolor='black', linewidth=1.2)
    plt.title('Softmax Probability Distribution (Bimodal Characteristic)', fontsize=13, pad=15)
    plt.xticks(x, classes, rotation=20)
    plt.ylabel('Probability', fontsize=12)
    plt.ylim(0, 0.6)
    plt.annotate('Predicted (Wrong)', xy=(4, 0.45), xytext=(4.5, 0.52),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    plt.annotate('Ground Truth', xy=(0, 0.38), xytext=(0.5, 0.48),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig5_Softmax_Bimodal.png', dpi=300)
    plt.close()

# ================= 图6 =================
def draw_fig6_confusion():
    plt.figure(figsize=(9, 7))
    cm_percent = cm_data.astype('float') / cm_data.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Greys',
                xticklabels=classes, yticklabels=classes, 
                annot_kws={"size": 10, "color": "black"})
    plt.title('Aggregated Confusion Matrix on CK+', fontsize=14, pad=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('fig6_CK_Confusion.png', dpi=300)
    plt.close()

# ================= 执行 =================
draw_fig4_accuracy()
draw_fig5_softmax()
draw_fig6_confusion()
print("✅ 所有黑白图片已生成！")