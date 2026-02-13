import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 数据准备 ---
datasets = {
    'FER2013': {
        'labels': ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
        'counts': [4953, 547, 5121, 8989, 6077, 4002, 6198]
    },
    'CK+': {
        'labels': ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'],
        'counts': [45, 18, 59, 25, 69, 28, 83]
    }
}

# --- 2. 学术样式配置 ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.unicode_minus': False
})
sns.set_style("ticks")

# --- 3. 创建并列子图 (1行2列) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.subplots_adjust(wspace=0.25) # 调整子图间距

for i, (name, data) in enumerate(datasets.items()):
    ax = axes[i]
    labels = data['labels']
    counts = data['counts']
    
    # 使用统一的低饱和度配色方案
    colors = sns.color_palette("muted", n_colors=len(labels), desat=0.6)
    
    # 绘图
    bars = ax.bar(labels, counts, color=colors, edgecolor='0.2', linewidth=0.8)
    
    # 细节微调
    ax.set_title(f'({chr(97+i)}) {name} Class Distribution', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Emotion Category')
    
    # 针对 CK+ 的标签进行倾斜，防止重叠（可选）
    if name == 'CK+':
        ax.set_xticklabels(labels, rotation=15)
    
    # 添加数值标注
    offset = max(counts) * 0.02 # 根据数据量自动调整标注高度
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{int(height)}', ha='center', va='bottom', fontsize=8.5)
    
    sns.despine(ax=ax) # 移除多余边框

# --- 4. 保存 ---
plt.savefig('datasets_comparison_combined.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('datasets_comparison_combined.svg', format='svg', bbox_inches='tight')

print("✅ 组合矢量图已生成：datasets_comparison_combined.pdf/svg")
plt.show()
