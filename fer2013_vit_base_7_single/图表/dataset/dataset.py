import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. 整理数据 (统一对齐类别)
# 共有类别：Angry, Disgust, Fear, Happy, Sad, Surprise
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']
fer_counts = [4953, 547, 5121, 8989, 6077, 4002]
ck_counts = [45, 59, 25, 69, 28, 83]

# 2. 学术风样式配置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.unicode_minus': False
})
sns.set_style("ticks")

x = np.arange(len(labels))
width = 0.35  # 每个柱子的宽度

fig, ax = plt.subplots(figsize=(10, 6))

# 3. 绘制并列柱状图
# 使用对数坐标 (log=True) 是在一张图里展示百倍差距数据的唯一科学方法
rects1 = ax.bar(x - width/2, fer_counts, width, label='FER2013', 
                color='0.3', edgecolor='black', linewidth=0.8)
rects2 = ax.bar(x + width/2, ck_counts, width, label='CK+', 
                color='0.7', edgecolor='black', linewidth=0.8)

# 4. 设置坐标轴和标题
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('FER2013 vs CK+', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yscale('log') # 启用对数坐标
ax.legend()

# 5. 添加数值标注
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)

sns.despine()

# 6. 输出矢量图
plt.savefig('combined_bar_chart.pdf', format='pdf', bbox_inches='tight')
plt.savefig('combined_bar_chart.svg', format='svg', bbox_inches='tight')

print("✅ 统一柱状图已生成：combined_bar_chart.pdf 和 .svg")
plt.show()
