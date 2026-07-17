import os

import random

import matplotlib.pyplot as plt

from PIL import Image



# 1. 路径配置 (请确保路径指向你的实际目录)

base_path = './data/train'  # 通常从训练集中提取样本展示

emotions = sorted(os.listdir(base_path)) # 读取文件夹名作为类别名：Angry, Disgust等



# 2. 论文级样式设置

plt.rcParams.update({

    'font.family': 'serif',

    'font.serif': ['Times New Roman'],

    'pdf.fonttype': 42,

    'axes.unicode_minus': False

})



n_samples = 5 # 每行显示多少张图

n_classes = len(emotions)



# 3. 创建画布

fig, axes = plt.subplots(n_classes, n_samples, figsize=(10, 12))

plt.subplots_adjust(wspace=0.05, hspace=0.05) # 减小图片间距，让排版更紧凑



for i, emotion in enumerate(emotions):

    emotion_path = os.path.join(base_path, emotion)

    # 获取该目录下所有图片文件名

    img_names = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    

    # 随机抽取 5 张

    selected_imgs = random.sample(img_names, n_samples)

    

    for j, img_name in enumerate(selected_imgs):

        img_path = os.path.join(emotion_path, img_name)

        img = Image.open(img_path).convert('L') # 转换为灰度图

        

        ax = axes[i, j]

        ax.imshow(img, cmap='gray')

        

        # 移除坐标轴

        ax.set_xticks([])

        ax.set_yticks([])

        

        # 在每一行的最左侧添加类别名称

        if j == 0:

            ax.set_ylabel(emotion, rotation=0, labelpad=40, 

                          verticalalignment='center', fontsize=12, fontweight='bold')



# 4. 保存高质量矢量图

# plt.suptitle('Representative Samples from FER2013 Dataset', fontsize=16, y=0.98)

plt.tight_layout(rect=[0.1, 0, 1, 1]) # 为左侧标签预留空间



plt.savefig('fer2013_folder_samples.pdf', format='pdf', bbox_inches='tight', dpi=300)

plt.savefig('fer2013_folder_samples.svg', format='svg', bbox_inches='tight')



print(f"✅ 样本预览图已生成！")

print(f"检测到类别: {emotions}")

print(f"文件保存为: fer2013_folder_samples.pdf")

plt.show()
