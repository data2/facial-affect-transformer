import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# 1. 配置路径 - 确保指向你正确的绝对路径
base_path = './data/train'  
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 2. 学术风设置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'pdf.fonttype': 42,
    'axes.unicode_minus': False
})

n_samples = 5  # 每行显示的图片数量
n_classes = len(emotions)

# 3. 创建画布
fig, axes = plt.subplots(n_classes, n_samples, figsize=(10, 12))
plt.subplots_adjust(wspace=0.02, hspace=0.02) 

for i, emotion in enumerate(emotions):
    emotion_path = os.path.join(base_path, emotion)
    
    if not os.path.exists(emotion_path):
        print(f"Warning: {emotion_path} not found.")
        # 如果路径缺失，关闭该行所有子图显示
        for ax in axes[i]:
            ax.axis('off')
        continue
        
    img_names = [f for f in os.listdir(emotion_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(img_names) > 0:
        selected_imgs = random.sample(img_names, min(len(img_names), n_samples))
        for j in range(n_samples):
            ax = axes[i, j]
            if j < len(selected_imgs):
                img_path = os.path.join(emotion_path, selected_imgs[j])
                img = Image.open(img_path).convert('L')
                ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(emotion, rotation=0, labelpad=35, 
                              verticalalignment='center', fontsize=12, fontweight='bold')
    else:
        for ax in axes[i]:
            ax.axis('off')

# 4. 保存为 SVG
output_file = 'fer2013_samples.svg'
plt.savefig(output_file, format='svg', bbox_inches='tight')

print(f"✅ SVG 矢量图已生成: {os.path.abspath(output_file)}")
