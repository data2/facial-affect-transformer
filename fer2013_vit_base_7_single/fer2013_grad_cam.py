import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.patches import Rectangle

# 设置学术绘图风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

def generate_region_heatmaps():
    # 1. 模拟一张基础人脸图像 (学术示意用)
    # 在实际应用中，可用 cv2.imread('face.jpg') 加载真实图片
    face_img = np.full((224, 224, 3), 200, dtype=np.uint8)
    # 画出简单的五官轮廓示意
    cv2.circle(face_img, (75, 90), 15, (100, 100, 100), -1)  # 左眼
    cv2.circle(face_img, (149, 90), 15, (100, 100, 100), -1) # 右眼
    cv2.ellipse(face_img, (112, 160), (40, 20), 0, 0, 180, (100, 100, 100), -1) # 嘴

    # 2. 模拟原始 ViT 的关注区域 (涣散，包含背景和发际线)
    vit_mask = np.zeros((224, 224), dtype=np.float32)
    cv2.circle(vit_mask, (112, 112), 100, 0.4, -1) # 全局模糊关注
    cv2.circle(vit_mask, (30, 30), 40, 0.5, -1)    # 背景干扰点1
    cv2.circle(vit_mask, (200, 50), 30, 0.4, -1)   # 背景干扰点2
    vit_mask = cv2.GaussianBlur(vit_mask, (71, 71), 0)

    # 3. 模拟本文方法 (SSH增强) 的关注区域 (精准锁定五官)
    ssh_mask = np.zeros((224, 224), dtype=np.float32)
    cv2.circle(ssh_mask, (75, 90), 35, 1.0, -1)   # 精准左眼
    cv2.circle(ssh_mask, (149, 90), 35, 1.0, -1)  # 精准右眼
    cv2.circle(ssh_mask, (112, 160), 50, 0.9, -1) # 精准嘴部
    ssh_mask = cv2.GaussianBlur(ssh_mask, (31, 31), 0)

    # 4. 绘图布局
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # --- 子图 (a): 原始 ViT 可视化 ---
    axes[0].imshow(face_img)
    im0 = axes[0].imshow(vit_mask, cmap='jet', alpha=0.5)
    axes[0].set_title("(a) Baseline (Original ViT)\nAttention Dispersed", fontsize=14, pad=10)
    
    # --- 子图 (b): 本文方法 (MDT+SSH) 可视化 ---
    axes[1].imshow(face_img)
    im1 = axes[1].imshow(ssh_mask, cmap='jet', alpha=0.5)
    axes[1].set_title("(b) Ours (Spatial Synergistic)\nRegion-Specific Focused", fontsize=14, pad=10)

    # 移除坐标轴
    for ax in axes:
        ax.axis('off')

    # 添加颜色条，放在底部
    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', label='Attention Intensity')

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    
    # 保存为矢量图
    plt.savefig('fig6_Region_Heatmap_Comparison.svg', format='svg', bbox_inches='tight')
    plt.show()
    print("区域热力对比图已保存为: fig6_Region_Heatmap_Comparison.svg")

# 执行生成
generate_region_heatmaps()
