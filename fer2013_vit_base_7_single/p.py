#!/usr/bin/env python3
# font_test.py - 独立字体测试脚本
import matplotlib
import matplotlib.pyplot as plt
import os

# 列出所有可用字体
from matplotlib import font_manager
print("🔍 系统可用字体:")
fonts = [f.name for f in font_manager.fontManager.ttflist]
chinese_fonts = [f for f in fonts if any(key in f.lower() for key in ['noto', 'uming', 'ukai', 'hei', 'song'])]
for i, font in enumerate(chinese_fonts[:20]):
    print(f"  {i+1:2d}. {font}")

# 测试几种字体
test_texts = ['中文测试', '预测标签', '混淆矩阵']

for font_name in ['Noto Sans CJK SC', 'AR PL UMing CN', 'DejaVu Sans']:
    try:
        matplotlib.rcParams['font.family'] = font_name
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, text in enumerate(test_texts):
            ax.text(0.5, 0.7 - i*0.2, text, fontsize=16, ha='center', va='center')
        ax.set_title(f'字体: {font_name}', fontsize=14)
        ax.axis('off')
        
        filename = f'font_test_{font_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ {font_name}: 保存到 {filename}")
    except Exception as e:
        print(f"❌ {font_name}: 失败 - {e}")

print("\n✅ 字体测试完成，请查看生成的PNG文件")
