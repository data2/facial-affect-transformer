import pandas as pd
import matplotlib.pyplot as plt

# 读取你刚刚跑出来的日志
df = pd.read_csv('./exp_baseline/train_log.csv')

plt.figure(figsize=(12, 5))

# 绘制 Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='#1f77b4', lw=2)
plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#ff7f0e', lw=2)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 绘制 Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['train_acc'], label='Train Acc', color='#2ca02c', lw=2)
plt.plot(df['epoch'], df['val_acc'], label='Val Acc', color='#d62728', lw=2)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('./exp_baseline/training_curves.png', dpi=300)
print("✅ 论文曲线图已生成：./exp_baseline/training_curves.png")
