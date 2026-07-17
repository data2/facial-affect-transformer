# 设置环境变量，使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 手动下载权重
python -c "
import timm
import torch
# 尝试下载权重
model = timm.create_model('vit_huge_patch14_224', pretrained=True, num_classes=7)
print('✅ 权重下载成功')
"
