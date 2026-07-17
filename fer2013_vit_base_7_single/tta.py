import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def run_tta_final():
    # --- 1. 环境配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = './exp5_refined_v2/best.pth'
    DATA_PATH = './data/test'
    
    # --- 2. 实例化模型架构 (完全对齐 Exp5-R2) ---
    print(f"正在创建 ViT Base 架构...")
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=7)
    
    # --- 3. 加载权重 (处理 state_dict) ---
    print(f"正在从 {MODEL_PATH} 加载最佳权重...")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # 自动处理分布式或 EMA 保存时可能出现的模块前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') # 移除可能的前缀
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval()

    # --- 4. 准备验证集 (完全对齐 tf_v 配置) ---
    tf_v = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    val_set = datasets.ImageFolder(DATA_PATH, tf_v)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # --- 5. TTA 核心逻辑 ---
    print("\n" + "="*50)
    print("🚀 开始 TTA 绝杀推理 (Original + Horizontal Flip)")
    print("="*50)
    
    correct_raw = 0
    correct_tta = 0
    total = 0
    
    with torch.no_grad():
        for imgs, lbls in tqdm(val_loader, desc="TTA 推理中"):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            total += lbls.size(0)
            
            # 原始预测
            out_orig = model(imgs)
            prob_orig = F.softmax(out_orig, dim=1)
            
            # 水平翻转预测 (TTA)
            imgs_flip = torch.flip(imgs, dims=[3])
            out_flip = model(imgs_flip)
            prob_flip = F.softmax(out_flip, dim=1)
            
            # 概率平均融合
            final_prob = (prob_orig + prob_flip) / 2
            
            # 统计
            pred_raw = prob_orig.argmax(1)
            pred_tta = final_prob.argmax(1)
            
            correct_raw += (pred_raw == lbls).sum().item()
            correct_tta += (pred_tta == lbls).sum().item()

    # --- 6. 结果展示 ---
    acc_raw = 100. * correct_raw / total
    acc_tta = 100. * correct_tta / total
    
    print("\n" + "*"*50)
    print(f"📊 原始单图验证集 Acc: {acc_raw:.2f}%")
    print(f"🔥 TTA 融合后最终 Acc: {acc_tta:.2f}%")
    print(f"📈 绝对增益提升: {acc_tta - acc_raw:+.3f}%")
    print("*"*50)
    
    if acc_tta >= 74.0:
        print("🎉 恭喜！5-R2 成功在 TTA 阶段绝杀 74%！")
    else:
        print("🤔 还没到 74%？别急，我们下一步把 5-R1 和 5-R2 的概率合并。")

if __name__ == "__main__":
    run_tta_final()
