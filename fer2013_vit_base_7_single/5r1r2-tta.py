import torch
import torch.nn.functional as F
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def run_final_ensemble():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- 确保路径正确 ---
    PATH_R1 = './exp5_refined_final/best.pth'  # 刚才跑出 73.98% 的那个
    PATH_R2 = './exp5_refined_v2/best.pth'     # Loss 0.35 的那个
    DATA_PATH = './data/test'
    
    print("正在加载双 ViT 架构...")
    model1 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=7).to(DEVICE)
    model2 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=7).to(DEVICE)
    
    def load_weights(model, path):
        sd = torch.load(path, map_location=DEVICE)
        new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(new_sd)
        model.eval()

    load_weights(model1, PATH_R1)
    load_weights(model2, PATH_R2)

    tf_v = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_loader = DataLoader(datasets.ImageFolder(DATA_PATH, tf_v), batch_size=64, shuffle=False)

    print("\n" + "="*50)
    print("🚀 正在执行终极融合: [R1 + R2] 集成推理")
    print("="*50)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, lbls in tqdm(val_loader):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            
            # 1. R1 原始 + 翻转 (TTA)
            out1 = model1(imgs)
            out1_flip = model1(torch.flip(imgs, dims=[3]))
            prob1 = (F.softmax(out1, dim=1) + F.softmax(out1_flip, dim=1)) / 2
            
            # 2. R2 原始 + 翻转 (TTA)
            out2 = model2(imgs)
            out2_flip = model2(torch.flip(imgs, dims=[3]))
            prob2 = (F.softmax(out2, dim=1) + F.softmax(out2_flip, dim=1)) / 2
            
            # 3. 终极平均 (Ensemble)
            final_prob = (prob1 + prob2) / 2
            
            correct += (final_prob.argmax(1) == lbls).sum().item()
            total += lbls.size(0)

    final_acc = 100. * correct / total
    print("\n" + "*"*50)
    print(f"🏆 最终集成 Acc (R1 + R2 + TTA): {final_acc:.4f}%")
    print("*"*50)

if __name__ == "__main__":
    run_final_ensemble()
