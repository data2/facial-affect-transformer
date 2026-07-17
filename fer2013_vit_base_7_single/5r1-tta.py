import torch
import torch.nn.functional as F
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def run_tta_r1():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- 确认 R1 的路径 ---
    MODEL_PATH = './exp5_refined_final/best.pth' 
    DATA_PATH = './data/test'
    
    print(f"正在加载 5-R1 模型架构...")
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=7)
    
    print(f"正在加载 5-R1 权重: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval()

    tf_v = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_loader = DataLoader(datasets.ImageFolder(DATA_PATH, tf_v), batch_size=64, shuffle=False, num_workers=8)

    print("\n" + "="*50)
    print("🚀 5-R1 TTA 冲刺评估中...")
    print("="*50)
    
    correct_raw = 0
    correct_tta = 0
    total = 0
    
    with torch.no_grad():
        for imgs, lbls in tqdm(val_loader):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            total += lbls.size(0)
            
            # 原始推理
            out_orig = model(imgs)
            prob_orig = F.softmax(out_orig, dim=1)
            
            # TTA 翻转推理
            imgs_flip = torch.flip(imgs, dims=[3])
            out_flip = model(imgs_flip)
            prob_flip = F.softmax(out_flip, dim=1)
            
            # 融合
            final_prob = (prob_orig + prob_flip) / 2
            
            correct_raw += (prob_orig.argmax(1) == lbls).sum().item()
            correct_tta += (final_prob.argmax(1) == lbls).sum().item()

    acc_raw = 100. * correct_raw / total
    acc_tta = 100. * correct_tta / total
    
    print("\n" + "*"*50)
    print(f"📊 5-R1 原始 Acc: {acc_raw:.2f}%")
    print(f"🔥 5-R1 TTA Acc: {acc_tta:.2f}%")
    print(f"📈 TTA 净增益: {acc_tta - acc_raw:+.3f}%")
    print("*"*50)

    if acc_tta >= 74.0:
        print("🎊 漂亮！5-R1 靠 TTA 成功破了 74%！")
    else:
        print("🚨 还差一点点？立刻运行刚才给你的 Ensemble (R1+R2) 脚本，必破！")

if __name__ == "__main__":
    run_tta_r1()
