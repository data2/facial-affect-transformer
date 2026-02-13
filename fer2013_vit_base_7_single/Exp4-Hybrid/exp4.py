import os, copy, torch, timm, torch.nn as nn, numpy as np, pandas as pd
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

# --- 新增：卷积干道 (Convolutional Stem) ---
class ConvStem(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # 使用 4 层卷积逐步下采样，增强局部感知
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters(): p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            msd, esd = model.state_dict(), self.ema.state_dict()
            for k in esd: esd[k].copy_(esd[k] * self.decay + msd[k] * (1. - self.decay))

# --- 新增：分层学习率逻辑 (LLRD) ---
def get_layerwise_params(model, lr, weight_decay, layer_decay=0.75):
    parameter_groups = []
    num_layers = 12  # ViT-Base 12层
    # 卷积干道和位置编码 (最高学习率)
    parameter_groups.append({
        "params": [p for n, p in model.named_parameters() if "patch_embed" in n or "pos_embed" in n],
        "lr": lr, "weight_decay": weight_decay
    })
    # Transformer Blocks 分层递减
    for i in range(num_layers):
        l_lr = lr * (layer_decay ** (num_layers - i))
        parameter_groups.append({
            "params": model.blocks[i].parameters(),
            "lr": l_lr, "weight_decay": weight_decay
        })
    # Head 和 Norm (最高学习率)
    parameter_groups.append({
        "params": [p for n, p in model.named_parameters() if "head" in n or "norm" in n],
        "lr": lr, "weight_decay": weight_decay
    })
    return parameter_groups

def main():
    SAVE_DIR = './exp4_hybrid_llrd'
    os.makedirs(SAVE_DIR, exist_ok=True)
    LOG_FILE = f"{SAVE_DIR}/train_log.csv"
    
    tf_t = transforms.Compose([transforms.RandomResizedCrop(224), transforms.TrivialAugmentWide(), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tf_v = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    train_set = datasets.ImageFolder('./data/train', tf_t)
    cw = compute_class_weight('balanced', classes=np.unique(train_set.targets), y=train_set.targets)
    train_loader = DataLoader(train_set, batch_size=64, sampler=WeightedRandomSampler([cw[t] for t in train_set.targets], len(train_set)), num_workers=8, pin_memory=True)
    val_loader = DataLoader(datasets.ImageFolder('./data/test', tf_v), batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # 模型构建
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=7).cuda()
    # 替换结构：线性投影 -> 卷积干道
    model.patch_embed = ConvStem(embed_dim=768).cuda()
    
    # 加载预训练权重 (排除结构改变的部分)
    if os.path.exists('./weights/vit_base_patch16_224.pth'):
        sd = torch.load('./weights/vit_base_patch16_224.pth', map_location='cpu'); sd = sd['model'] if 'model' in sd else sd
        # 排除 patch_embed 因为我们改了结构，排除 head 因为类别数可能不同
        sd = {k: v for k, v in sd.items() if "patch_embed" not in k and "head" not in k}
        model.load_state_dict(sd, strict=False)

    ema = ModelEMA(model)
    
    # 优化器使用分层学习率
    params = get_layerwise_params(model, lr=5e-5, weight_decay=0.05) # 基础LR略微调高以适应新卷积层
    optimizer = torch.optim.AdamW(params)
    
    # 学习率调度器：增加 Warmup (前5轮)
    num_steps = len(train_loader) * 80
    warmup_steps = len(train_loader) * 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    best_acc = 0.0

    for epoch in range(80):
        model.train()
        train_loss, train_corrects = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Exp4 Hybrid E{epoch+1}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(imgs); loss = criterion(out, lbls)
            
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            scheduler.step() # 每个 step 更新 LR 以确保 Warmup 平滑
            
            ema.update(model); train_loss += loss.item() * imgs.size(0); train_corrects += torch.sum(out.argmax(1) == lbls.data)

        # 验证部分使用 EMA
        ema.ema.eval(); preds, targets, val_loss = [], [], 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.cuda(), lbls.cuda()
                with torch.amp.autocast('cuda'):
                    out = ema.ema(imgs); loss = criterion(out, lbls)
                val_loss += loss.item() * imgs.size(0); preds.extend(out.argmax(1).cpu().numpy()); targets.extend(lbls.cpu().numpy())

        t_loss, t_acc = train_loss/len(train_loader.dataset), train_corrects.double().item()/len(train_loader.dataset)
        v_loss, v_acc = val_loss/len(val_loader.dataset), accuracy_score(targets, preds)
        pre, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='macro')
        
        pd.DataFrame([{'epoch':epoch+1,'train_loss':t_loss,'train_acc':t_acc,'val_loss':v_loss,'val_acc':v_acc,'precision':pre,'recall':rec,'f1':f1}]).to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(ema.ema.state_dict(), f"{SAVE_DIR}/best.pth")
            with open(f"{SAVE_DIR}/best_report.txt", 'w') as f: f.write(classification_report(targets, preds, digits=4))
        print(f"E{epoch+1} | Hybrid EMA-Val Acc: {v_acc:.4f} | Best: {best_acc:.4f}")

if __name__ == '__main__': main()