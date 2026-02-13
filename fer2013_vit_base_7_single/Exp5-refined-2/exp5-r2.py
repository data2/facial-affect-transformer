import os, copy, torch, timm, torch.nn as nn, numpy as np, pandas as pd
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

# --- 1. EMA 模块 (完全保持之前的逻辑) ---
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters(): p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            msd, esd = model.state_dict(), self.ema.state_dict()
            for k in esd: esd[k].copy_(esd[k] * self.decay + msd[k] * (1. - self.decay))

# --- 2. 自动探测型 LLRD 逻辑 (5-R2 核心修改: Layer Decay 0.85) ---
def get_vit_lr_groups(model, base_lr, weight_decay, layer_decay=0.85):
    num_layers = 12 + 1 
    param_groups = {}
    
    print("\n" + "="*50)
    print("LR Grouping Analysis (5-R2 Strategy):")
    
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        this_wd = 0. if (param.ndim <= 1 or name.endswith(".bias")) else weight_decay
        
        is_new_module = any(kw in name.lower() for kw in ["head", "spatial", "synergistic"])
        
        if is_new_module:
            this_lr = base_lr * 1.5 
            layer_id = num_layers
            print(f"🔥 [1.5x Boost] {name}")
        elif name.startswith("patch_embed") or name.startswith("pos_embed") or name == "cls_token":
            layer_id = 0
            this_lr = base_lr * (layer_decay ** num_layers)
        elif name.startswith("blocks"):
            layer_id = int(name.split('.')[1]) + 1
            this_lr = base_lr * (layer_decay ** (num_layers - layer_id))
        else:
            layer_id = num_layers
            this_lr = base_lr

        group_key = f"layer_{layer_id}_wd_{this_wd}"
        if group_key not in param_groups:
            param_groups[group_key] = {"params": [], "lr": this_lr, "weight_decay": this_wd}
        param_groups[group_key]["params"].append(param)
        
    print("="*50 + "\n")
    return list(param_groups.values())

def main():
    # --- 基础配置 ---
    SAVE_DIR = './exp5_refined_v2'
    os.makedirs(SAVE_DIR, exist_ok=True)
    LOG_FILE = f"{SAVE_DIR}/train_log.csv"
    
    BASE_LR = 2e-5      
    LAYER_DECAY = 0.85  # 5-R2 修改：保护主干，压榨 Head
    WEIGHT_DECAY = 0.05
    EPOCHS = 80
    WARMUP_EPOCHS = 10  # 5-R2 修改：预热翻倍，稳定开局
    
    # --- 数据准备 ---
    tf_t = transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.TrivialAugmentWide(), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tf_v = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    train_set = datasets.ImageFolder('./data/train', tf_t)
    cw = compute_class_weight('balanced', classes=np.unique(train_set.targets), y=train_set.targets)
    train_loader = DataLoader(train_set, batch_size=64, sampler=WeightedRandomSampler([cw[t] for t in train_set.targets], len(train_set)), num_workers=8, pin_memory=True)
    val_loader = DataLoader(datasets.ImageFolder('./data/test', tf_v), batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # --- 模型初始化 (请确保你的 Spatial Head 结构已定义在此) ---
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=7).cuda()
    
    if os.path.exists('./weights/vit_base_patch16_224.pth'):
        sd = torch.load('./weights/vit_base_patch16_224.pth', map_location='cpu')
        sd = sd['model'] if 'model' in sd else sd
        sd = {k: v for k, v in sd.items() if "head" not in k}
        model.load_state_dict(sd, strict=False)

    ema = ModelEMA(model)

    # --- 优化器与双调度器 (完全还原 5-R 逻辑) ---
    params = get_vit_lr_groups(model, base_lr=BASE_LR, weight_decay=WEIGHT_DECAY, layer_decay=LAYER_DECAY)
    optimizer = torch.optim.AdamW(params)
    
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-WARMUP_EPOCHS, eta_min=1e-7)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    
    # 5-R2 修改：Label Smoothing 降至 0.05 提升决断力
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler('cuda')
    best_acc = 0.0

    # --- 训练主循环 ---
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_corrects = 0.0, 0
        # 还原进度条名称风格
        pbar = tqdm(train_loader, desc=f"Exp5-Refined2 E{epoch+1}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            optimizer.zero_grad(set_to_none=True) 
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            ema.update(model)
            train_loss += loss.item() * imgs.size(0)
            train_corrects += torch.sum(out.argmax(1) == lbls.data)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- 验证 (EMA) ---
        ema.ema.eval(); preds, targets, val_loss = [], [], 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.cuda(), lbls.cuda()
                with torch.amp.autocast('cuda'):
                    out = ema.ema(imgs)
                    loss = criterion(out, lbls)
                val_loss += loss.item() * imgs.size(0)
                preds.extend(out.argmax(1).cpu().numpy())
                targets.extend(lbls.cpu().numpy())

        # --- 日志计算 (还原 CSV 列名与计算逻辑) ---
        t_loss, t_acc = train_loss/len(train_loader.dataset), train_corrects.double().item()/len(train_loader.dataset)
        v_loss, v_acc = val_loss/len(val_loader.dataset), accuracy_score(targets, preds)
        pre, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='macro')
        
        pd.DataFrame([{'epoch':epoch+1,'train_loss':t_loss,'train_acc':t_acc,'val_loss':v_loss,'val_acc':v_acc,'precision':pre,'recall':rec,'f1':f1}]).to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))
        
        # 调度器逻辑
        current_lr = optimizer.param_groups[-1]['lr']
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(ema.ema.state_dict(), f"{SAVE_DIR}/best.pth")
            with open(f"{SAVE_DIR}/best_report.txt", 'w') as f:
                f.write(classification_report(targets, preds, digits=4))
        
        # 还原日志打印格式：E{epoch} | EMA-Val Acc: {acc} | F1: {f1} | LR_max: {lr}
        print(f"E{epoch+1} | EMA-Val Acc: {v_acc:.4f} | F1: {f1:.4f} | LR_max: {current_lr:.2e}")

if __name__ == '__main__': main()