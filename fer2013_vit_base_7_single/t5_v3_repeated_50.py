import os, copy, torch, timm, torch.nn as nn, numpy as np, pandas as pd
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
import gc

# --- 1. EMA 模块 ---
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters(): p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            msd, esd = model.state_dict(), self.ema.state_dict()
            for k in esd: esd[k].copy_(esd[k] * self.decay + msd[k] * (1. - self.decay))

# --- 2. 自动探测型 LLRD 逻辑 (1.5x Boost) ---
def get_vit_lr_groups(model, base_lr, weight_decay, layer_decay=0.90):
    num_layers = 12 + 1 
    param_groups = {}
    
    print("\n" + "="*50)
    print("LR Grouping Analysis (Automated Detection):")
    
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

def train_one_run(run_id, base_save_dir):
    # --- 基础配置 ---
    SAVE_DIR = os.path.join(base_save_dir, f'run_{run_id}')
    os.makedirs(SAVE_DIR, exist_ok=True)
    LOG_FILE = f"{SAVE_DIR}/train_log.csv"
    
    BASE_LR = 2e-5      
    LAYER_DECAY = 0.90  
    WEIGHT_DECAY = 0.05
    EPOCHS = 80
    WARMUP_EPOCHS = 5   
    
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

    # --- 模型初始化 ---
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=7).cuda()
    if os.path.exists('./weights/vit_base_patch16_224.pth'):
        sd = torch.load('./weights/vit_base_patch16_224.pth', map_location='cpu')
        sd = sd['model'] if 'model' in sd else sd
        sd = {k: v for k, v in sd.items() if "head" not in k}
        model.load_state_dict(sd, strict=False)

    ema = ModelEMA(model)
    params = get_vit_lr_groups(model, base_lr=BASE_LR, weight_decay=WEIGHT_DECAY, layer_decay=LAYER_DECAY)
    optimizer = torch.optim.AdamW(params)
    
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-WARMUP_EPOCHS, eta_min=1e-7)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    best_acc = 0.0

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_corrects = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Run {run_id} | E{epoch+1}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = model(imgs); loss = criterion(out, lbls)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
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
                    out = ema.ema(imgs); loss = criterion(out, lbls)
                val_loss += loss.item() * imgs.size(0)
                preds.extend(out.argmax(1).cpu().numpy()); targets.extend(lbls.cpu().numpy())

        v_acc = accuracy_score(targets, preds)
        pre, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='macro')
        t_loss, t_acc = train_loss/len(train_loader.dataset), train_corrects.double().item()/len(train_loader.dataset)
        
        # 记录日志 (对标 CSV 格式)
        pd.DataFrame([{'epoch':epoch+1,'train_loss':t_loss,'train_acc':t_acc,'val_loss':val_loss/len(val_loader.dataset),'val_acc':v_acc,'precision':pre,'recall':rec,'f1':f1}]).to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))
        
        current_lr = optimizer.param_groups[-1]['lr']
        if epoch < WARMUP_EPOCHS: warmup_scheduler.step()
        else: main_scheduler.step()
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(ema.ema.state_dict(), f"{SAVE_DIR}/best.pth")
            with open(f"{SAVE_DIR}/best_report.txt", 'w') as f:
                f.write(classification_report(targets, preds, digits=4))
        
        print(f"E{epoch+1} | Val Acc: {v_acc:.4f} | F1: {f1:.4f} | LR: {current_lr:.2e}")

    # 显存深度清理
    del model, ema, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    return best_acc

if __name__ == '__main__':
    TOTAL_RUNS = 50 # 设定为 10 轮
    BASE_DIR = './exp5_refined_v3_repeated_50'
    results = []
    
    print(f"🔥 启动 {TOTAL_RUNS} 轮自动化训练任务...")
    for i in range(1, TOTAL_RUNS + 1):
        print(f"\n" + "🚀"*15 + f"\n# 第 {i} 轮实验正式开始\n" + "🚀"*15)
        best_acc = train_one_run(i, BASE_DIR)
        results.append({'run': i, 'best_acc': best_acc})
        
        # 实时同步总览表
        pd.DataFrame(results).to_csv(f"{BASE_DIR}/all_runs_summary.csv", index=False)
    
    print("\n✅ 所有任务已圆满完成！")
