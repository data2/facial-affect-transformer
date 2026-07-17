import os, copy, torch, timm, torch.nn as nn, numpy as np, pandas as pd
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# --- 1. EMA 模块 (对标 Exp5) ---
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters(): p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            msd, esd = model.state_dict(), self.ema.state_dict()
            for k in esd: esd[k].copy_(esd[k] * self.decay + msd[k] * (1. - self.decay))

# --- 2. LLRD 逻辑 (对标 Exp5, 包含 1.5x 补贴) ---
def get_vit_lr_groups(model, base_lr, weight_decay, layer_decay=0.90):
    num_layers = 12 + 1 
    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        this_wd = 0. if (param.ndim <= 1 or name.endswith(".bias")) else weight_decay
        is_new_module = any(kw in name.lower() for kw in ["head", "spatial", "synergistic"])
        
        if is_new_module:
            this_lr = base_lr * 1.5 
            layer_id = num_layers
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
    return list(param_groups.values())

# --- 3. 单个 Fold 训练函数 ---
def train_one_fold(fold, train_idx, val_idx, full_dataset, fer_weights):
    SAVE_DIR = f'./exp7/fold_{fold+1}'
    os.makedirs(SAVE_DIR, exist_ok=True)
    LOG_FILE = f"{SAVE_DIR}/train_log.csv" # 对标之前的 train_log.csv
    
    BASE_LR = 1e-5 # 微调学习率
    EPOCHS = 50
    WARMUP_EPOCHS = 5
    
    # 数据划分与变换
    train_sub = Subset(copy.deepcopy(full_dataset), train_idx)
    train_sub.dataset.transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_sub = Subset(copy.deepcopy(full_dataset), val_idx)
    val_sub.dataset.transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    fold_targets = [full_dataset.targets[i] for i in train_idx]
    cw = compute_class_weight('balanced', classes=np.unique(fold_targets), y=fold_targets)
    sampler = WeightedRandomSampler([cw[t] for t in fold_targets], len(fold_targets))
    
    train_loader = DataLoader(train_sub, batch_size=64, sampler=sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # 模型加载 (请确保此处的结构与你 FER 实验完全一致)
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=7).cuda()
    if os.path.exists(fer_weights):
        model.load_state_dict(torch.load(fer_weights, map_location='cuda'), strict=True)
        print(f"✅ Fold {fold+1}: FER 最佳权重加载成功")

    ema = ModelEMA(model)
    optimizer = torch.optim.AdamW(get_vit_lr_groups(model, BASE_LR, 0.05, 0.90))
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-WARMUP_EPOCHS, eta_min=1e-7)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler('cuda')
    
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_corrects = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Fold {fold+1} E{epoch+1}")
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

        # 验证 (EMA)
        ema.ema.eval(); preds, targets, val_loss = [], [], 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.cuda(), lbls.cuda()
                out = ema.ema(imgs)
                loss = criterion(out, lbls)
                val_loss += loss.item() * imgs.size(0)
                preds.extend(out.argmax(1).cpu().numpy())
                targets.extend(lbls.cpu().numpy())

        # 日志计算 (完全对标之前的 CSV 字段)
        t_loss = train_loss/len(train_loader.dataset)
        t_acc = train_corrects.double().item()/len(train_loader.dataset)
        v_loss = val_loss/len(val_loader.dataset)
        v_acc = accuracy_score(targets, preds)
        pre, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='macro')
        
        # 写入 CSV 日志
        log_data = {'epoch':epoch+1,'train_loss':t_loss,'train_acc':t_acc,'val_loss':v_loss,'val_acc':v_acc,'precision':pre,'recall':rec,'f1':f1}
        pd.DataFrame([log_data]).to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))

        # 更新学习率
        current_lr = optimizer.param_groups[-1]['lr']
        if epoch < WARMUP_EPOCHS: warmup_scheduler.step()
        else: main_scheduler.step()
            
        # 保存最佳报告 (对标 best_report.txt)
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(ema.ema.state_dict(), f"{SAVE_DIR}/best.pth")
            with open(f"{SAVE_DIR}/best_report.txt", 'w') as f:
                f.write(classification_report(targets, preds, digits=4))
            
        print(f"Fold {fold+1} E{epoch+1} | EMA-Val Acc: {v_acc:.4f} | F1: {f1:.4f} | LR: {current_lr:.2e}")

    return best_acc

def main():
    DATA_ROOT = './ck' # 包含 7 个文件夹
    FER_WEIGHTS = './exp5_refined_v3/run_8/best.pth' # 读取之前的最佳模型
    
    full_dataset = datasets.ImageFolder(DATA_ROOT)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    fold_accuracies = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_dataset)))):
        acc = train_one_fold(fold, train_idx, val_idx, full_dataset, FER_WEIGHTS)
        fold_accuracies.append(acc)
        
    # 汇总最终 CV 结果
    summary = pd.DataFrame({'fold': range(1,11), 'best_acc': fold_accuracies})
    summary.to_csv('./exp7/cv_final_summary.csv', index=False)
    
    print("\n" + "="*40)
    print(f"🏆 10-Fold CV Mean Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"📊 10-Fold CV Std: {np.std(fold_accuracies):.4f}")
    print("="*40)

if __name__ == '__main__':
    main()
