import os, copy, torch, timm, torch.nn as nn, numpy as np, pandas as pd
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

# ==========================================
# 创新点模块：空间协同注意力头 (Spatial Synergistic Head)
# ==========================================
class SpatialSynergisticHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        # 空间重要性分支：对 196 个 Patch Token 进行显著性评分
        self.spatial_gate = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x shape: [Batch, 197, 768] (ViT-Base 输出的全部序列)
        cls_token = x[:, 0]        # 全局语义特征 [B, 768]
        patch_tokens = x[:, 1:]    # 局部空间特征 [B, 196, 768]

        # 1. 计算每个 Patch 的注意力权重
        weights = self.spatial_gate(patch_tokens) # [B, 196, 1]
        weights = torch.softmax(weights, dim=1)   # 空间维度归一化

        # 2. 聚合局部显著性特征
        local_context = torch.sum(patch_tokens * weights, dim=1) # [B, 768]

        # 3. 协同融合：全局 + 局部
        final_feat = cls_token + local_context
        return self.classifier(self.norm(final_feat))

# ==========================================
# 辅助模块 (EMA & LLRD 保留 Exp3 逻辑)
# ==========================================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters(): p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            msd, esd = model.state_dict(), self.ema.state_dict()
            for k in esd: esd[k].copy_(esd[k] * self.decay + msd[k] * (1. - self.decay))

def get_vit_lr_groups(model, base_lr, weight_decay, layer_decay=0.85):
    num_layers = len(model.blocks) + 1
    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        this_wd = 0. if (param.ndim <= 1 or name.endswith(".bias")) else weight_decay
        if name.startswith("patch_embed") or name.startswith("pos_embed") or name == "cls_token":
            layer_id = 0
        elif name.startswith("blocks"):
            layer_id = int(name.split('.')[1]) + 1
        else: # 自动将新定义的 head 识别为最后一层，给予最高学习率
            layer_id = num_layers
        lr_scale = layer_decay ** (num_layers - layer_id)
        this_lr = base_lr * lr_scale
        group_key = f"layer_{layer_id}_wd_{this_wd}"
        if group_key not in param_groups:
            param_groups[group_key] = {"params": [], "lr": this_lr, "weight_decay": this_wd}
        param_groups[group_key]["params"].append(param)
    return list(param_groups.values())

def main():
    # --- 配置区 ---
    SAVE_DIR = './exp5_vit_gli_head' 
    os.makedirs(SAVE_DIR, exist_ok=True)
    LOG_FILE = f"{SAVE_DIR}/train_log.csv"
    
    BASE_LR = 2e-5
    LAYER_DECAY = 0.85
    WEIGHT_DECAY = 0.05
    EPOCHS = 80

    # --- 数据准备 (保留 Exp2/3 强力增强方案) ---
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

    # --- 模型构建 ---
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    # 注入创新 Head
    model.head = SpatialSynergisticHead(model.embed_dim, num_classes=7)
    
    # 加载 Backbone 预训练权重
    if os.path.exists('./weights/vit_base_patch16_224.pth'):
        sd = torch.load('./weights/vit_base_patch16_224.pth', map_location='cpu')
        sd = sd['model'] if 'model' in sd else sd
        sd = {k: v for k, v in sd.items() if "head" not in k}
        model.load_state_dict(sd, strict=False)

    model = model.cuda()
    ema = ModelEMA(model)

    # --- 优化器与策略 ---
    params = get_vit_lr_groups(model, base_lr=BASE_LR, weight_decay=WEIGHT_DECAY, layer_decay=LAYER_DECAY)
    optimizer = torch.optim.AdamW(params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    best_acc = 0.0

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_corrects = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Exp5 E{epoch+1}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                # forward_features 得到 [B, 197, 768]
                feats = model.forward_features(imgs)
                out = model.head(feats)
                loss = criterion(out, lbls)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            
            train_loss += loss.item() * imgs.size(0)
            train_corrects += torch.sum(out.argmax(1) == lbls.data)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- 验证环节 (EMA) ---
        ema.ema.eval(); preds, targets, val_loss = [], [], 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.cuda(), lbls.cuda()
                with torch.amp.autocast('cuda'):
                    f = ema.ema.forward_features(imgs)
                    out = ema.ema.head(f)
                    loss = criterion(out, lbls)
                val_loss += loss.item() * imgs.size(0)
                preds.extend(out.argmax(1).cpu().numpy())
                targets.extend(lbls.cpu().numpy())

        # 指标计算 (完整保留 Exp3 逻辑)
        t_loss = train_loss / len(train_loader.dataset)
        t_acc = train_corrects.double().item() / len(train_loader.dataset)
        v_loss = val_loss / len(val_loader.dataset)
        v_acc = accuracy_score(targets, preds)
        pre, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='macro')
        
        # 记录日志 (精准匹配你的需求)
        pd.DataFrame([{
            'epoch': epoch + 1,
            'train_loss': t_loss,
            'train_acc': t_acc,
            'val_loss': v_loss,
            'val_acc': v_acc,
            'precision': pre,
            'recall': rec,
            'f1': f1
        }]).to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))
        
        scheduler.step()
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(ema.ema.state_dict(), f"{SAVE_DIR}/best.pth")
            with open(f"{SAVE_DIR}/best_report.txt", 'w') as f:
                f.write(classification_report(targets, preds, digits=4))
        
        print(f"E{epoch+1} | EMA-Val Acc: {v_acc:.4f} | F1: {f1:.4f} | LR_max: {optimizer.param_groups[-1]['lr']:.2e}")

if __name__ == '__main__':
    main()