import os, copy, torch, timm, torch.nn as nn, numpy as np, pandas as pd
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

# --- 1. 定义 Hybrid Stem (卷积干道) ---
class ConvStem(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # 使用多层卷积替代线性投影，增加局部归纳偏置
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

# --- 2. EMA 策略 ---
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters(): p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            msd, esd = model.state_dict(), self.ema.state_dict()
            for k in esd: esd[k].copy_(esd[k] * self.decay + msd[k] * (1. - self.decay))

# --- 3. 修复后的 LLRD (分层学习率) 分组逻辑 ---
def get_layerwise_params(model, lr, weight_decay, layer_decay=0.75):
    parameter_groups = []
    assigned_params = set() # 核心修复：防止参数重复分配
    num_layers = 12 

    # 组 A: 新初始化的层 + 关键层 (最高学习率)
    group_top = []
    for n, p in model.named_parameters():
        if any(x in n for x in ["patch_embed", "pos_embed", "head", "norm", "cls_token"]):
            group_top.append(p)
            assigned_params.add(id(p))
    parameter_groups.append({"params": group_top, "lr": lr, "weight_decay": weight_decay})

    # 组 B: Transformer Blocks (从深到浅，LR 递减)
    # Block 11 靠近 Head，LR 较高；Block 0 靠近输入，LR 最低
    for i in range(num_layers):
        l_lr = lr * (layer_decay ** (num_layers - i))
        block_params = []
        for p in model.blocks[i].parameters():
            if id(p) not in assigned_params:
                block_params.append(p)
                assigned_params.add(id(p))
        if block_params:
            parameter_groups.append({"params": block_params, "lr": l_lr, "weight_decay": weight_decay})

    return parameter_groups

def main():
    # 路径与日志设置
    SAVE_DIR = './exp4_hybrid_final'
    os.makedirs(SAVE_DIR, exist_ok=True)
    LOG_FILE = f"{SAVE_DIR}/train_log.csv"
    
    # 数据增强保持与 Exp2 一致
    tf_t = transforms.Compose([transforms.RandomResizedCrop(224), transforms.TrivialAugmentWide(), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tf_v = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    train_set = datasets.ImageFolder('./data/train', tf_t)
    cw = compute_class_weight('balanced', classes=np.unique(train_set.targets), y=train_set.targets)
    train_loader = DataLoader(train_set, batch_size=64, sampler=WeightedRandomSampler([cw[t] for t in train_set.targets], len(train_set)), num_workers=8, pin_memory=True)
    val_loader = DataLoader(datasets.ImageFolder('./data/test', tf_v), batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # 模型加载
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=7).cuda()
    model.patch_embed = ConvStem(embed_dim=768).cuda() # 注入卷积干道
    
    if os.path.exists('./weights/vit_base_patch16_224.pth'):
        sd = torch.load('./weights/vit_base_patch16_224.pth', map_location='cpu'); sd = sd['model'] if 'model' in sd else sd
        # 排除结构不匹配的键
        sd = {k: v for k, v in sd.items() if "patch_embed" not in k and "head" not in k}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded. Missing: {len(missing)} keys (expected for ConvStem/Head).")

    ema = ModelEMA(model)
    
    # 优化器与调度器 (5 Epoch Warmup)
    base_lr = 6e-5 # 略微调高以带动新初始化的卷积层
    params = get_layerwise_params(model, lr=base_lr, weight_decay=0.05)
    optimizer = torch.optim.AdamW(params)
    
    num_epochs = 80
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * 5
    
    # 使用 Step-based 调度器确保 Warmup 平滑
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    best_acc = 0.0

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_corrects = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Exp4 Hybrid E{epoch+1}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(imgs); loss = criterion(out, lbls)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step() # 每个 step 更新学习率
            ema.update(model)
            
            train_loss += loss.item() * imgs.size(0)
            train_corrects += torch.sum(out.argmax(1) == lbls.data)

        # 验证循环
        ema.ema.eval(); preds, targets, val_loss = [], [], 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.cuda(), lbls.cuda()
                with torch.amp.autocast('cuda'):
                    out = ema.ema(imgs); loss = criterion(out, lbls)
                val_loss += loss.item() * imgs.size(0); preds.extend(out.argmax(1).cpu().numpy()); targets.extend(lbls.cpu().numpy())

        # 指标计算
        t_acc = train_corrects.double().item()/len(train_loader.dataset)
        v_acc = accuracy_score(targets, preds)
        pre, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
        
        # 日志保存
        pd.DataFrame([{'epoch':epoch+1,'train_acc':t_acc,'val_acc':v_acc,'f1':f1}]).to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(ema.ema.state_dict(), f"{SAVE_DIR}/best.pth")
            with open(f"{SAVE_DIR}/best_report.txt", 'w') as f: f.write(classification_report(targets, preds, digits=4))
        
        print(f"E{epoch+1} | Val Acc: {v_acc:.4f} | Best: {best_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

if __name__ == '__main__': main()
