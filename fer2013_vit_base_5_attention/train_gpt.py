from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import os
import copy

# ==================== 配置 ====================
class Config3090:
    def __init__(self):
        self.model_name = 'vit_base_patch16_224'
        self.num_classes = 7
        self.img_size = 224
        self.batch_size = 32
        self.num_epochs = 80
        self.learning_rate = 2.5e-5
        self.weight_decay = 0.05
        self.cutmix_prob = 0.45
        self.mixup_prob = 0.25
        self.cutmix_alpha = 0.7
        self.mixup_alpha = 0.1
        self.drop_rate = 0.3
        self.label_smoothing = 0.1
        self.grad_accum_steps = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = True
        self.amp_dtype = torch.float16
        self.grad_clip = 1.0
        self.min_lr = 1e-6
        self.local_pretrained_path = './weights/vit_base_patch16_224.pth'
        self.ema_decay = 0.999  # EMA权重衰减

# ==================== 数据增强 ====================
class EnhancedAugmentation:
    def __init__(self, config):
        self.config = config
    def get_train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(self.config.img_size, scale=(0.7,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.Grayscale(num_output_channels=3),
            transforms.ColorJitter(0.4,0.4,0.4,0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    def get_val_transform(self):
        return transforms.Compose([
            transforms.Resize((self.config.img_size,self.config.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

# ==================== Attention ====================
class EnhancedSEAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_classes=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=False),
            nn.Sigmoid()
        )
        self.class_aware_attention = nn.Sequential(
            nn.Linear(channel, channel//4),
            nn.ReLU(),
            nn.Linear(channel//4,num_classes),
            nn.Softmax(dim=-1)
        )
    def forward(self,x,labels=None):
        if x.dim()==3:
            avg_out = self.mlp(self.avg_pool(x.transpose(1,2)).squeeze(-1))
        else:
            avg_out = self.mlp(self.avg_pool(x.unsqueeze(-1)).squeeze(-1))
        attention = avg_out
        if x.dim()==3:
            attended_features = x*attention.unsqueeze(1)
        else:
            attended_features = x*attention
        if labels is not None and self.training:
            class_attention = self.class_aware_attention(attended_features.mean(dim=1) if attended_features.dim()==3 else attended_features)
            return attended_features,class_attention
        return attended_features

# ==================== Adaptive Loss ====================
class FocalLoss(nn.Module):
    def __init__(self,gamma=2.0,alpha=None,reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self,inputs,targets):
        ce_loss = F.cross_entropy(inputs,targets,reduction='none',weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = ((1-pt)**self.gamma)*ce_loss
        if self.reduction=='mean': return loss.mean()
        elif self.reduction=='sum': return loss.sum()
        else: return loss

class AdaptiveLossFunction:
    def __init__(self,class_weights,config):
        self.class_weights = class_weights
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.focal_loss = FocalLoss(alpha=class_weights)
        self.label_smoothing = config.label_smoothing
        self.epoch=0
        self.training=True
    def __call__(self,outputs,targets):
        if self.label_smoothing>0:
            confidence = 1-self.label_smoothing
            logprobs = F.log_softmax(outputs,dim=-1)
            nll_loss = -logprobs.gather(dim=-1,index=targets.unsqueeze(1)).squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            ce_loss = (confidence*nll_loss+self.label_smoothing*smooth_loss).mean()
        else:
            ce_loss = self.ce_loss(outputs,targets)
        focal_weight = 0.5 if self.epoch>35 else 0.3
        focal_component = self.focal_loss(outputs,targets)
        total_loss = (1-focal_weight)*ce_loss + focal_weight*focal_component
        return total_loss

# ==================== CutMix / MixUp ====================
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=0.7, device='cuda'):
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    index = torch.randperm(x.size(0)).to(device)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

# ==================== 创建模型 ====================
def create_enhanced_vit(config):
    print(f"加载本地预训练模型: {config.local_pretrained_path}")
    model = timm.create_model(config.model_name, pretrained=False, num_classes=config.num_classes)
    if os.path.exists(config.local_pretrained_path):
        state_dict = torch.load(config.local_pretrained_path,map_location='cpu')
        for k in list(state_dict.keys()):
            if 'head' in k:
                del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        print("本地预训练权重加载完成")
    else:
        print("警告：本地预训练模型不存在，将从零初始化")
    feature_dim = getattr(model,'num_features',768)
    attention = EnhancedSEAttention(feature_dim,num_classes=config.num_classes)
    if hasattr(model,'head'): model.head = nn.Identity()
    elif hasattr(model,'fc'): model.fc = nn.Identity()
    new_head = nn.Linear(feature_dim,config.num_classes)
    def new_forward(x,labels=None):
        features = model.forward_features(x)
        if features.dim()==3: features = features.mean(dim=1)
        if labels is not None and model.training: features,_ = attention(features,labels)
        else: features = attention(features)
        output = new_head(features)
        return output
    model.forward = new_forward
    model.attention = attention
    model.new_head = new_head
    return model.to(config.device)

# ==================== EMA ====================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if k in msd:
                    v.copy_(v * self.decay + msd[k]*(1.-self.decay))
    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

# ==================== 训练函数 ====================
def train_epoch(model, train_loader, criterion, optimizer, config, epoch, scaler=None, ema=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)
    for images, labels in loop:
        images, labels = images.to(config.device), labels.to(config.device)
        r = np.random.rand()
        if r < config.cutmix_prob:
            images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=config.cutmix_alpha, device=config.device)
        elif r < config.cutmix_prob + config.mixup_prob:
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=config.mixup_alpha)
            targets_a = targets_a.to(config.device)
            targets_b = targets_b.to(config.device)
        else:
            lam = 1.0

        optimizer.zero_grad()
        if config.use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = lam*criterion(outputs, targets_a) + (1-lam)*criterion(outputs, targets_b) if lam!=1 else criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = lam*criterion(outputs, targets_a) + (1-lam)*criterion(outputs, targets_b) if lam!=1 else criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # 更新每类统计
        for t, p in zip(labels, predicted):
            class_total[int(t)] += 1
            if t == p:
                class_correct[int(t)] += 1

        if ema:
            ema.update(model)

    train_acc = correct / total
    train_loss = total_loss / len(train_loader)

    # 打印训练总体和各类准确率
    print(f"[Train Epoch {epoch+1}] Loss={train_loss:.4f}, Acc={train_acc*100:.2f}%")
    for cls in sorted(class_total.keys()):
        acc = 100 * class_correct[cls] / class_total[cls]
        print(f"  Class {cls}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

    return train_loss, train_acc

# ==================== 验证函数 ====================
def validate(model, val_loader, criterion, config):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels, predicted):
                class_total[int(t)] += 1
                if t == p:
                    class_correct[int(t)] += 1

    val_acc = correct / total
    val_loss = total_loss / len(val_loader)

    print(f"[Validation] Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%")
    for cls in sorted(class_total.keys()):
        acc = 100 * class_correct[cls] / class_total[cls]
        print(f"  Class {cls}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

    return val_loss, val_acc

# ==================== 主训练 ====================
def train_main():
    config = Config3090()
    data_aug = EnhancedAugmentation(config)

    # 数据集和 DataLoader
    train_dataset = datasets.ImageFolder('./data/train', transform=data_aug.get_train_transform())
    val_dataset = datasets.ImageFolder('./data/test', transform=data_aug.get_val_transform())
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 计算类别权重
    labels = [l for _, l in train_dataset]
    class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(labels), y=labels),
                                 dtype=torch.float32).to(config.device)

    criterion = AdaptiveLossFunction(class_weights, config)
    model = create_enhanced_vit(config)
    ema = ModelEMA(model, decay=config.ema_decay)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=config.min_lr)
    scaler = torch.amp.GradScaler(enabled=config.use_amp)

    for epoch in range(config.num_epochs):
        print(f"\n{'='*20} Epoch {epoch+1}/{config.num_epochs} {'='*20}\n")

        criterion.epoch = epoch

        # --------- 训练 ---------
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config, epoch, scaler, ema)

        # --------- 验证 ---------
        val_loss, val_acc = validate(ema.ema_model, val_loader, criterion, config)

        # 学习率更新
        scheduler.step()
        # epoch 总结
        print(f"\n[Epoch {epoch+1} Summary]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc*100:.2f}%")
        print(f"{'='*60}\n")

if __name__=='__main__':
    train_main()
