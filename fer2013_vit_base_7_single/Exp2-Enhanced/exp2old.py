import os
import copy
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import timm

# ==================== 1. 全局配置 ====================
class Config3090:
    def __init__(self):
        self.model_name = 'vit_base_patch16_224'
        self.num_classes = 7
        self.img_size = 224
        self.batch_size = 64        # 3090 推荐
        self.num_epochs = 80        # 建议跑满，获取完整收敛曲线
        self.warmup_epochs = 5      # 线性预热
        self.learning_rate = 2e-5   
        self.weight_decay = 0.05    
        self.label_smoothing = 0.1  # 论文标配
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = True         # 开启混合精度训练
        self.local_pretrained_path = './weights/vit_base_patch16_224.pth'
        self.ema_decay = 0.999      # 指数移动平均，使验证集表现更稳
        self.save_dir = './exp_baseline' 

# ==================== 2. 数据增强 ====================
class EnhancedAugmentation:
    def __init__(self, config):
        self.config = config
    def get_train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(self.config.img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(), # 自动增强，论文加分点
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    def get_val_transform(self):
        return transforms.Compose([
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# ==================== 3. 核心工具 (EMA & Logger) ====================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for p in self.ema_model.parameters(): p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if k in msd: v.copy_(v * self.decay + msd[k] * (1. - self.decay))

class Logger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.history = []
        if not os.path.exists(save_dir): os.makedirs(save_dir)
    def log(self, metrics):
        self.history.append(metrics)
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.save_dir, 'train_log.csv'), index=False)

# ==================== 4. 训练与验证逻辑 ====================
def train_epoch(model, loader, criterion, optimizer, scaler, config, ema):
    model.train()
    total_loss, correct, total = 0, 0, 0
    loop = tqdm(loader, desc="🚀 Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(config.device), labels.to(config.device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=config.use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        ema.update(model)
        total_loss += loss.item()
        _, pred = outputs.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, config, class_names):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = outputs.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds, average='macro')
    # 生成各类别精细报告 (P/R/F1)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    return total_loss / len(loader), acc, f1, report, all_labels, all_preds

# ==================== 5. 主程序 ====================
def main():
    config = Config3090()
    aug = EnhancedAugmentation(config)
    logger = Logger(config.save_dir)
    
    # 自动加载类别信息
    train_set = datasets.ImageFolder('./data/train', transform=aug.get_train_transform())
    val_set = datasets.ImageFolder('./data/test', transform=aug.get_val_transform())
    class_names = train_set.classes

    # 类别权重采样 (解决不平衡)
    class_weights = compute_class_weight('balanced', classes=np.unique(train_set.targets), y=train_set.targets)
    weights = torch.tensor([class_weights[t] for t in train_set.targets])
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # 1. 创建模型
    model = timm.create_model(config.model_name, pretrained=False, num_classes=config.num_classes)

    # 2. 修复维度不匹配加载权重
    if os.path.exists(config.local_pretrained_path):
        print(f"📦 正在加载预训练权重: {config.local_pretrained_path}")
        sd = torch.load(config.local_pretrained_path, map_location='cpu')
        if 'model' in sd: sd = sd['model']
        
        # 核心：过滤掉ImageNet分类层的权重 (1000维 -> 7维)
        sd = {k: v for k, v in sd.items() if 'head' not in k}
        
        msg = model.load_state_dict(sd, strict=False)
        print(f"✅ 特征层权重加载成功 (已过滤分类层)")
    
    model.to(config.device)

    # 3. 初始化优化器、学习率计划
    ema = ModelEMA(model, config.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs-config.warmup_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=config.use_amp)

    best_acc = 0.0
    print(f"🚀 实验启动 | 类别数: {len(class_names)} | 目标轮次: {config.num_epochs}")

    for epoch in range(config.num_epochs):
        # 线性预热阶段
        if epoch < config.warmup_epochs:
            lr = 1e-6 + (config.learning_rate - 1e-6) * (epoch / config.warmup_epochs)
            for pg in optimizer.param_groups: pg['lr'] = lr
        
        # 训练与验证
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, config, ema)
        v_loss, v_acc, v_f1, report, labels, preds = validate(ema.ema_model, val_loader, criterion, config, class_names)
        
        if epoch >= config.warmup_epochs: scheduler.step()

        # 日志存档
        curr_lr = optimizer.param_groups[0]['lr']
        logger.log({
            'epoch': epoch + 1, 'train_loss': t_loss, 'train_acc': t_acc,
            'val_loss': v_loss, 'val_acc': v_acc, 'val_f1': v_f1, 'lr': curr_lr
        })

        # 控制台实时反馈
        print(f"\n📊 Epoch {epoch+1}/80:")
        print(f"   [Train] Loss: {t_loss:.4f} Acc: {t_acc:.4f} | [Val] Loss: {v_loss:.4f} Acc: {v_acc:.4f}")
        print(f"   [Performance] Macro-F1: {v_f1:.4f} | LR: {curr_lr:.2e}")
        
        # 保存最佳模型与混淆矩阵
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(ema.ema_model.state_dict(), os.path.join(config.save_dir, 'best_baseline.pth'))
            
            # 自动生成高质量混淆矩阵 (Figure 2)
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix(labels, preds), annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Baseline Confusion Matrix (Acc: {v_acc:.4f})')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig(os.path.join(config.save_dir, 'best_cm.png'), dpi=300)
            plt.close()

            # 保存分类报告 (P/R/F1 表格素材)
            with open(os.path.join(config.save_dir, 'best_classification_report.txt'), 'w') as f:
                f.write(report)
            print(f"🌟 新纪录: {best_acc:.4f} | 权重与图表已存入 {config.save_dir}")

    print(f"\n✅ 训练完成! 最佳准确率: {best_acc*100:.2f}%")

if __name__ == '__main__':
    main()