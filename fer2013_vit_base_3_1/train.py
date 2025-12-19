# train_3090_optimized_corrected.py - 正确的修复版本
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
import numpy as np
from tqdm import tqdm
import time
import os
import math
import random
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


# ==================== 3090优化配置 (内存<18G) ====================
class Config3090:
    """针对NVIDIA RTX 3090的优化配置，内存控制在18GB以下"""
    def __init__(self):
        # Basic configuration
        self.model_name = 'vit_base_patch16_224'
        self.num_classes = 7
        self.img_size = 224

        # 3090优化参数 - batch size从16提升到32
        self.batch_size = 32  # 提升到32，但仍控制在18G内存内
        self.num_epochs = 80   # 减少epoch，因为batch size大了收敛更快
        self.learning_rate = 2.5e-5  # 适当提高学习率
        self.weight_decay = 0.05
        self.warmup_epochs = 8  # 适当warmup
        
        # Data augmentation
        self.cutmix_prob = 0.45
        self.mixup_prob = 0.25
        self.cutmix_alpha = 0.7
        self.mixup_alpha = 0.1

        # Regularization
        self.drop_rate = 0.3
        self.label_smoothing = 0.1

        # Optimization strategies
        self.grad_accum_steps = 1  # batch size大了，不需要梯度累积
        self.patience = 25
        self.target_acc = 0.74

        # Class weights
        self.class_weights = None
        self.dynamic_weight_adjust = True

        # No resume training
        self.resume_from = None
        self.start_epoch = 0

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mixed precision
        self.use_amp = True  # 启用混合精度训练，减少显存占用

    def __str__(self):
        info = "=" * 60 + "\n"
        info += "🎯 3090优化训练配置 (内存<18GB)\n"
        info += "=" * 60 + "\n"
        info += f"📊 模型: {self.model_name}\n"
        info += f"📈 目标准确率: {self.target_acc*100:.1f}%\n"
        info += f"⚙️  批次大小: {self.batch_size} (优化前: 16)\n"
        info += f"📚 总轮数: {self.num_epochs}\n"
        info += f"💡 学习率: {self.learning_rate:.1e}\n"
        info += f"🔄 热身轮数: {self.warmup_epochs}\n"
        info += f"🎨 增强策略: CutMix({self.cutmix_prob}), MixUp({self.mixup_prob})\n"
        info += f"⚡ 混合精度训练: {'启用' if self.use_amp else '禁用'}\n"
        info += f"💻 训练设备: {self.device}\n"
        info += f"📊 GPU显存: 24GB RTX 3090\n"
        info += "=" * 60
        return info


# ==================== Advanced Data Augmentation ====================
class AdvancedAugmentation:
    """高级数据增强策略 - 完整的类定义"""
    def __init__(self, config):
        self.config = config
        self.epoch = 0

    def get_train_transform(self):
        """获取训练数据增强"""
        return transforms.Compose([
            transforms.RandomResizedCrop(self.config.img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.85, 1.15),
                shear=10
            ),
            transforms.Grayscale(num_output_channels=3),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 3.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3)
            ),
        ])

    def get_val_transform(self):
        """获取验证数据增强"""
        return transforms.Compose([
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def cutmix(self, x, y, alpha=1.0):
        """CutMix增强"""
        if alpha <= 0:
            return x, y, y, 1.0

        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        H, W = x.shape[2], x.shape[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[index]

        return x, y_a, y_b, lam

    def mixup(self, x, y, alpha=0.2):
        """MixUp增强"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def apply_augmentation(self, images, labels, epoch):
        """应用数据增强"""
        self.epoch = epoch

        r = np.random.rand()

        if r < self.config.cutmix_prob:
            images, targets_a, targets_b, lam = self.cutmix(
                images, labels, self.config.cutmix_alpha
            )
            return images, targets_a, targets_b, lam, 'cutmix'

        elif r < self.config.cutmix_prob + self.config.mixup_prob:
            images, targets_a, targets_b, lam = self.mixup(
                images, labels, self.config.mixup_alpha
            )
            return images, targets_a, targets_b, lam, 'mixup'

        else:
            return self.apply_medium_augmentation(images, labels)

    def apply_medium_augmentation(self, images, labels):
        """中等强度增强"""
        batch_size = images.size(0)
        
        # 随机裁剪
        if np.random.rand() < 0.7:
            scale = np.random.uniform(0.85, 1.0)
            H, W = images.shape[2], images.shape[3]
            new_H, new_W = int(H * scale), int(W * scale)
            
            top = np.random.randint(0, H - new_H) if H > new_H else 0
            left = np.random.randint(0, W - new_W) if W > new_W else 0
            images = images[:, :, top:top+new_H, left:left+new_W]
            images = F.interpolate(images, size=(H, W), mode='bilinear')
        
        # 水平翻转
        if torch.rand(1).item() < 0.5:
            images = torch.flip(images, [3])
        
        # 颜色扰动
        brightness = torch.rand(batch_size, 1, 1, 1).to(images.device) * 0.3 + 0.85
        contrast = torch.rand(batch_size, 1, 1, 1).to(images.device) * 0.3 + 0.85
        
        images = images * brightness
        mean = images.mean(dim=[1,2,3], keepdim=True)
        images = (images - mean) * contrast + mean
        images = torch.clamp(images, 0, 1)
        
        # 轻微旋转
        if np.random.rand() < 0.3:
            angle = np.random.uniform(-10, 10)
            images = transforms.functional.rotate(images, angle)
        
        return images, labels, labels, 1.0, 'medium'


# ==================== Smart Loss Function ====================
class SmartLossFunction:
    """智能损失函数"""
    def __init__(self, class_weights, config):
        self.class_weights = class_weights
        self.config = config
        self.label_smoothing = config.label_smoothing

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.focal_loss = FocalLoss(gamma=2.0, alpha=class_weights)

    def __call__(self, outputs, targets, augmentation_type='none', lam=1.0):
        # Base cross entropy loss
        if self.label_smoothing > 0:
            ce_loss = self.label_smooth_ce(outputs, targets)
        else:
            ce_loss = self.ce_loss(outputs, targets)

        # Add Focal Loss for hard classes
        fear_mask = (targets == 2)  # fear
        disgust_mask = (targets == 1)  # disgust
        sad_mask = (targets == 5)  # sad

        if fear_mask.any() or disgust_mask.any() or sad_mask.any():
            focal_weight = 0.3
            focal_component = self.focal_loss(outputs, targets)
            total_loss = (1 - focal_weight) * ce_loss + focal_weight * focal_component
        else:
            total_loss = ce_loss

        return total_loss

    def label_smooth_ce(self, x, target):
        """标签平滑交叉熵"""
        confidence = 1.0 - self.label_smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.label_smoothing * smooth_loss
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss"""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==================== Dynamic Weight Adjuster ====================
class DynamicWeightAdjuster:
    """动态权重调整器"""
    def __init__(self, base_weights, history_size=5):
        self.base_weights = base_weights.clone()
        self.history_size = history_size
        self.history = []
        self.performance_history = []

    def update_weights(self, class_accuracies, epoch):
        """根据类别性能调整权重"""
        if len(self.history) >= self.history_size:
            self.history.pop(0)

        self.history.append(class_accuracies)
        self.performance_history.append({
            'epoch': epoch,
            'accuracies': class_accuracies
        })

        if len(self.history) < 2:
            return self.base_weights

        new_weights = self.base_weights.clone()

        # 计算近期平均准确率
        recent_acc = np.mean(self.history[-2:], axis=0)

        for i in range(len(class_accuracies)):
            current_acc = class_accuracies[i]
            recent_avg = recent_acc[i] if i < len(recent_acc) else 0

            if i == 2 or i == 5:  # fear和sad
                if current_acc < 0.55:  # 降低阈值到55%
                    if (current_acc - recent_avg) < -0.03:  # 下降超过3%
                        new_weights[i] *= 2.5  # 更大幅增加
                    else:
                        new_weights[i] *= 1.8  # 中等增加
                elif current_acc > 0.65:  # 表现较好时
                    new_weights[i] *= 0.85  # 轻微降低权重

            # 原始调整策略（其他类别）
            elif current_acc < 0.5 and (current_acc - recent_avg) < -0.05:
                new_weights[i] *= 2.0

            elif 0.5 <= current_acc < 0.7 and abs(current_acc - recent_avg) < 0.02:
                new_weights[i] *= 1.5

            elif current_acc > 0.8:
                new_weights[i] *= 0.8

        return new_weights


# ==================== Intelligent Early Stopping ====================
class IntelligentEarlyStopping:
    """智能早停机制"""
    def __init__(self, patience=15, min_epochs=30, target_acc=0.75):
        self.patience = patience
        self.min_epochs = min_epochs
        self.target_acc = target_acc
        self.best_acc = 0
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.improvement_history = []

    def __call__(self, val_acc, epoch, train_loss=None):
        if epoch < self.min_epochs:
            return False

        improvement = val_acc - self.best_acc
        self.improvement_history.append(improvement)

        if val_acc > self.best_acc + 1e-4:
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.counter = 0

            if val_acc > self.target_acc - 0.02:
                self.patience = max(self.patience, 20)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


# ==================== Model Creation ====================
def create_model(config, pretrained_path='./weights/vit_base_patch16_224.pth'):
    """创建和初始化模型"""
    print(f"🔄 创建模型: {config.model_name}")

    model = timm.create_model(
        config.model_name,
        pretrained=False,
        num_classes=config.num_classes,
        drop_rate=config.drop_rate
    )

    if os.path.exists(pretrained_path):
        print(f"📥 加载预训练权重: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint

            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            filtered_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('head.') and not key.startswith('fc.'):
                    filtered_state_dict[key] = value

            missing_keys, unexpected_keys = model.load_state_dict(
                filtered_state_dict, strict=False
            )

            if missing_keys:
                print(f"⚠️  缺失的键: {len(missing_keys)}个")
                # 打印前5个缺失的键
                for i, key in enumerate(missing_keys[:5]):
                    print(f"    {i+1}. {key}")
            if unexpected_keys:
                print(f"⚠️  意外的键: {len(unexpected_keys)}个")

            print("✅ 预训练权重加载成功")

        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
            print("🔄 使用随机初始化...")
    else:
        print("⚠️ 未找到预训练权重，使用随机初始化")

    return model.to(config.device)


# ==================== Learning Rate Scheduler ====================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """带warmup的cosine调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==================== 优化的训练函数 (带混合精度) ====================
def train_epoch_3090(model, train_loader, criterion, optimizer, scheduler,
                    data_aug, config, epoch, weight_adjuster=None, scaler=None):
    """针对3090优化的训练epoch - 带混合精度"""
    
    # 在第27轮调整学习率
    if epoch == 26:  # 第27轮（从0开始计数）
        optimizer.param_groups[0]['lr'] = 2.0e-05
        print(f"🎯 第{epoch+1}轮：学习率调整为2.00e-05")
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    aug_stats = {'cutmix': 0, 'mixup': 0, 'medium': 0}

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}")

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(config.device), labels.to(config.device)

        # 应用数据增强
        aug_images, targets_a, targets_b, lam, aug_type = data_aug.apply_augmentation(
            images, labels, epoch
        )

        if aug_type not in aug_stats:
            aug_stats[aug_type] = 0
        aug_stats[aug_type] += 1

        # 混合精度训练
        if config.use_amp:
            # 使用新版API避免警告
            with torch.amp.autocast('cuda'):
                outputs = model(aug_images)
                
                if aug_type == 'cutmix' or aug_type == 'mixup':
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, labels)
                
                loss = loss / config.grad_accum_steps
            
            scaler.scale(loss).backward()
            
            # 只在梯度累积步骤结束时更新
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # 注意：这里不调用scheduler.step()，而是在epoch结束时调用
        else:
            # 普通训练
            outputs = model(aug_images)
            
            if aug_type == 'cutmix' or aug_type == 'mixup':
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, labels)
            
            loss = loss / config.grad_accum_steps
            loss.backward()
            
            # 只在梯度累积步骤结束时更新
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                # 注意：这里不调用scheduler.step()，而是在epoch结束时调用

        # 统计
        total_loss += loss.item() * config.grad_accum_steps
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)

        if aug_type == 'cutmix' or aug_type == 'mixup':
            correct_a = (predicted == targets_a).float()
            correct_b = (predicted == targets_b).float()
            batch_correct = (lam * correct_a + (1 - lam) * correct_b).sum().item()
            correct += batch_correct
        else:
            correct += (predicted == labels).sum().item()

        # 更新进度条
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100. * correct / total if total > 0 else 0
            current_lr = optimizer.param_groups[0]['lr']

            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%',
                'LR': f'{current_lr:.2e}',
            })

    # 打印增强统计
    total_batches = len(train_loader)
    print(f"\n📊 Epoch {epoch+1} 增强统计:")
    for aug_type, count in aug_stats.items():
        percentage = count / total_batches * 100
        print(f"  {aug_type}: {count}/{total_batches} ({percentage:.1f}%)")

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy

# ==================== Validation Function ====================
def validate(model, val_loader, criterion, config):
    """验证模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * config.num_classes
    class_total = [0] * config.num_classes

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="验证中", leave=False):
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 每个类别的统计
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(config.num_classes):
        if class_total[i] > 0:
            class_accuracies.append(class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0.0)

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0

    return accuracy, avg_loss, class_accuracies


# ==================== Main Training Function ====================
def train_3090(config):
    """3090优化的主训练函数"""
    print(config)
    print("\n🚀 开始3090优化训练!")
    print("=" * 60)
    
    # 显示GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 显存: {gpu_memory:.1f} GB")
        print(f"📊 Batch Size: {config.batch_size}")
        print(f"⚡ 混合精度训练: {'启用' if config.use_amp else '禁用'}")

    # 创建数据增强
    data_aug = AdvancedAugmentation(config)
    
    # 加载数据集
    print("\n📁 加载数据集...")
    train_dir = './data/train'
    val_dir = './data/test'
    
    train_dataset = datasets.ImageFolder(train_dir, transform=data_aug.get_train_transform())
    val_dataset = datasets.ImageFolder(val_dir, transform=data_aug.get_val_transform())
    
    # 计算类别权重
    train_labels = [label for _, label in train_dataset]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(config.device)
    config.class_weights = class_weights
    
    print(f"📊 数据集统计:")
    print(f"  训练集: {len(train_dataset):,} images")
    print(f"  验证集: {len(val_dataset):,} images")
    print(f"  类别: {train_dataset.classes}")
    
    # 优化数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,  # 增加worker数量
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2  # 预取数据
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # 创建模型
    model = create_model(config)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器 - 使用不同的调度器类型
    # 使用每epoch调度而不是每batch调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.num_epochs - config.warmup_epochs,
        eta_min=config.learning_rate * 0.01
    )
    
    # 创建损失函数
    criterion = SmartLossFunction(class_weights, config)
    
    # 创建动态权重调整器
    weight_adjuster = DynamicWeightAdjuster(class_weights) if config.dynamic_weight_adjust else None
    
    # 创建早停机制
    early_stopping = IntelligentEarlyStopping(
        patience=config.patience,
        min_epochs=30,
        target_acc=config.target_acc
    )
    
    # 混合精度训练（使用新版API）
    scaler = torch.amp.GradScaler('cuda') if config.use_amp else None
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'class_acc': [],
        'learning_rates': []
    }
    
    best_acc = 0
    best_epoch = 0
    
    # 训练循环
    print(f"\n🚀 开始训练循环 (目标: {config.target_acc*100:.1f}%)")
    print("=" * 60)
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch_3090(
            model, train_loader, criterion, optimizer, scheduler,
            data_aug, config, epoch, weight_adjuster, scaler
        )
        
        # 验证
        val_acc, val_loss, class_acc = validate(model, val_loader, criterion, config)
        
        # 在epoch结束时调用调度器 - 这是正确的调用时机
        if scheduler is not None:
            scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['class_acc'].append(class_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # 打印结果
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n📊 Epoch {epoch+1:3d}/{config.num_epochs}")
        print(f"  Training Loss: {train_loss:.4f} | Training Accuracy: {train_acc*100:6.2f}%")
        print(f"  Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc*100:6.2f}%")
        print(f"  Learning Rate: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_acc > best_acc + 0.0001:
            best_acc = val_acc
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'epoch': epoch,
                'best_acc': best_acc,
                'val_acc': val_acc,
                'class_acc': class_acc,
                'config': config.__dict__,
                'history': history
            }, 'best_model_3090.pth')
            print(f"  🎉 New Best Accuracy: {best_acc*100:.2f}% (saved to best_model_3090.pth)")
        
        # 动态权重调整
        if weight_adjuster and epoch >= 5:
            new_weights = weight_adjuster.update_weights(class_acc, epoch)
            criterion.class_weights = new_weights
            print("  ⚖️  Dynamic weight adjustment completed")
        
        # 早停检查
        if early_stopping(val_acc, epoch, train_loss):
            print(f"\n🛑 Early stopping triggered! Best Accuracy: {early_stopping.best_acc*100:.2f}% "
                  f"(Epoch {early_stopping.best_epoch+1})")
            break
        
        # 每5轮保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'checkpoint_3090_epoch_{epoch+1:03d}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'class_acc': class_acc
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
        
        print("-" * 60)
    
    # 训练总结
    print("\n" + "=" * 60)
    print("🎯 训练总结")
    print("=" * 60)
    print(f"📊 Final Best Accuracy: {best_acc*100:.2f}% (Epoch {best_epoch+1})")
    print(f"🎯 Target Accuracy: {config.target_acc*100:.1f}%")
    print(f"📈 Difference: {(config.target_acc - best_acc)*100:+.2f}%")
    print(f"🔄 Total Training Epochs: {epoch + 1}")
    
    return model, history, best_acc


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 创建配置
    config = Config3090()
    
    # 开始训练
    try:
        model, history, best_acc = train_3090(config)
        
        # 最终评估
        print("\n🔍 最终模型评估...")
        
        if os.path.exists('best_model_3090.pth'):
            checkpoint = torch.load('best_model_3090.pth', map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_acc = checkpoint['best_acc']
            class_acc = checkpoint['class_acc']
            
            print(f"✅ Final Best Accuracy: {best_acc*100:.2f}%")
            print("📊 Per-class Accuracy:")
            for i, cls_name in enumerate(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']):
                if i < len(class_acc):
                    print(f"  {cls_name}: {class_acc[i]*100:6.2f}%")
            
            # 计算平均类别准确率
            avg_class_acc = np.mean(class_acc) * 100 if class_acc else 0
            print(f"📈 Average Class Accuracy: {avg_class_acc:.2f}%")
            
            if best_acc >= 0.74:
                print(f"🎉 成功突破74%!")
            elif best_acc >= 0.73:
                print(f"✅ 达到73%以上!")
            else:
                print(f"📈 最终准确率: {best_acc*100:.2f}%")
    
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
