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


# ==================== Hyperparameter Configuration ====================
class Config:
    """Complete training configuration"""
    def __init__(self):
        # Basic configuration
        self.model_name = 'vit_base_patch16_224'
        self.num_classes = 7
        self.img_size = 224
        
        # Training parameters
        self.batch_size = 16
        self.num_epochs = 100
        self.learning_rate = 2e-5
        self.weight_decay = 0.05
        self.warmup_epochs = 12
        
        # Data augmentation
        self.cutmix_prob = 0.4
        self.mixup_prob = 0.2
        self.cutmix_alpha = 0.7
        self.mixup_alpha = 0.1
        
        # Regularization
        self.drop_rate = 0.2
        self.label_smoothing = 0.1
        
        # Optimization strategies
        self.grad_accum_steps = 2
        self.patience = 15
        self.target_acc = 0.80
        
        # Class weights
        self.class_weights = None
        self.dynamic_weight_adjust = True
        
        # Resume training
        self.resume_from = None
        self.start_epoch = 0
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def __str__(self):
        info = "=" * 60 + "\n"
        info += "🎯 冲刺80%准确率训练配置\n"
        info += "=" * 60 + "\n"
        info += f"📊 模型: {self.model_name}\n"
        info += f"📈 目标准确率: {self.target_acc*100:.1f}%\n"
        info += f"⚙️  批次大小: {self.batch_size}\n"
        info += f"📚 总轮数: {self.num_epochs}\n"
        info += f"💡 学习率: {self.learning_rate:.1e}\n"
        info += f"🔄 热身轮数: {self.warmup_epochs}\n"
        info += f"🎨 增强策略: CutMix({self.cutmix_prob}), MixUp({self.mixup_prob})\n"
        info += f"⚖️  动态权重调整: {'启用' if self.dynamic_weight_adjust else '禁用'}\n"
        info += f"🛡️  早停耐心: {self.patience}\n"
        if self.resume_from:
            info += f"🔄 从检查点恢复: {self.resume_from} (Epoch {self.start_epoch})\n"
        info += f"💻 训练设备: {self.device}\n"
        info += "=" * 60
        return info


# ==================== Advanced Data Augmentation ====================
class AdvancedAugmentation:
    """Advanced data augmentation strategies"""
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        
    def get_train_transform(self):
        """Get training data augmentation"""
        return transforms.Compose([
            # Multi-scale augmentation
            transforms.RandomResizedCrop(self.config.img_size, scale=(0.7, 1.0)),
            
            # Spatial transformations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(
                degrees=15, 
                translate=(0.1, 0.1), 
                scale=(0.85, 1.15),
                shear=10
            ),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
            
            # Color transformations
            transforms.Grayscale(num_output_channels=3),
            transforms.ColorJitter(
                brightness=0.4, 
                contrast=0.4, 
                saturation=0.4, 
                hue=0.1
            ),
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 3.0)),
            
            # To tensor
            transforms.ToTensor(),
            
            # Normalization
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            
            # Random erasing
            transforms.RandomErasing(
                p=0.5, 
                scale=(0.02, 0.2), 
                ratio=(0.3, 3.3)
            ),
        ])
    
    def get_val_transform(self):
        """Get validation data augmentation"""
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
        """CutMix augmentation"""
        if alpha <= 0:
            return x, y, y, 1.0
            
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        
        # Generate crop region
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
        
        # Apply CutMix
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[index]
        
        return x, y_a, y_b, lam
    
    def mixup(self, x, y, alpha=0.2):
        """MixUp augmentation"""
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
        """Apply data augmentation"""
        self.epoch = epoch
        
        # Adjust augmentation intensity based on epoch
        cutmix_prob = self.config.cutmix_prob
        mixup_prob = self.config.mixup_prob
        
        if epoch < 20:  # Strong augmentation in early stage
            cutmix_prob = min(0.7, cutmix_prob + 0.2)
            mixup_prob = min(0.5, mixup_prob + 0.2)
        elif epoch > 60:  # Weak augmentation in later stage
            cutmix_prob = max(0.3, cutmix_prob - 0.2)
            mixup_prob = max(0.1, mixup_prob - 0.2)
        
        # Apply CutMix
        if np.random.rand() < cutmix_prob:
            images, targets_a, targets_b, lam = self.cutmix(
                images, labels, self.config.cutmix_alpha
            )
            return images, targets_a, targets_b, lam, 'cutmix'
        
        # Apply MixUp
        elif np.random.rand() < mixup_prob:
            images, targets_a, targets_b, lam = self.mixup(
                images, labels, self.config.mixup_alpha
            )
            return images, targets_a, targets_b, lam, 'mixup'
        
        # No augmentation
        return images, labels, labels, 1.0, 'none'


# ==================== Smart Loss Function ====================
class SmartLossFunction:
    """Smart loss function with multiple optimization strategies"""
    def __init__(self, class_weights, config):
        self.class_weights = class_weights
        self.config = config
        self.label_smoothing = config.label_smoothing
        
        # Initialize various losses
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
        """Label smoothing cross entropy"""
        confidence = 1.0 - self.label_smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.label_smoothing * smooth_loss
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
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
    """Dynamic class weight adjustment"""
    def __init__(self, base_weights, history_size=5):
        self.base_weights = base_weights.clone()
        self.history_size = history_size
        self.history = []
        self.performance_history = []
        
    def update_weights(self, class_accuracies, epoch):
        """Adjust weights based on class performance"""
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
        
        # Calculate recent average accuracy
        recent_acc = np.mean(self.history[-2:], axis=0)
        
        # Adjustment strategy
        for i in range(len(class_accuracies)):
            current_acc = class_accuracies[i]
            recent_avg = recent_acc[i] if i < len(recent_acc) else 0
            
            # If accuracy is low and decreasing
            if current_acc < 0.5 and (current_acc - recent_avg) < -0.05:
                new_weights[i] *= 2.0  # Significantly increase weight
            
            # If accuracy is medium but stagnant
            elif 0.5 <= current_acc < 0.7 and abs(current_acc - recent_avg) < 0.02:
                new_weights[i] *= 1.5  # Moderately increase weight
            
            # If accuracy is high
            elif current_acc > 0.8:
                new_weights[i] *= 0.8  # Reduce weight
        
        return new_weights
    
    def get_performance_trend(self, class_idx):
        """Get class performance trend"""
        if len(self.performance_history) < 2:
            return 0
        
        recent = [h['accuracies'][class_idx] for h in self.performance_history[-2:]]
        if len(recent) >= 2:
            return recent[-1] - recent[-2]
        return 0


# ==================== Intelligent Early Stopping ====================
class IntelligentEarlyStopping:
    """Intelligent early stopping mechanism"""
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
        
        # Record improvement history
        improvement = val_acc - self.best_acc
        self.improvement_history.append(improvement)
        
        if val_acc > self.best_acc + 1e-4:  # Even tiny improvement counts
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.counter = 0
            
            # If close to target, increase patience
            if val_acc > self.target_acc - 0.02:
                self.patience = max(self.patience, 20)
        else:
            self.counter += 1
            
            # If training loss is still decreasing, slow down early stopping
            if train_loss is not None and len(self.improvement_history) > 3:
                recent_improvements = self.improvement_history[-3:]
                if all(imp < 0.001 for imp in recent_improvements):
                    self.counter = min(self.counter, self.patience - 3)
        
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


# ==================== Model Creation ====================
def create_model(config, pretrained_path='./weights/vit_base_patch16_224.pth'):
    """Create and initialize model"""
    print(f"🔄 创建模型: {config.model_name}")
    
    # Create model
    model = timm.create_model(
        config.model_name,
        pretrained=False,
        num_classes=config.num_classes,
        drop_rate=config.drop_rate
    )
    
    # Load pretrained weights
    if os.path.exists(pretrained_path):
        print(f"📥 加载预训练权重: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint
            
            # Handle different checkpoint formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Filter classification head weights
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('head.') and not key.startswith('fc.'):
                    filtered_state_dict[key] = value
            
            # Load weights
            missing_keys, unexpected_keys = model.load_state_dict(
                filtered_state_dict, strict=False
            )
            
            if missing_keys:
                print(f"⚠️  缺失的键: {len(missing_keys)}个")
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
    """Cosine annealing scheduler with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==================== Training Function ====================
def train_epoch(model, train_loader, criterion, optimizer, scheduler, 
                data_aug, config, epoch, weight_adjuster=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(config.device), labels.to(config.device)
        
        # Apply data augmentation
        aug_images, targets_a, targets_b, lam, aug_type = data_aug.apply_augmentation(
            images, labels, epoch
        )
        
        # Forward pass
        outputs = model(aug_images)
        
        # Calculate loss
        if aug_type == 'cutmix' or aug_type == 'mixup':
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            loss = criterion(outputs, labels)
        
        # Backward pass
        loss = loss / config.grad_accum_steps
        loss.backward()
        
        if (batch_idx + 1) % config.grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Statistics
        total_loss += loss.item() * config.grad_accum_steps
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        
        if aug_type == 'none':
            correct += (predicted == labels).sum().item()
        
        # Update progress bar
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100. * correct / total if total > 0 else 0
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
    
    return total_loss / len(train_loader), correct / total if total > 0 else 0


# ==================== Validation Function ====================
def validate(model, val_loader, criterion, config):
    """Validate model"""
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
            
            # Statistics per class
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # Calculate per-class accuracies
    class_accuracies = []
    for i in range(config.num_classes):
        if class_total[i] > 0:
            class_accuracies.append(class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0.0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(val_loader)
    
    return accuracy, avg_loss, class_accuracies


# ==================== Main Training Function ====================
def train(config):
    """Main training function"""
    print(config)
    print("\n🚀 开始冲刺80%准确率训练!")
    print("=" * 60)
    
    # Create data augmentation
    data_aug = AdvancedAugmentation(config)
    
    # Load datasets
    print("\n📁 加载数据集...")
    train_dir = './data/train'
    val_dir = './data/test'
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"训练数据路径不存在: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"验证数据路径不存在: {val_dir}")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=data_aug.get_train_transform())
    val_dataset = datasets.ImageFolder(val_dir, transform=data_aug.get_val_transform())
    
    # Get training labels for class weights
    train_labels = [label for _, label in train_dataset]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(config.device)
    config.class_weights = class_weights
    
    print(f"📊 数据集统计:")
    print(f"  训练集: {len(train_dataset):,} images")
    print(f"  验证集: {len(val_dataset):,} images")
    print(f"  类别: {train_dataset.classes}")
    
    print("📈 类别权重:")
    for i, cls_name in enumerate(val_dataset.classes):
        print(f"  {cls_name}: {class_weights[i].cpu().numpy():.3f}")

    if class_weights[1] > 5.0:  # disgust是索引1
        print(f"\n⚖️  手动调整disgust权重:")
        print(f"  原始: {class_weights[1].cpu().numpy():.3f} → 目标: 3.0")
        class_weights[1] = 3.0
        print(f"  调整后disgust权重: {class_weights[1].cpu().numpy():.3f}")

    config.class_weights = class_weights
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = create_model(config)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # ==================== 检查点恢复逻辑 ====================
    start_epoch = 0
    best_acc = 0
    best_epoch = 0
    
    # 检查是否有检查点文件
    if config.resume_from and os.path.exists(config.resume_from):
        print(f"\n🔄 从检查点恢复训练: {config.resume_from}")
        try:
            checkpoint = torch.load(config.resume_from, map_location=config.device)
            
            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ 模型权重加载成功")
            
            # 加载优化器状态
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ 优化器状态加载成功")
            
            # 设置起始轮数
            if config.start_epoch > 0:
                start_epoch = config.start_epoch
            elif 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            
            # 加载最佳准确率
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
                best_epoch = checkpoint.get('epoch', 0)
                print(f"📊 检查点最佳准确率: {best_acc*100:.2f}% (Epoch {best_epoch+1})")
            
            print(f"🔄 将从第 {start_epoch} 轮继续训练")
            
        except Exception as e:
            print(f"⚠️ 检查点加载失败: {e}")
            print("🔄 从头开始训练...")
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * config.num_epochs // config.grad_accum_steps
    num_warmup_steps = len(train_loader) * config.warmup_epochs // config.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )
    
    # 加载调度器状态
    if config.resume_from and os.path.exists(config.resume_from):
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✅ 学习率调度器状态加载成功")
            except:
                pass
    
    # Create loss function
    criterion = SmartLossFunction(class_weights, config)
    
    # Create dynamic weight adjuster
    weight_adjuster = DynamicWeightAdjuster(class_weights) if config.dynamic_weight_adjust else None
    
    # Create early stopping
    early_stopping = IntelligentEarlyStopping(
        patience=config.patience,
        min_epochs=30,
        target_acc=config.target_acc
    )
    
    # 如果从检查点恢复，更新early stopping
    if best_acc > 0:
        early_stopping.best_acc = best_acc
        early_stopping.best_epoch = best_epoch
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'class_acc': [],
        'learning_rates': []
    }
    
    # Training loop
    print(f"\n🚀 开始训练循环 (目标: {config.target_acc*100:.1f}%)")
    print("=" * 60)
    
    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            data_aug, config, epoch, weight_adjuster
        )
        
        # Validation
        val_acc, val_loss, class_acc = validate(model, val_loader, criterion, config)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['class_acc'].append(class_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print results
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n📊 Epoch {epoch+1:3d}/{config.num_epochs}")
        print(f"  Training Loss: {train_loss:.4f} | Training Accuracy: {train_acc*100:6.2f}%")
        print(f"  Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc*100:6.2f}%")
        print(f"  Learning Rate: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        print("  📈 Per-class Accuracy:")
        for i, cls_name in enumerate(val_dataset.classes):
            arrow = ""
            if epoch > 0 and len(history['class_acc']) >= 2 and i < len(history['class_acc'][-2]):
                diff = class_acc[i] - history['class_acc'][-2][i]
                arrow = "↑" if diff > 0.01 else "↓" if diff < -0.01 else "→"
            print(f"    {cls_name}: {class_acc[i]*100:6.2f}% {arrow}")
        
        # Save best model
        if val_acc > best_acc + 0.001:
            best_acc = val_acc
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'val_acc': val_acc,
                'class_acc': class_acc,
                'config': config.__dict__,
                'history': history
            }, 'best_model_80_target.pth')
            print(f"  🎉 New Best Accuracy: {best_acc*100:.2f}% (saved to best_model_80_target.pth)")
        
        # Dynamic weight adjustment
        if weight_adjuster and epoch >= 5:
            new_weights = weight_adjuster.update_weights(class_acc, epoch)
            criterion.class_weights = new_weights
            print("  ⚖️  Dynamic weight adjustment completed")
        
        # Early stopping check
        if early_stopping(val_acc, epoch, train_loss):
            print(f"\n🛑 Early stopping triggered! Best Accuracy: {early_stopping.best_acc*100:.2f}% "
                  f"(Epoch {early_stopping.best_epoch+1})")
            break
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1:03d}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'class_acc': class_acc
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
        
        print("-" * 60)
    
    # Training summary
    print("\n" + "=" * 60)
    print("🎯 训练总结")
    print("=" * 60)
    print(f"📊 Final Best Accuracy: {best_acc*100:.2f}% (Epoch {best_epoch+1})")
    print(f"🎯 Target Accuracy: {config.target_acc*100:.1f}%")
    print(f"📈 Difference: {(config.target_acc - best_acc)*100:+.2f}%")
    print(f"🔄 Total Training Epochs: {epoch + 1}")
    if start_epoch > 0:
        print(f"🔄 Trained Epochs: {epoch + 1 - start_epoch} (from epoch {start_epoch})")
    
    # Plot training curves
    plot_training_curves(history, val_dataset.classes, best_acc)
    
    return model, history, best_acc


# ==================== Visualization Function (ENGLISH) ====================
def plot_training_curves(history, class_names, best_acc):
    """Plot training curves (All in English)"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Loss Curves
        axes[0, 0].plot(history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Accuracy Curve
        axes[0, 1].plot(history['val_acc'], 'g-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].axhline(y=best_acc, color='r', linestyle='--', 
                          label=f'Best: {best_acc*100:.2f}%')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Validation Accuracy Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0.5, 1.0)
        
        # 3. Per-class Accuracy Evolution
        if history['class_acc']:
            class_acc_array = np.array(history['class_acc'])
            for i in range(len(class_names)):
                axes[0, 2].plot(class_acc_array[:, i], label=class_names[i], alpha=0.7)
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].set_title('Per-class Accuracy Evolution')
            axes[0, 2].legend(loc='upper left', fontsize=8)
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Learning Rate Schedule
        axes[1, 0].plot(history['learning_rates'], 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Training vs Validation Accuracy
        axes[1, 1].plot(history['train_acc'], 'b-', label='Training Accuracy', alpha=0.7)
        axes[1, 1].plot(history['val_acc'], 'r-', label='Validation Accuracy', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Training vs Validation Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Final Per-class Accuracy Bar Chart
        if history['class_acc']:
            final_class_acc = history['class_acc'][-1]
            x = np.arange(len(class_names))
            colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
            bars = axes[1, 2].bar(x, final_class_acc, color=colors, alpha=0.8)
            axes[1, 2].set_xlabel('Emotion Class')
            axes[1, 2].set_ylabel('Accuracy')
            axes[1, 2].set_title('Final Per-class Accuracy')
            axes[1, 2].set_xticks(x)
            axes[1, 2].set_xticklabels(class_names, rotation=45)
            axes[1, 2].grid(True, alpha=0.3, axis='y')
            
            # Add values on bars
            for bar, acc in zip(bars, final_class_acc):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('training_curves_80_target.png', dpi=300, bbox_inches='tight')
        plt.savefig('training_curves_80_target.pdf', bbox_inches='tight')
        print("✅ 训练曲线已保存: training_curves_80_target.png/pdf")
        
    except Exception as e:
        print(f"⚠️ 无法绘制训练曲线: {e}")


# ==================== Main Function ====================
def main():
    """Main function"""
    import argparse
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Vision Transformer for FER')
    parser.add_argument('--resume', type=str, default='', 
                       help='从检查点恢复训练，例如: --resume best_model_80_target.pth')
    parser.add_argument('--epoch', type=int, default=0,
                       help='从指定轮数恢复训练，例如: --epoch 10')
    parser.add_argument('--auto', action='store_true',
                       help='自动检测最新检查点并继续训练')
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create configuration
    config = Config()
    
    # 自动检测检查点
    if args.auto:
        import glob
        # 查找最新检查点
        checkpoints = glob.glob('checkpoint_epoch_*.pth')
        if checkpoints:
            # 按轮数排序
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_checkpoint = checkpoints[-1]
            config.resume_from = latest_checkpoint
            # 从文件名提取轮数
            try:
                epoch_str = latest_checkpoint.split('_')[-1].split('.')[0]
                config.start_epoch = int(epoch_str) + 1
            except:
                config.start_epoch = 0
            print(f"🔄 自动检测到最新检查点: {latest_checkpoint}")
        elif os.path.exists('best_model_80_target.pth'):
            config.resume_from = 'best_model_80_target.pth'
            print(f"🔄 自动检测到最佳模型: best_model_80_target.pth")
    elif args.resume:
        config.resume_from = args.resume
        config.start_epoch = args.epoch
    
    # Start training
    try:
        model, history, best_acc = train(config)
        
        # Final evaluation
        print("\n🔍 最终模型评估...")
        
        # Load best model
        if os.path.exists('best_model_80_target.pth'):
            checkpoint = torch.load('best_model_80_target.pth', map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_acc = checkpoint['best_acc']
            class_acc = checkpoint['class_acc']
            
            print(f"✅ Final Best Accuracy: {best_acc*100:.2f}%")
            print("📊 Per-class Accuracy:")
            for i, cls_name in enumerate(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']):
                if i < len(class_acc):
                    print(f"  {cls_name}: {class_acc[i]*100:6.2f}%")
            
            # Calculate average class accuracy
            avg_class_acc = np.mean(class_acc) * 100
            print(f"📈 Average Class Accuracy: {avg_class_acc:.2f}%")
            
            if best_acc >= 0.75:
                print(f"🎉 Congratulations! Reached 75%+ target!")
            elif best_acc >= 0.73:
                print(f"✅ Excellent! Accuracy > 73%")
            elif best_acc >= 0.72:
                print(f"📈 Good! Accuracy > 72%")
            else:
                print(f"⚠️ Needs further optimization")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()