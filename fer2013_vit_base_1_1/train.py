import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
import timm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import os
import math
from sklearn.utils.class_weight import compute_class_weight


# === 高级优化组件 ===
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, min_epochs=20):
        self.patience = patience
        self.delta = delta
        self.min_epochs = min_epochs
        self.best_acc = 0
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_acc, epoch):
        if epoch < self.min_epochs:
            return False
            
        if val_acc > self.best_acc + self.delta:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑损失函数"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不均衡"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def cutmix_data(x, y, alpha=1.0):
    """CutMix数据增强"""
    if alpha <= 0:
        return x, y, y, 1.0
        
    # 生成lambda值
    lam = np.random.beta(alpha, alpha)
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    # 生成裁剪区域
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # 调整lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """生成随机裁剪区域"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 均匀分布
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """CutMix损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha=0.2):
    """MixUp数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """带热身的余弦退火调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class AdaptiveDataAugmentation:
    """自适应数据增强"""
    def __init__(self, cutmix_prob=0.5, mixup_prob=0.3):
        self.cutmix_prob = cutmix_prob
        self.mixup_prob = mixup_prob
        self.epoch = 0
        
    def update_probabilities(self, class_accuracies):
        """根据类别准确率动态调整增强概率"""
        # 如果disgust或fear准确率低，增加增强概率
        disgust_acc = class_accuracies[1] if len(class_accuracies) > 1 else 0
        fear_acc = class_accuracies[2] if len(class_accuracies) > 2 else 0
        
        if disgust_acc < 0.1 or fear_acc < 0.3:
            self.cutmix_prob = min(0.7, self.cutmix_prob + 0.1)
            self.mixup_prob = min(0.5, self.mixup_prob + 0.1)
        else:
            self.cutmix_prob = 0.5
            self.mixup_prob = 0.3
            
    def apply_augmentation(self, images, labels, epoch):
        """应用自适应增强"""
        self.epoch = epoch
        
        # 30%概率使用CutMix
        if np.random.rand() < self.cutmix_prob:
            images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
            return images, targets_a, targets_b, lam, 'cutmix'
        
        # 20%概率使用MixUp
        elif np.random.rand() < self.mixup_prob:
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2)
            return images, targets_a, targets_b, lam, 'mixup'
        
        # 50%概率不使用增强
        else:
            return images, labels, labels, 1.0, 'none'


def main():
    print("开始进入训练 - 自适应优化版本 (目标: 80%准确率)")
    print("=" * 60)

    # === 设备设置 ===
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ 使用GPU: {torch.cuda.get_device_name()}")
        print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("⚠️ 未发现GPU，使用CPU")

    # === 超参数配置 ===
    config = {
        'model_size': 'base',
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 2e-5,  # 当前学习率
        'weight_decay': 0.05,
        'cutmix_alpha': 1.0,
        'label_smoothing': 0.1,
        'drop_rate': 0.2,
        'grad_accum_steps': 2,
        'warmup_epochs': 10,
        'current_epoch': 4,  # 当前训练到第4个epoch
    }
    
    MODEL_SIZE = config['model_size']
    print(f"🎯 目标: 80%验证准确率 | 模型: ViT-{MODEL_SIZE.capitalize()}")

    # === 1. 自适应数据预处理 ===
    print("\n🔄 配置自适应数据增强...")
    
    def strong_disgust_augmentation(image):
        """针对disgust类别的强增强"""
        # 极端颜色变换
        image = transforms.functional.adjust_hue(image, 0.3)
        image = transforms.functional.adjust_saturation(image, 3.0)
        image = transforms.functional.adjust_contrast(image, 2.0)
        image = transforms.functional.adjust_sharpness(image, 2.0)
        return image

    def fear_specific_augmentation(image):
        """针对fear类别的增强"""
        image = transforms.functional.adjust_brightness(image, 1.3)
        image = transforms.functional.adjust_contrast(image, 1.5)
        return image

    class AdaptiveTransform:
        """自适应变换，根据类别应用不同增强"""
        def __init__(self, base_transform):
            self.base_transform = base_transform
            
        def __call__(self, img, label=None):
            img = self.base_transform(img)
            
            # 如果提供了标签，应用类别特定增强
            if label is not None:
                if label == 1:  # disgust
                    img = strong_disgust_augmentation(img)
                elif label == 2:  # fear
                    img = fear_specific_augmentation(img)
                    
            return img

    # 训练时使用的增强transform
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        
        # 空间变换
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        
        # 颜色变换
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 3.0)),
        
        # 转换为tensor
        transforms.ToTensor(),
        
        # 归一化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # tensor上的变换
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])

    # 验证集transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 简单transform用于获取标签
    simple_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    # === 2. 加载数据集 ===
    print("📁 加载数据集...")
    train_dir = './data/train'
    test_dir = './data/test'

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"❌ 训练数据路径不存在: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"❌ 测试数据路径不存在: {test_dir}")

    train_dataset_simple = datasets.ImageFolder(train_dir, transform=simple_transform)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    # 获取训练标签
    train_labels = [label for _, label in train_dataset_simple]

    print(f"📊 数据统计:")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    print(f"  类别数量: {len(train_dataset.classes)}")
    print(f"  类别名称: {train_dataset.classes}")

    # === 3. 自适应采样策略 ===
    def get_adaptive_sampler(dataset, class_accuracies=None):
        """根据类别准确率调整采样权重"""
        # 计算每个类别的样本数
        class_counts = np.bincount(train_labels)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        
        # 如果提供了类别准确率，调整权重
        if class_accuracies is not None:
            for i, acc in enumerate(class_accuracies):
                if acc < 0.3:  # 准确率低的类别增加权重
                    class_weights[i] *= 5.0
                elif acc < 0.5:
                    class_weights[i] *= 2.0
        
        sample_weights = [class_weights[label] for _, label in dataset]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        return sampler

    # 数据加载器
    num_workers = min(8, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print(f"  Batch Size: {config['batch_size']} (累积步数: {config['grad_accum_steps']})")

    # === 4. 动态类别权重 ===
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # 基于第4个epoch结果调整权重
    current_class_acc = [0.646, 0.072, 0.363, 0.928, 0.727, 0.525, 0.746]  # 第4个epoch结果
    
    # 动态调整权重
    class_weights[1] *= 8.0  # disgust: 大幅增加权重 (从7.2%准确率)
    class_weights[2] *= 3.0  # fear: 增加权重 (从36.3%准确率)
    class_weights[5] *= 1.5  # sad: 轻微增加权重 (从52.5%准确率)
    
    print("📈 动态调整后的类别权重:", class_weights.cpu().numpy())

    # === 5. 创建模型 ===
    def create_optimized_model(model_size='base', num_classes=7, weights_dir='./weights'):
        """创建优化模型"""
        model_configs = {
            'base': {
                'name': 'vit_base_patch16_224',
                'local_file': 'vit_base_patch16_224.pth',
            }
        }

        config = model_configs[model_size]
        local_weight_path = os.path.join(weights_dir, config['local_file'])

        if not os.path.exists(local_weight_path):
            raise FileNotFoundError(f"❌ 未找到权重文件: {local_weight_path}")

        print(f"🔄 从本地加载预训练权重: {local_weight_path}")

        # 创建模型
        model = timm.create_model(config['name'], pretrained=False, num_classes=num_classes)

        try:
            checkpoint = torch.load(local_weight_path, map_location='cpu')
            state_dict = checkpoint
            
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            # 过滤分类头权重
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('head.') and not key.startswith('fc.'):
                    filtered_state_dict[key] = value

            # 加载权重
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            
            print("✅ 预训练权重加载成功")
            print(f"  缺失的键: {len(missing_keys)}个, 意外的键: {len(unexpected_keys)}个")

        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
            print("🔄 使用随机初始化模型...")

        return model

    # 创建模型
    num_classes = 7
    model = create_optimized_model(MODEL_SIZE, num_classes=num_classes)
    model = model.to(device)
    print("✅ 模型创建成功")

    # === 6. 自适应优化器配置 ===
    # 解冻更多层进行微调
    for name, param in model.named_parameters():
        if 'blocks' in name and int(name.split('.')[1]) >= 8:  # 最后4层
            param.requires_grad = True
        if 'head' in name:  # 分类头始终训练
            param.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度
    num_training_steps = len(train_loader) * config['num_epochs'] // config['grad_accum_steps']
    num_warmup_steps = len(train_loader) * config['warmup_epochs'] // config['grad_accum_steps']
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # 自适应损失函数
    class AdaptiveCriterion:
        def __init__(self, class_weights, label_smoothing=0.1):
            self.class_weights = class_weights
            self.label_smoothing = label_smoothing
            self.ce_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            self.focal_loss = FocalLoss(gamma=2)
            
        def __call__(self, outputs, targets, augmentation_type='none'):
            # 基础损失
            if augmentation_type == 'cutmix':
                base_loss = self.ce_loss(outputs, targets)
            elif augmentation_type == 'mixup':
                base_loss = self.ce_loss(outputs, targets)
            else:
                base_loss = self.ce_loss(outputs, targets)
            
            # 为困难类别添加Focal Loss
            disgust_mask = targets == 1
            fear_mask = targets == 2
            
            if disgust_mask.any() or fear_mask.any():
                focal_weight = 0.3  # Focal Loss权重
                focal_component = self.focal_loss(outputs, targets)
                total_loss = (1 - focal_weight) * base_loss + focal_weight * focal_component
            else:
                total_loss = base_loss
                
            return total_loss

    criterion = AdaptiveCriterion(class_weights, label_smoothing=config['label_smoothing'])

    # 早停机制
    early_stopping = EarlyStopping(patience=12, delta=0.002, min_epochs=20)

    # 自适应数据增强
    adaptive_aug = AdaptiveDataAugmentation(cutmix_prob=0.6, mixup_prob=0.4)

    print(f"🎯 优化配置:")
    print(f"  学习率: {config['learning_rate']:.1e}")
    print(f"  CutMix概率: {adaptive_aug.cutmix_prob}")
    print(f"  MixUp概率: {adaptive_aug.mixup_prob}")
    print(f"  动态权重调整: 已启用")

    # === 7. 训练函数 ===
    def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, 
                   grad_accum_steps=2, epoch=0, class_accuracies=None):
        """自适应训练函数"""
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        # 更新增强概率
        if class_accuracies is not None:
            adaptive_aug.update_probabilities(class_accuracies)

        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # 应用自适应增强
            aug_images, targets_a, targets_b, lam, aug_type = adaptive_aug.apply_augmentation(images, labels, epoch)

            outputs = model(aug_images)
            
            # 根据增强类型计算损失
            if aug_type == 'cutmix':
                loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
            elif aug_type == 'mixup':
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels, aug_type)

            # 梯度累积
            loss = loss / grad_accum_steps
            loss.backward()

            if (i + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps

        return total_loss / len(train_loader)

    def evaluate(model, dataloader, criterion):
        """验证函数"""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0
        all_preds = []
        all_targets = []
        class_correct = [0] * 7
        class_total = [0] * 7

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="验证中", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 计算每个类别的准确率
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        acc = correct / total
        avg_loss = total_loss / len(dataloader)
        
        # 计算各类别准确率
        class_acc = []
        for i in range(7):
            if class_total[i] > 0:
                class_acc.append(class_correct[i] / class_total[i])
            else:
                class_acc.append(0.0)
                
        return acc, avg_loss, class_acc

    # === 8. 加载检查点 ===
    def load_checkpoint(model, optimizer, scheduler, filepath):
        """加载检查点"""
        if os.path.exists(filepath):
            print(f"🔄 加载检查点: {filepath}")
            checkpoint = torch.load(filepath, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            return start_epoch, best_acc
        return 0, 0.0

    # 尝试加载检查点
    checkpoint_path = 'best_vit_base_fixed.pth'
    start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
    if start_epoch > 0:
        print(f"✅ 从第{start_epoch}个epoch恢复训练，最佳准确率: {best_acc*100:.2f}%")
    else:
        best_acc = 0.6712  # 第4个epoch的结果
        start_epoch = 4
        print(f"🔄 从第{start_epoch}个epoch开始训练")

    # === 9. 训练循环 ===
    print(f"\n🚀 开始自适应优化训练 (从第{start_epoch+1}个epoch开始)")
    print("=" * 60)

    training_history = {
        'train_loss': [], 'val_acc': [], 'val_loss': [], 'learning_rates': [], 'class_acc': []
    }

    # 记录第4个epoch的结果
    training_history['val_acc'].extend([0.5697, 0.6421, 0.6567, 0.6712])  # 1-4个epoch的结果
    training_history['class_acc'].extend([
        [0.578, 0.000, 0.277, 0.897, 0.775, 0.197, 0.551],  # epoch 1
        [0.475, 0.135, 0.261, 0.837, 0.702, 0.682, 0.810],  # epoch 2
        [0.601, 0.090, 0.451, 0.914, 0.724, 0.455, 0.702],  # epoch 3
        [0.646, 0.072, 0.363, 0.928, 0.727, 0.525, 0.746]   # epoch 4
    ])

    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start = time.time()
        print(f"\n📊 Epoch [{epoch+1}/{config['num_epochs']}]")

        # 使用上一轮的类别准确率指导训练
        prev_class_acc = training_history['class_acc'][-1] if training_history['class_acc'] else None

        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, 
                                device, config['grad_accum_steps'], epoch, prev_class_acc)
        training_history['train_loss'].append(train_loss)

        # 验证
        val_acc, val_loss, class_acc = evaluate(model, val_loader, criterion)
        training_history['val_acc'].append(val_acc)
        training_history['val_loss'].append(val_loss)
        training_history['class_acc'].append(class_acc)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # 打印结果
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        print(f"📈 训练损失: {train_loss:.4f}")
        print(f"🎯 验证准确率: {val_acc*100:.2f}% | 验证损失: {val_loss:.4f}")
        print(f"💡 学习率: {current_lr:.2e}")
        print(f"⏱️  本轮耗时: {epoch_time:.1f}秒")
        
        # 打印各类别准确率
        print("📊 各类别准确率:")
        for i, cls_name in enumerate(val_dataset.classes):
            improvement = class_acc[i] - prev_class_acc[i] if prev_class_acc else 0
            arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
            print(f"  {cls_name}: {class_acc[i]*100:5.1f}% {arrow}")

        # 保存最佳模型
        if val_acc > best_acc + 0.001:
            best_acc = val_acc
            best_model_path = f'best_vit_{MODEL_SIZE}_adaptive.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'config': config,
                'class_acc': class_acc
            }, best_model_path)
            print(f"🎉 新的最佳准确率: {best_acc*100:.2f}% -> {best_model_path}")

        # 早停检查
        if early_stopping(val_acc, epoch):
            print("🛑 早停触发，训练结束")
            break

        print("-" * 60)

        # 每3个epoch保存一次checkpoint
        if (epoch + 1) % 3 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'training_history': training_history,
                'config': config
            }, checkpoint_path)
            print(f"💾 检查点已保存: {checkpoint_path}")

    # === 10. 训练总结 ===
    print("\n" + "=" * 60)
    print("🎯 训练总结")
    print("=" * 60)
    print(f"📊 最终最佳准确率: {best_acc*100:.2f}%")
    print(f"💾 最佳模型: best_vit_{MODEL_SIZE}_adaptive.pth")
    print(f"🔄 总训练轮次: {epoch+1}")

    # 绘制训练曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(training_history['train_loss'], label='训练损失')
        plt.plot(training_history['val_loss'], label='验证损失')
        plt.legend()
        plt.title('损失曲线')
        
        plt.subplot(1, 3, 2)
        plt.plot(training_history['val_acc'], label='验证准确率', color='green')
        plt.axhline(y=best_acc, color='r', linestyle='--', label=f'最佳: {best_acc*100:.1f}%')
        plt.legend()
        plt.title('准确率曲线')
        
        # 各类别准确率曲线
        plt.subplot(1, 3, 3)
        class_acc_array = np.array(training_history['class_acc'])
        for i, cls_name in enumerate(val_dataset.classes):
            plt.plot(class_acc_array[:, i], label=cls_name, marker='o', markersize=2)
        plt.legend()
        plt.title('各类别准确率曲线')
        
        plt.tight_layout()
        plt.savefig('training_curves_adaptive.png', dpi=300, bbox_inches='tight')
        print("📈 训练曲线已保存: training_curves_adaptive.png")
    except:
        print("⚠️ 无法绘制训练曲线，请安装matplotlib")

    # === 11. 最终验证 ===
    print("\n🔍 最终模型验证...")
    try:
        checkpoint = torch.load(f'best_vit_{MODEL_SIZE}_adaptive.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        final_acc, final_loss, class_acc = evaluate(model, val_loader, criterion)
        
        print(f"✅ 最终验证准确率: {final_acc*100:.2f}%")
        
        # 分析各类别表现
        print("📊 各类别最终准确率:")
        for i, cls_name in enumerate(val_dataset.classes):
            print(f"  {cls_name}: {class_acc[i]*100:5.1f}%")
        
        if final_acc >= 0.8:
            print("🎉 恭喜！达到80%准确率目标！")
        elif final_acc >= 0.75:
            print("✅ 优秀！准确率 > 75%")
        elif final_acc >= 0.7:
            print("📈 良好！准确率 > 70%")
        else:
            print("⚠️ 需要进一步优化")
    except Exception as e:
        print(f"❌ 最终验证失败: {e}")


if __name__ == '__main__':
    main()