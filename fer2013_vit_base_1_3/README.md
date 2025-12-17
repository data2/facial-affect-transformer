以下是完整的稳定性优化版本代码，基于您的原始代码结构进行优化：

```python
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


# === 超参数配置 ===
config = {
    'model_size': 'base',
    'batch_size': 16,
    'num_epochs': 50,  # 减少训练轮次
    'learning_rate': 1e-5,  # 大幅降低学习率
    'weight_decay': 0.01,  # 增加权重衰减
    'cutmix_alpha': 0.5,  # 降低增强强度
    'label_smoothing': 0.1,
    'drop_rate': 0.2,
    'grad_accum_steps': 2,
    'warmup_epochs': 5,  # 减少热身轮次
    'current_epoch': 18,  # 从第18个epoch继续
    'best_acc': 0.6956,  # 当前最佳69.56%
    'patience': 8,  # 早停耐心值
}


# === 稳定性优化组件 ===
class StabilizedEarlyStopping:
    """稳定性优化的早停机制"""
    def __init__(self, patience=8, delta=0.001, min_epochs=5):
        self.patience = patience
        self.delta = delta
        self.min_epochs = min_epochs
        self.best_acc = 0
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_acc, model_weights=None):
        if val_acc > self.best_acc + self.delta:
            self.best_acc = val_acc
            self.counter = 0
            if model_weights is not None:
                self.best_weights = {k: v.clone() for k, v in model_weights.items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class StabilizedDataAugmentation:
    """稳定性数据增强"""
    def __init__(self, cutmix_prob=0.3, mixup_prob=0.2):  # 大幅降低增强概率
        self.cutmix_prob = cutmix_prob
        self.mixup_prob = mixup_prob
        self.epoch = 0
        
    def apply_augmentation(self, images, labels, epoch):
        """应用稳定性增强"""
        self.epoch = epoch
        
        # 降低增强概率，提高稳定性
        if np.random.rand() < self.cutmix_prob:
            images, targets_a, targets_b, lam = self.cutmix(images, labels, alpha=0.5)
            return images, targets_a, targets_b, lam, 'cutmix'
        
        elif np.random.rand() < self.mixup_prob:
            images, targets_a, targets_b, lam = self.mixup(images, labels, alpha=0.1)
            return images, targets_a, targets_b, lam, 'mixup'
        
        return images, labels, labels, 1.0, 'none'
    
    def cutmix(self, x, y, alpha=0.5):
        """CutMix增强"""
        if alpha <= 0:
            return x, y, y, 1.0
            
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        y_a, y_b = y, y[index]
        
        return x, y_a, y_b, lam
    
    def mixup(self, x, y, alpha=0.1):
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
    
    def rand_bbox(self, size, lam):
        """生成随机裁剪区域"""
        W = size[2]
        H = size[3]
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


class StabilizedLossFunction:
    """稳定性损失函数"""
    def __init__(self, class_weights=None, smoothing=0.1):
        self.class_weights = class_weights
        self.smoothing = smoothing
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
    def __call__(self, outputs, targets, augmentation_type='none'):
        # 基础交叉熵损失
        base_loss = self.ce_loss(outputs, targets)
        
        # 添加标签平滑
        if self.smoothing > 0:
            confidence = 1.0 - self.smoothing
            logprobs = F.log_softmax(outputs, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            smooth_loss = confidence * nll_loss + self.smoothing * smooth_loss
            base_loss = smooth_loss.mean()
        
        return base_loss


class StabilizedWeightAdjuster:
    """稳定性权重调整器"""
    def __init__(self, base_weights, max_adjustment=1.5):
        self.base_weights = base_weights.clone()
        self.max_adjustment = max_adjustment
        self.previous_acc = None
        
    def update_weights(self, current_acc, previous_acc):
        """基于准确率变化调整权重（更稳定）"""
        if previous_acc is None or current_acc is None:
            return self.base_weights.clone()
            
        new_weights = self.base_weights.clone()
        improvements = []
        
        for i in range(len(current_acc)):
            improvement = current_acc[i] - previous_acc[i]
            improvements.append((i, improvement))
        
        # 找出表现最差的3个类别
        worst_classes = sorted(improvements, key=lambda x: x[1])[:3]
        
        # 只对最差的类别进行适度调整
        for idx, improvement in worst_classes:
            if improvement < 0:  # 准确率下降
                adjustment = min(1.0 + abs(improvement) * 5, self.max_adjustment)
                new_weights[idx] *= adjustment
        
        return new_weights


# === 模型训练主循环 ===
def main():
    print("开始进入训练 - 稳定性优化版本")
    print("=" * 60)
    
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")
    
    # 创建稳定性数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 固定尺寸
        transforms.Grayscale(num_output_channels=3),
        
        # 温和的空间变换
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        
        # 温和的颜色变换
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # 降低遮挡概率
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    print("📁 加载数据集...")
    train_dir = './data/train'
    test_dir = './data/test'
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"❌ 训练数据路径不存在: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"❌ 测试数据路径不存在: {test_dir}")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    
    # 获取训练标签计算类别权重
    train_labels = [label for _, label in train_dataset]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print(f"📊 数据统计:")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    print(f"  类别数量: {len(train_dataset.classes)}")
    print(f"  类别名称: {train_dataset.classes}")
    
    # 数据加载器
    num_workers = min(8, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=num_workers, 
                             pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # 创建模型
    print("🔄 创建模型...")
    def create_stabilized_model(model_size='base', num_classes=7, weights_dir='./weights'):
        """创建稳定性优化模型"""
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
            
        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
            print("🔄 使用随机初始化模型...")
        
        return model
    
    model = create_stabilized_model(config['model_size'], num_classes=7)
    model = model.to(device)
    print("✅ 模型创建成功")
    
    # 优化器配置
    print("⚙️ 配置优化器...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度 - 使用更稳定的余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'], eta_min=1e-7
    )
    
    # 损失函数
    criterion = StabilizedLossFunction(class_weights=class_weights, smoothing=config['label_smoothing'])
    
    # 数据增强
    data_aug = StabilizedDataAugmentation(
        cutmix_prob=config['cutmix_alpha'],
        mixup_prob=0.2
    )
    
    # 权重调整器
    weight_adjuster = StabilizedWeightAdjuster(class_weights, max_adjustment=1.5)
    
    # 早停机制
    early_stopping = StabilizedEarlyStopping(
        patience=config['patience'],
        delta=0.001,
        min_epochs=5
    )
    
    # 尝试加载检查点
    def load_checkpoint(model, optimizer, filepath):
        """加载检查点"""
        if os.path.exists(filepath):
            print(f"🔄 加载检查点: {filepath}")
            checkpoint = torch.load(filepath, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_acc = checkpoint.get('best_acc', 0)
            return start_epoch, best_acc
        return 0, config['best_acc']
    
    # 加载检查点
    checkpoint_path = 'best_vit_base_targeted.pth'
    start_epoch, best_acc = load_checkpoint(model, optimizer, checkpoint_path)
    if start_epoch > 0:
        print(f"✅ 从第{start_epoch}个epoch恢复训练，最佳准确率: {best_acc*100:.2f}%")
    else:
        start_epoch = config['current_epoch']
        print(f"🔄 从第{start_epoch}个epoch开始训练")
    
    print(f"🎯 稳定性优化配置:")
    print(f"  学习率: {config['learning_rate']:.1e} (降低)")
    print(f"  CutMix概率: {data_aug.cutmix_prob} (降低)")
    print(f"  MixUp概率: {data_aug.mixup_prob} (降低)")
    print(f"  早停耐心值: {config['patience']}")
    
    # 训练函数
    def train_epoch(model, train_loader, criterion, optimizer, device, 
                   grad_accum_steps=2, epoch=0, previous_class_acc=None):
        """稳定性训练函数"""
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # 应用数据增强
            aug_images, targets_a, targets_b, lam, aug_type = data_aug.apply_augmentation(images, labels, epoch)
            
            outputs = model(aug_images)
            
            # 根据增强类型计算损失
            if aug_type == 'cutmix':
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            elif aug_type == 'mixup':
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, labels)
            
            # 梯度累积
            loss = loss / grad_accum_steps
            loss.backward()
            
            if (i + 1) % grad_accum_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accum_steps
        
        return total_loss / len(train_loader)
    
    def validate(model, val_loader, criterion, device):
        """验证模型"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = [0] * 7
        class_total = [0] * 7
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="验证中", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        acc = correct / total
        avg_loss = total_loss / len(val_loader)
        
        # 计算各类别准确率
        class_acc = []
        for i in range(7):
            if class_total[i] > 0:
                class_acc.append(class_correct[i] / class_total[i])
            else:
                class_acc.append(0.0)
        
        return acc, avg_loss, class_acc
    
    # 训练循环
    print(f"\n🚀 开始稳定性优化训练 (从第{start_epoch+1}个epoch开始)")
    print("=" * 60)
    
    training_history = {
        'train_loss': [], 'val_acc': [], 'val_loss': [], 'learning_rates': [], 'class_acc': []
    }
    
    previous_class_acc = None
    
    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start = time.time()
        print(f"\n📊 Epoch [{epoch+1}/{config['num_epochs']}]")
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, 
                               config['grad_accum_steps'], epoch, previous_class_acc)
        training_history['train_loss'].append(train_loss)
        
        # 验证
        val_acc, val_loss, class_acc = validate(model, val_loader, criterion, device)
        training_history['val_acc'].append(val_acc)
        training_history['val_loss'].append(val_loss)
        training_history['class_acc'].append(class_acc)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # 更新学习率
        scheduler.step()
        
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
            if previous_class_acc and i < len(previous_class_acc):
                improvement = class_acc[i] - previous_class_acc[i]
                arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
                print(f"  {cls_name}: {class_acc[i]*100:5.1f}% {arrow}")
            else:
                print(f"  {cls_name}: {class_acc[i]*100:5.1f}%")
        
        # 保存最佳模型
        if val_acc > best_acc + 0.001:
            best_acc = val_acc
            best_model_path = f'best_vit_{config["model_size"]}_stable.pth'
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
        if early_stopping(val_acc, {k: v.clone() for k, v in model.state_dict().items()}):
            print("🛑 早停触发，训练结束")
            # 如果早停，恢复最佳权重
            if early_stopping.best_weights is not None:
                model.load_state_dict(early_stopping.best_weights)
            break
        
        # 更新权重调整
        if previous_class_acc is not None:
            class_weights = weight_adjuster.update_weights(class_acc, previous_class_acc)
            criterion.class_weights = class_weights
        
        previous_class_acc = class_acc
        
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
    
    # 训练总结
    print("\n" + "=" * 60)
    print("🎯 训练总结")
    print("=" * 60)
    print(f"📊 最终最佳准确率: {best_acc*100:.2f}%")
    print(f"💾 最佳模型: best_vit_{config['model_size']}_stable.pth")
    print(f"🔄 总训练轮次: {epoch+1}")
    
    # 绘制训练曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(training_history['train_loss'], label='训练损失')
        plt.plot(training_history['val_loss'], label='验证损失')
        plt.legend()
        plt.title('损失曲线')
        
        plt.subplot(1, 2, 2)
        plt.plot(training_history['val_acc'], label='验证准确率', color='green')
        plt.axhline(y=best_acc, color='r', linestyle='--', label=f'最佳: {best_acc*100:.1f}%')
        plt.legend()
        plt.title('准确率曲线')
        
        plt.tight_layout()
        plt.savefig('training_curves_stable.png', dpi=300, bbox_inches='tight')
        print("📈 训练曲线已保存: training_curves_stable.png")
    except:
        print("⚠️ 无法绘制训练曲线，请安装matplotlib")
    
    # 最终验证
    print("\n🔍 最终模型验证...")
    try:
        checkpoint = torch.load(f'best_vit_{config["model_size"]}_stable.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        final_acc, final_loss, class_acc = validate(model, val_loader, criterion, device)
        
        print(f"✅ 最终验证准确率: {final_acc*100:.2f}%")
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
```

## 🔧 **稳定性优化的关键改进**

### 1. **降低学习率**
```python
'learning_rate': 1e-5,  # 从4.5e-5大幅降低
```

### 2. **减少数据增强强度**
```python
class StabilizedDataAugmentation:
    def __init__(self, cutmix_prob=0.3, mixup_prob=0.2):  # 大幅降低
```

### 3. **更温和的权重调整**
```python
class StabilizedWeightAdjuster:
    def __init__(self, base_weights, max_adjustment=1.5):  # 限制最大调整幅度
```

### 4. **改进的早停机制**
```python
class StabilizedEarlyStopping:
    def __init__(self, patience=8, delta=0.001):  # 增加耐心值
```

## 🎯 **预期效果**

基于当前69.56%的最佳结果，稳定性优化版本预期：

| 指标 | 优化前 | 预期稳定性优化后 |
|------|--------|------------------|
| **准确率波动** | ±3% | ±1% |
| **各类别稳定性** | 差 | 良好 |
| **收敛速度** | 快但不稳定 | 慢但稳定 |
| **最终准确率** | 69-70% | 70-72% |

## 🚀 **使用说明**

直接运行稳定性优化版本：
```bash
python train_stable.py
```

**这个版本应该能够显著减少训练波动，使各类别表现更加稳定，有望在稳定基础上实现小幅提升！**