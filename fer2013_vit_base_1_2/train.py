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
    'num_epochs': 100,
    'learning_rate': 4.5e-5,  # 从4.0e-5微调
    'weight_decay': 0.05,
    'cutmix_alpha': 0.8,
    'label_smoothing': 0.1,
    'drop_rate': 0.2,
    'grad_accum_steps': 2,
    'warmup_epochs': 10,
    'current_epoch': 9,
    'best_acc': 0.6945,
}

# === 关键优化点 ===
class AdvancedAugmentation:
    """针对第9个epoch结果的增强策略"""
    def __init__(self):
        # 基于第9个epoch各类别表现配置
        self.class_specific_params = {
            'disgust': {'prob': 0.5, 'alpha': 1.2},  # 49.5%准确率，需要突破50%
            'fear': {'prob': 0.6, 'alpha': 1.5},    # 39.9%准确率，需要重点突破
            'surprise': {'prob': 0.3, 'alpha': 0.8} # 85.7%准确率，维持即可
        }
    
    def get_augmentation(self, class_name):
        """获取类别特定的增强参数"""
        return self.class_specific_params.get(class_name, {'prob': 0.4, 'alpha': 1.0})

class DynamicWeightAdjuster:
    """动态权重调整器"""
    def __init__(self, base_weights):
        self.base_weights = base_weights
        self.previous_acc = None
    
    def update_weights(self, current_acc, previous_acc):
        """基于准确率变化调整权重"""
        new_weights = self.base_weights.copy()
        
        # disgust (49.5% → 目标52%)
        if current_acc[1] < 0.5 and current_acc[1] - previous_acc[1] < 0.05:
            new_weights[1] *= 1.8  # 大幅增加权重助力突破
        
        # fear (39.9% → 目标45%)
        if current_acc[2] < 0.45:
            new_weights[2] *= 2.0  # 最大权重
        
        # surprise (85.7% → 维持)
        if current_acc[6] > 0.85:
            new_weights[6] *= 0.8  # 降低权重
            
        return new_weights

# === 模型训练主循环 ===
def main():
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config['model_size'], num_classes=7).to(device)
    
    # 数据加载
    train_loader, val_loader = get_data_loaders(config['batch_size'])
    
    # 优化器配置
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    
    # 损失函数
    criterion = AdaptiveLoss()
    
    # 训练循环
    for epoch in range(config['current_epoch'], config['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_loss, class_acc = validate(model, val_loader, criterion, device)
        
        # 更新最佳模型
        if val_acc > config['best_acc']:
            config['best_acc'] = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc)
        
        # 打印进度
        print(f'Epoch {epoch+1}: Val Acc={val_acc:.4f}, Best Acc={config["best_acc"]:.4f}')
        print_class_accuracies(class_acc, val_loader.dataset.classes)

# === 关键组件实现 ===
class AdaptiveLoss(nn.Module):
    """自适应损失函数"""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.focal = FocalLoss(gamma=2)
        
    def forward(self, outputs, targets):
        # 基础交叉熵损失
        ce_loss = self.ce(outputs, targets)
        
        # 为困难类别添加Focal Loss
        focal_mask = (targets == 1) | (targets == 2)  # disgust和fear
        if focal_mask.any():
            focal_loss = self.focal(outputs[focal_mask], targets[focal_mask])
            return 0.7 * ce_loss + 0.3 * focal_loss
        return ce_loss

def create_model(model_size, num_classes):
    """创建模型并加载预训练权重"""
    model = timm.create_model(f'vit_{model_size}_patch16_224', 
                             pretrained=True,
                             num_classes=num_classes)
    return model

def get_data_loaders(batch_size):
    """获取数据加载器"""
    # 数据增强策略
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
    ])
    
    # 数据集
    train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
    val_dataset = datasets.ImageFolder('data/test', transform=val_transform)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def train_epoch(model, loader, optimizer, criterion, device):
    """训练单个epoch"""
    model.train()
    total_loss = 0
    
    for inputs, targets in tqdm(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * 7
    class_total = [0] * 7
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # 各类别准确率
            for i in range(7):
                mask = targets == i
                class_correct[i] += (predicted[mask] == i).sum().item()
                class_total[i] += mask.sum().item()
    
    # 计算各类别准确率
    class_acc = [c/t if t > 0 else 0 for c, t in zip(class_correct, class_total)]
    
    return correct/total, total_loss/len(loader), class_acc

def print_class_accuracies(accuracies, class_names):
    """打印各类别准确率"""
    print("\n各类别准确率:")
    for name, acc in zip(class_names, accuracies):
        print(f"{name}: {acc*100:.1f}%")

def save_checkpoint(model, optimizer, epoch, acc):
    """保存模型检查点"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': acc
    }, f'checkpoint_epoch_{epoch}.pth')

if __name__ == '__main__':
    main()