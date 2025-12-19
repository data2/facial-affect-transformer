# train_3090_with_attention_optimized_complete.py
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


# ==================== 3090优化配置 (内存<18G) - 完整优化版 ====================
class Config3090:
    """针对NVIDIA RTX 3090的优化配置，内存控制在18GB以下"""
    def __init__(self):
        # Basic configuration
        self.model_name = 'vit_base_patch16_224'
        self.num_classes = 7
        self.img_size = 224

        # 3090优化参数
        self.batch_size = 32
        self.num_epochs = 80
        self.learning_rate = 2.5e-5
        self.weight_decay = 0.05
        self.warmup_epochs = 10
        
        # Data augmentation (初始值)
        self.cutmix_prob = 0.45
        self.mixup_prob = 0.25
        self.cutmix_alpha = 0.7
        self.mixup_alpha = 0.1

        # Regularization
        self.drop_rate = 0.3
        self.label_smoothing = 0.1

        # Optimization strategies
        self.grad_accum_steps = 1
        self.patience = 25
        self.target_acc = 0.74

        # Class weights
        self.class_weights = None
        self.dynamic_weight_adjust = True
        
        # 注意力机制配置
        self.use_attention = True
        self.attention_type = 'se'
        self.attention_reduction = 16
        self.hard_class_focus = True
        
        # 续训配置
        # self.resume_from = 'checkpoint_3090_epoch_035_with_attention.pth'  # 从第35轮检查点继续
        # self.start_epoch = 35  # 已完成的轮数
        # self.resume_optimizer = True
        # self.resume_scheduler = True
        # self.resume_history = True

        # No resume training
        self.resume_from = None
        self.start_epoch = 0
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mixed precision
        self.use_amp = True
        self.amp_dtype = torch.float16
        
        # 梯度累积策略
        self.grad_clip = 1.0
        
        # 学习率调度策略
        self.use_warmup = True
        self.min_lr = 1e-6

    def __str__(self):
        info = "=" * 60 + "\n"
        info += "🎯 3090优化训练配置 (内存<18GB) - 完整优化版\n"
        info += "=" * 60 + "\n"
        info += f"📊 模型: {self.model_name} + {self.attention_type.upper()}注意力\n"
        info += f"📈 目标准确率: {self.target_acc*100:.1f}%\n"
        info += f"⚙️  批次大小: {self.batch_size}\n"
        info += f"📚 总轮数: {self.num_epochs}\n"
        if self.resume_from:
            info += f"🔄 续训从: {self.resume_from} (第{self.start_epoch}轮后)\n"
        info += f"💡 学习率: {self.learning_rate:.1e}\n"
        info += f"🔄 热身轮数: {self.warmup_epochs}\n"
        info += f"🎨 增强策略: CutMix({self.cutmix_prob}), MixUp({self.mixup_prob})\n"
        info += f"🎯 注意力机制: {self.attention_type.upper()} (启用)\n"
        info += f"⚡ 混合精度训练: {'启用' if self.use_amp else '禁用'}\n"
        info += f"💻 训练设备: {self.device}\n"
        info += f"📊 GPU显存: 24GB RTX 3090\n"
        info += "=" * 60
        return info


# ==================== Enhanced Data Augmentation ====================
class EnhancedAugmentation:
    """增强的数据增强策略"""
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.adaptive_aug_strength = 1.0  # 自适应增强强度

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
        """应用数据增强 - 自适应调整强度"""
        self.epoch = epoch
        
        # 自适应调整增强强度
        if epoch >= 35:
            # 第36轮后逐渐降低增强强度
            reduction_factor = max(0.5, 1.0 - (epoch - 35) * 0.02)
            self.adaptive_aug_strength = reduction_factor
        
        r = np.random.rand()

        # 根据自适应强度调整概率
        effective_cutmix_prob = self.config.cutmix_prob * self.adaptive_aug_strength
        effective_mixup_prob = self.config.mixup_prob * self.adaptive_aug_strength

        if r < effective_cutmix_prob:
            images, targets_a, targets_b, lam = self.cutmix(
                images, labels, self.config.cutmix_alpha * self.adaptive_aug_strength
            )
            return images, targets_a, targets_b, lam, 'cutmix'

        elif r < effective_cutmix_prob + effective_mixup_prob:
            images, targets_a, targets_b, lam = self.mixup(
                images, labels, self.config.mixup_alpha * self.adaptive_aug_strength
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


# ==================== Enhanced Attention Module ====================
class EnhancedSEAttention(nn.Module):
    """
    增强的Squeeze-and-Excitation注意力模块
    """
    def __init__(self, channel, reduction=16, num_classes=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 增强的MLP层
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
        # 类别感知注意力（新增）
        self.class_aware_attention = nn.Sequential(
            nn.Linear(channel, channel // 4),
            nn.ReLU(),
            nn.Linear(channel // 4, num_classes),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x, labels=None):
        # 全局平均和最大池化
        if x.dim() == 3:
            avg_out = self.mlp(self.avg_pool(x.transpose(1, 2)).squeeze(-1))
            max_out = self.mlp(self.max_pool(x.transpose(1, 2)).squeeze(-1))
        else:
            avg_out = self.mlp(self.avg_pool(x.unsqueeze(-1)).squeeze(-1))
            max_out = self.mlp(self.max_pool(x.unsqueeze(-1)).squeeze(-1))
        
        # 通道注意力权重
        attention = (avg_out + max_out) / 2.0
        
        if x.dim() == 3:
            attended_features = x * attention.unsqueeze(1)
        else:
            attended_features = x * attention
        
        # 类别感知注意力（训练时）
        if labels is not None and self.training:
            class_attention = self.class_aware_attention(
                attended_features.mean(dim=1) if attended_features.dim() == 3 else attended_features
            )
            return attended_features, class_attention
        
        return attended_features


# ==================== Adaptive Loss Function ====================
# ==================== Adaptive Loss Function - 修复版 ====================
class AdaptiveLossFunction:
    """自适应损失函数 - 修复版"""
    def __init__(self, class_weights, config):
        self.class_weights = class_weights
        self.config = config
        self.label_smoothing = config.label_smoothing
        self.epoch = 0
        self.training = True  # 添加training属性
        
        # 基础损失函数
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.focal_loss = FocalLoss(gamma=2.0, alpha=class_weights)
        
        # 类别困难度跟踪
        self.class_difficulty = torch.zeros(config.num_classes)
        self.class_samples = torch.zeros(config.num_classes)
    
    def train(self):
        """设置为训练模式"""
        self.training = True
        
    def eval(self):
        """设置为评估模式"""
        self.training = False
        
    def update_class_difficulty(self, outputs, targets):
        """更新类别困难度"""
        with torch.no_grad():
            probs = F.softmax(outputs, dim=1)
            pred_probs, preds = torch.max(probs, dim=1)
            
            for i in range(len(targets)):
                cls_idx = targets[i].item()
                if preds[i] == targets[i]:
                    # 正确预测，难度降低（置信度越高，难度降低越多）
                    self.class_difficulty[cls_idx] += pred_probs[i].item()
                else:
                    # 错误预测，难度增加（置信度越低，难度增加越多）
                    self.class_difficulty[cls_idx] -= (1 - pred_probs[i].item())
                self.class_samples[cls_idx] += 1
    
    def __call__(self, outputs, targets, augmentation_type='none', lam=1.0):
        # 基础交叉熵损失
        if self.label_smoothing > 0:
            ce_loss = self.label_smooth_ce(outputs, targets)
        else:
            ce_loss = self.ce_loss(outputs, targets)
        
        # 只在训练时更新类别困难度
        if self.training:
            self.update_class_difficulty(outputs, targets)
        
        # 自适应Focal Loss权重
        fear_mask = (targets == 2)  # fear
        disgust_mask = (targets == 1)  # disgust
        sad_mask = (targets == 5)  # sad

        if fear_mask.any() or disgust_mask.any() or sad_mask.any():
            # 动态调整Focal Loss权重（后期增加权重）
            if self.epoch > 35:  # 第36轮后
                focal_weight = 0.5  # 增加权重
            else:
                focal_weight = 0.3
            
            focal_component = self.focal_loss(outputs, targets)
            
            # 根据类别困难度调整损失（仅训练时且第30轮后）
            if self.training and self.epoch > 30:
                avg_difficulty = self.class_difficulty / (self.class_samples + 1e-8)
                
                # 特别关注困难类别
                for i in range(len(targets)):
                    if targets[i] == 2 or targets[i] == 5:  # fear或sad
                        difficulty = avg_difficulty[targets[i]]
                        if difficulty < 0.3:  # 非常困难的样本
                            focal_component = focal_component * 2.0
                        elif difficulty < 0.5:  # 中等困难的样本
                            focal_component = focal_component * 1.5
            
            total_loss = (1 - focal_weight) * ce_loss + focal_weight * focal_component
        else:
            total_loss = ce_loss

        return total_loss
    
    def label_smooth_ce(self, x, target):
        """标签平滑交叉熵"""
        confidence = 1.0 - self.label_smoothing
        
        if target.dim() > 1:
            target = target.squeeze()
        
        logprobs = F.log_softmax(x, dim=-1)
        
        if target.dim() == 1:
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
        else:
            nll_loss = -torch.sum(logprobs * target, dim=-1)
        
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


# ==================== Advanced Dynamic Weight Adjuster ====================
class AdvancedWeightAdjuster:
    """高级动态权重调整器"""
    def __init__(self, base_weights, history_size=5):
        self.base_weights = base_weights.clone()
        self.history_size = history_size
        self.history = []
        self.performance_history = []
        self.improvement_threshold = 0.01  # 改进阈值

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
            improvement = current_acc - recent_avg

            # 特别处理困难类别
            if i == 2 or i == 5:  # fear和sad
                if current_acc < 0.50:  # 表现很差
                    if improvement < -0.05:  # 大幅下降
                        new_weights[i] *= 3.0  # 显著增加权重
                        print(f"    ⚠️  {['fear','sad'][i==5]}: 大幅下降({improvement:.3f})，权重×3.0")
                    elif improvement < 0:  # 轻微下降
                        new_weights[i] *= 2.0
                    else:  # 没有改进
                        new_weights[i] *= 1.5
                elif 0.50 <= current_acc < 0.65:  # 中等表现
                    if improvement < -0.03:  # 下降
                        new_weights[i] *= 1.8
                    elif improvement < 0.02:  # 停滞
                        new_weights[i] *= 1.3
                    else:  # 改进
                        new_weights[i] *= 1.1
                elif current_acc >= 0.65:  # 表现良好
                    new_weights[i] *= 0.9  # 降低权重

            # 其他类别
            else:
                if current_acc < 0.55 and improvement < -0.04:
                    new_weights[i] *= 1.8
                elif 0.55 <= current_acc < 0.75 and abs(improvement) < 0.02:
                    new_weights[i] *= 1.2
                elif current_acc > 0.80:
                    new_weights[i] *= 0.7

        return new_weights


# ==================== Adaptive Early Stopping ====================
class AdaptiveEarlyStopping:
    """自适应早停机制"""
    def __init__(self, patience=15, min_epochs=30, target_acc=0.75):
        self.patience = patience
        self.original_patience = patience
        self.min_epochs = min_epochs
        self.target_acc = target_acc
        self.best_acc = 0
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.improvement_history = []
        
    def __call__(self, val_acc, epoch, train_acc=None):
        if epoch < self.min_epochs:
            return False
        
        # 计算改进
        improvement = val_acc - self.best_acc
        self.improvement_history.append(improvement)
        
        # 自适应调整耐心值
        if val_acc > self.target_acc - 0.02:  # 接近目标
            self.patience = max(self.original_patience, 20)
        elif val_acc < self.target_acc - 0.05:  # 远离目标
            self.patience = min(self.original_patience, 10)
        
        if val_acc > self.best_acc + 1e-4:
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.counter = 0
            print(f"    ✅ 准确率提升: +{improvement*100:.3f}%")
        else:
            self.counter += 1
            if improvement < -0.02:  # 显著下降
                self.counter += 1  # 加速早停
                print(f"    ⚠️  准确率下降: {improvement*100:.3f}%，加速早停")
        
        if self.counter >= self.patience:
            self.early_stop = True
            print(f"    🛑 早停触发: {self.counter}轮无改进")
        
        return self.early_stop


# ==================== Model Creation with Enhanced Attention ====================
def create_enhanced_model(config, pretrained_path='./weights/vit_base_patch16_224.pth'):
    """创建增强的模型 - 集成改进的注意力机制"""
    print(f"🔄 创建增强模型: {config.model_name} (带{config.attention_type.upper()}注意力)")
    
    # 创建基础ViT模型
    model = timm.create_model(
        config.model_name,
        pretrained=False,
        num_classes=config.num_classes,
        drop_rate=config.drop_rate
    )
    
    # 加载预训练权重
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
                for i, key in enumerate(missing_keys[:3]):
                    print(f"    {i+1}. {key}")
            if unexpected_keys:
                print(f"⚠️  意外的键: {len(unexpected_keys)}个")

            print("✅ 预训练权重加载成功")

        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
            print("🔄 使用随机初始化...")
    else:
        print("⚠️ 未找到预训练权重，使用随机初始化")
    
    # =============== 集成增强的注意力模块 ===============
    if config.use_attention:
        print(f"🎯 集成增强的{config.attention_type.upper()}注意力模块")
        
        # 获取原始分类头
        original_head = None
        if hasattr(model, 'head'):
            original_head = model.head
            model.head = nn.Identity()
        elif hasattr(model, 'fc'):
            original_head = model.fc
            model.fc = nn.Identity()
        
        # 特征维度
        if hasattr(model, 'num_features'):
            feature_dim = model.num_features
        else:
            feature_dim = 768
        
        # 创建增强的注意力模块
        attention = EnhancedSEAttention(
            channel=feature_dim, 
            reduction=config.attention_reduction,
            num_classes=config.num_classes
        )
        
        # 创建新的分类头
        if original_head is not None and isinstance(original_head, nn.Linear):
            new_head = nn.Linear(feature_dim, config.num_classes)
            if original_head.weight.shape[0] == config.num_classes and original_head.weight.shape[1] == feature_dim:
                new_head.weight.data.copy_(original_head.weight.data)
                new_head.bias.data.copy_(original_head.bias.data)
                print(f"✅ 复用原始分类头权重 (维度: {feature_dim}->{config.num_classes})")
            else:
                print(f"⚠️  原始分类头维度不匹配，使用随机初始化")
        else:
            new_head = nn.Linear(feature_dim, config.num_classes)
            print(f"✅ 创建新的分类头 (维度: {feature_dim}->{config.num_classes})")
        
        # 创建新的前向传播
        # 在create_enhanced_model函数中，修复前向传播函数
        def new_forward(x, labels=None):
            # 获取特征
            features = model.forward_features(x)
            
            # 如果特征是3D [batch, num_patches, dim]，取全局平均
            if features.dim() == 3:
                features = features.mean(dim=1)
            
            # 应用注意力
            if labels is not None and model.training:  # 使用model.training而不是self.training
                attended_features, class_attention = attention(features, labels)
            else:
                attended_features = attention(features)
            
            # 分类
            output = new_head(attended_features)
            
            if labels is not None and model.training:  # 使用model.training
                return output, class_attention
            return output
        
        # 替换前向传播
        model.forward = new_forward
        
        # 保存组件
        model.attention = attention
        model.new_head = new_head
        model.has_attention = True
        
        print(f"✅ 增强注意力模块集成完成 (特征维度: {feature_dim})")
    
    return model.to(config.device)


# ==================== 优化后的训练函数 ====================
def train_epoch_optimized(model, train_loader, criterion, optimizer, scheduler,
                          data_aug, config, epoch, weight_adjuster=None, scaler=None):
    """针对3090优化的训练epoch - 完整优化版"""
    
    # =============== 关键调整点 ===============
    if epoch == 24:  # 第25轮
        optimizer.param_groups[0]['lr'] = 2.0e-05
        print(f"🎯 第{epoch+1}轮：学习率调整为2.00e-05")
    
    if epoch == 35:  # 第36轮（已经应用）
        config.cutmix_prob = 0.35
        config.mixup_prob = 0.15
        optimizer.param_groups[0]['lr'] = 1.8e-05
        if hasattr(model, 'drop_rate'):
            model.drop_rate = 0.4
        print(f"🚨 第{epoch+1}轮：综合调整对抗过拟合")
    
    # 新增：第40轮进一步优化（关键调整）
    if epoch == 39:  # 第40轮
        # 1. 进一步降低数据增强强度
        config.cutmix_prob = 0.20
        config.mixup_prob = 0.08
        
        # 2. 显著降低学习率（精细调整阶段）
        optimizer.param_groups[0]['lr'] = 8.0e-06
        
        # 3. 增加正则化
        if hasattr(model, 'drop_rate'):
            model.drop_rate = 0.5
        
        # 4. 增加标签平滑
        if hasattr(criterion, 'label_smoothing'):
            criterion.label_smoothing = 0.15
        
        print(f"🔒 第{epoch+1}轮：启动精细调整阶段")
        print(f"  📉 数据增强: CutMix={config.cutmix_prob}, MixUp={config.mixup_prob}")
        print(f"  📉 学习率: 1.6e-05 → 8.0e-06")
        print(f"  🔧 Dropout: {model.drop_rate}")
        
        # 5. 冻结部分网络层（减少过拟合）
        freeze_blocks = ['patch_embed', 'blocks.0', 'blocks.1']
        for name, param in model.named_parameters():
            if any(block in name for block in freeze_blocks):
                param.requires_grad = False
        
        # 验证冻结效果
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  🔍 参数状态: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.1f}%)可训练")
    
    # 新增：第50轮学习率再次衰减
    if epoch == 49:  # 第50轮
        optimizer.param_groups[0]['lr'] = 3.0e-06
        print(f"🎯 第{epoch+1}轮：进入超精细调整阶段，学习率: 3.0e-06")
    
    # =============== 训练循环 ===============
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
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(aug_images)
                
                if aug_type == 'cutmix' or aug_type == 'mixup':
                    targets_a = targets_a.long()
                    targets_b = targets_b.long()
                    loss_a = criterion(outputs, targets_a)
                    loss_b = criterion(outputs, targets_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss = criterion(outputs, labels)
                
                loss = loss / config.grad_accum_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(aug_images)
            
            if aug_type == 'cutmix' or aug_type == 'mixup':
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, labels)
            
            loss = loss / config.grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

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
    total_augmented = sum(aug_stats.values())
    for aug_type, count in aug_stats.items():
        percentage = count / total_batches * 100
        print(f"  {aug_type}: {count:3d}/{total_batches:3d} ({percentage:5.1f}%)")
    print(f"  Total augmented batches: {total_augmented}/{total_batches} ({total_augmented/total_batches*100:.1f}%)")

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


# ==================== 增强的验证函数 ====================
def enhanced_validate(model, val_loader, criterion, config, epoch):
    """增强的验证函数"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * config.num_classes
    class_total = [0] * config.num_classes
    
    # 混淆矩阵统计
    confusion_matrix = np.zeros((config.num_classes, config.num_classes), dtype=int)
    
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
                pred = predicted[i]
                class_correct[label] += (pred == label).item()
                class_total[label] += 1
                confusion_matrix[label][pred] += 1

    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(config.num_classes):
        if class_total[i] > 0:
            class_accuracies.append(class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0.0)

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0

    # 分析混淆矩阵
    if epoch % 5 == 0 and epoch > 30:
        print("\n🔍 混淆矩阵分析:")
        class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # 特别关注fear和sad的混淆
        fear_confusions = confusion_matrix[2]
        sad_confusions = confusion_matrix[5]
        
        print(f"  Fear主要被误判为: ", end="")
        for i in range(config.num_classes):
            if i != 2 and fear_confusions[i] > fear_confusions[2] * 0.1:  # 超过10%的误判
                print(f"{class_names[i]}({fear_confusions[i]}) ", end="")
        print()
        
        print(f"  Sad主要被误判为: ", end="")
        for i in range(config.num_classes):
            if i != 5 and sad_confusions[i] > sad_confusions[5] * 0.1:
                print(f"{class_names[i]}({sad_confusions[i]}) ", end="")
        print()

    return accuracy, avg_loss, class_accuracies, confusion_matrix


# ==================== 主训练函数 ====================
def train_optimized_3090(config):
    """优化的主训练函数"""
    print(config)
    print("\n🚀 开始优化训练!")
    if config.resume_from:
        print(f"🔄 续训模式：从 {config.resume_from} 继续训练")
    print("=" * 60)
    
    # 显示GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 显存: {gpu_memory:.1f} GB")
        print(f"📊 Batch Size: {config.batch_size}")
        print(f"⚡ 混合精度训练: {'启用' if config.use_amp else '禁用'}")
        if config.use_attention:
            print(f"🎯 注意力机制: {config.attention_type.upper()} (启用)")

    # 创建数据增强
    data_aug = EnhancedAugmentation(config)
    
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
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # 创建模型
    model = create_enhanced_model(config)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    print(f"✅ 优化器: AdamW (lr={config.learning_rate:.2e}, weight_decay={config.weight_decay})")

    # 学习率调度器
    if config.use_warmup and config.warmup_epochs > 0:
        print(f"🔥 启用学习率Warmup ({config.warmup_epochs}轮)")
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=config.warmup_epochs
        )
        
        after_warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.num_epochs - config.warmup_epochs,
            eta_min=config.min_lr
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, after_warmup_scheduler],
            milestones=[config.warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.num_epochs,
            eta_min=config.min_lr
        )
    
    # ==================== 续训逻辑 ====================
    start_epoch = 0
    best_acc = 0
    best_epoch = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'class_acc': [], 'learning_rates': []
    }
    
    # 动态权重调整器
    if config.dynamic_weight_adjust:
        weight_adjuster = AdvancedWeightAdjuster(class_weights)
    else:
        weight_adjuster = None
    
    # 加载检查点
    if config.resume_from and os.path.exists(config.resume_from):
        print(f"\n📥 加载续训检查点: {config.resume_from}")
        try:
            checkpoint = torch.load(config.resume_from, map_location=config.device)
            
            # 检查检查点是否包含注意力模块
            checkpoint_has_attention = checkpoint.get('has_attention', False)
            model_has_attention = hasattr(model, 'has_attention')
            
            if checkpoint_has_attention and model_has_attention:
                print("✅ 检查点包含注意力模块，与当前模型匹配")
            elif not checkpoint_has_attention and model_has_attention:
                print("⚠️  检查点不包含注意力模块，但当前模型有注意力模块")
                print("🔄 将加载基础模型权重，注意力模块随机初始化")
            elif checkpoint_has_attention and not model_has_attention:
                print("⚠️  检查点包含注意力模块，但当前模型没有")
                print("🔄 将忽略检查点中的注意力权重")
            
            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ 模型权重加载成功")
            
            # 加载优化器状态
            if config.resume_optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ 优化器状态加载成功")
            
            # 加载调度器状态
            if config.resume_scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✅ 调度器状态加载成功")
            
            # 加载训练历史
            if config.resume_history and 'history' in checkpoint:
                history = checkpoint['history']
                print(f"✅ 训练历史加载成功（已训练 {len(history['train_loss'])} 轮）")
            
            # 加载最佳准确率
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
                best_epoch = checkpoint.get('epoch', config.start_epoch)
                print(f"✅ 最佳准确率: {best_acc*100:.2f}% (第{best_epoch+1}轮)")
            
            # 设置起始epoch
            start_epoch = checkpoint.get('epoch', config.start_epoch) + 1
            print(f"🔄 从第 {start_epoch} 轮继续训练")
            
            # 恢复动态权重
            if 'class_acc' in checkpoint and weight_adjuster:
                recent_acc = checkpoint['class_acc']
                weight_adjuster.history = [recent_acc]
                print("✅ 动态权重调整器已初始化")
                
        except Exception as e:
            print(f"❌ 检查点加载失败: {e}")
            print("🔄 使用默认配置开始训练")
            start_epoch = 0
    
    # 创建损失函数
    criterion = AdaptiveLossFunction(class_weights, config)
    
    # 创建早停机制
    early_stopping = AdaptiveEarlyStopping(
        patience=config.patience,
        min_epochs=30,
        target_acc=config.target_acc
    )
    
    # 混合精度训练
    if config.use_amp:
        try:
            scaler = torch.amp.GradScaler('cuda')
            print("✅ 已启用混合精度训练 (torch.amp API)")
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()
            print("✅ 已启用混合精度训练 (torch.cuda.amp API)")
    else:
        scaler = None
        print("⏭️  混合精度训练已禁用")
    
    # 训练循环
    print(f"\n🚀 开始训练循环 (从第{start_epoch}轮开始，目标: {config.target_acc*100:.1f}%)")
    print("=" * 60)
    
    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()
        
        # 更新损失函数的epoch
        criterion.epoch = epoch
        
        # 训练
        train_loss, train_acc = train_epoch_optimized(
            model, train_loader, criterion, optimizer, scheduler,
            data_aug, config, epoch, weight_adjuster, scaler
        )
        
        # 验证
        val_acc, val_loss, class_acc, confusion_matrix = enhanced_validate(
            model, val_loader, criterion, config, epoch
        )
        
        # 在epoch结束时调用调度器
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
        
        # 特别关注困难类别的表现
        if class_acc:
            print(f"  🎯 困难类别准确率 - fear: {class_acc[2]*100:.2f}%, sad: {class_acc[5]*100:.2f}%")
            
            # 计算训练-验证差距
            train_val_gap = train_acc - val_acc
            if train_val_gap > 0.10:  # 差距大于10%
                print(f"  ⚠️  训练-验证差距较大: {train_val_gap*100:.2f}% (可能过拟合)")
            elif train_val_gap < 0.02:  # 差距小于2%
                print(f"  ✅ 训练-验证差距良好: {train_val_gap*100:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc + 0.0001:
            best_acc = val_acc
            best_epoch = epoch
            save_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'epoch': epoch,
                'best_acc': best_acc,
                'val_acc': val_acc,
                'class_acc': class_acc,
                'config': config.__dict__,
                'history': history,
                'has_attention': hasattr(model, 'has_attention')
            }
            
            filename = 'best_model_3090_with_attention.pth'
            torch.save(save_dict, filename)
            print(f"  🎉 New Best Accuracy: {best_acc*100:.2f}% (saved to {filename})")
        
        # 动态权重调整
        if weight_adjuster and epoch >= 5:
            new_weights = weight_adjuster.update_weights(class_acc, epoch)
            criterion.class_weights = new_weights
            print("  ⚖️  动态权重调整完成")
        
        # 早停检查
        if early_stopping(val_acc, epoch, train_acc):
            print(f"\n🛑 Early stopping triggered! Best Accuracy: {early_stopping.best_acc*100:.2f}% "
                  f"(Epoch {early_stopping.best_epoch+1})")
            break
        
        # 每5轮保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'checkpoint_3090_epoch_{epoch+1:03d}_with_attention.pth'
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'class_acc': class_acc,
                'history': history,
                'has_attention': hasattr(model, 'has_attention')
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
    print(f"🔄 Total Training Epochs: {epoch + 1} (从第{start_epoch}轮开始)")
    if config.use_attention:
        print(f"🎯 注意力机制: {config.attention_type.upper()} (启用)")
    
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
    
    # 打印配置信息
    print(config)
    
    # 开始训练
    try:
        model, history, best_acc = train_optimized_3090(config)
        
        # 最终评估
        print("\n🔍 最终模型评估...")
        
        model_file = 'best_model_3090_with_attention.pth'
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_acc = checkpoint['best_acc']
            class_acc = checkpoint.get('class_acc', [])
            
            print(f"✅ Final Best Accuracy: {best_acc*100:.2f}%")
            
            if class_acc:
                print("📊 Per-class Accuracy:")
                class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
                for i, (cls_name, acc) in enumerate(zip(class_names, class_acc)):
                    if i < len(class_acc):
                        print(f"  {cls_name}: {acc*100:6.2f}%")
            
            # 计算平均类别准确率
            avg_class_acc = np.mean(class_acc) * 100 if class_acc else 0
            print(f"📈 Average Class Accuracy: {avg_class_acc:.2f}%")
            
            # 特别关注困难类别
            if len(class_acc) > 5:
                hard_class_avg = (class_acc[2] + class_acc[5]) / 2 * 100
                print(f"🎯 困难类别(fear+sad)平均: {hard_class_avg:.2f}%")
            
            if best_acc >= 0.74:
                print(f"🎉 成功突破74%!")
            elif best_acc >= 0.73:
                print(f"✅ 达到73%以上!")
            else:
                print(f"📈 最终准确率: {best_acc*100:.2f}%")
    
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
