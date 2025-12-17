import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import os
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
import math
import argparse
import os
from pathlib import Path


# === 命令行参数解析 ===
def get_config():
    parser = argparse.ArgumentParser(description='表情识别训练脚本')

    # 设备配置
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'cuda:0', 'cuda:1'],
                        help='训练设备: auto(自动选择), cuda, cpu, cuda:0等')

    # 模型配置
    parser.add_argument('--model_size', type=str, default='huge',
                        choices=['tiny', 'small', 'base', 'large', 'huge'],
                        help='模型大小')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch大小')

    # 权重加载配置
    parser.add_argument('--weights_dir', type=str, default='./vit_weights',
                        help='预训练权重目录')
    parser.add_argument('--use_pretrained', action='store_true', default=True,
                        help='使用预训练权重')
    parser.add_argument('--force_download', action='store_true', default=False,
                        help='强制重新下载权重')

    # 安全模式配置
    parser.add_argument('--safe_mode', action='store_true', default=False,
                        help='启用安全模式')
    parser.add_argument('--memory_limit', type=int, default=12,
                        help='显存限制(GB)')
    parser.add_argument('--grad_accum_steps', type=int, default=8,
                        help='梯度累积步数')
    parser.add_argument('--check_interval', type=int, default=300,
                        help='安全检查间隔(秒)')

    # 训练配置
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮次')
    parser.add_argument('--lr', type=float, default=-1,
                        help='学习率，-1为自动设置')

    args = parser.parse_args()
    return args


# 全局配置
config = get_config()

# === GPU优化：混合精度训练 ===
try:
    from torch.cuda.amp import autocast, GradScaler

    AMP_AVAILABLE = True
    print("✅ 混合精度训练可用")
except ImportError:
    AMP_AVAILABLE = False
    print("⚠️ 混合精度训练不可用，使用普通训练")


# === 智能权重加载器 - 修复版本 ===
class SmartWeightLoader:
    def __init__(self, weights_dir='./vit_weights', use_pretrained=True, force_download=False):
        self.weights_dir = Path(weights_dir)
        self.use_pretrained = use_pretrained
        self.force_download = force_download
        self.weights_dir.mkdir(exist_ok=True, parents=True)

        # 完整的模型配置表 - 修复版本
        self.model_configs = {
            'tiny': {
                'name': 'vit_tiny_patch16_224',
                'hf_repo': 'google/vit-tiny-patch16-224',
                'hidden_size': 192,
                'intermediate_size': 768,  # 192 * 4
                'num_hidden_layers': 12,
                'num_attention_heads': 3,
                'patch_size': 16,
                'lr': 5e-5
            },
            'small': {
                'name': 'vit_small_patch16_224',
                'hf_repo': 'google/vit-small-patch16-224',
                'hidden_size': 384,
                'intermediate_size': 1536,  # 384 * 4
                'num_hidden_layers': 12,
                'num_attention_heads': 6,
                'patch_size': 16,
                'lr': 3e-5
            },
            'base': {
                'name': 'vit_base_patch16_224',
                'hf_repo': 'google/vit-base-patch16-224',
                'hidden_size': 768,
                'intermediate_size': 3072,  # 768 * 4
                'num_hidden_layers': 12,
                'num_attention_heads': 12,
                'patch_size': 16,
                'lr': 2e-5
            },
            'large': {
                'name': 'vit_large_patch16_224',
                'hf_repo': 'google/vit-large-patch16-224',
                'hidden_size': 1024,
                'intermediate_size': 4096,  # 1024 * 4
                'num_hidden_layers': 24,
                'num_attention_heads': 16,
                'patch_size': 16,
                'lr': 1e-5
            },
            'huge': {
                'name': 'vit_huge_patch14_224',
                'hf_repo': 'google/vit-huge-patch14-224-in21k',
                'hidden_size': 1280,
                'intermediate_size': 5120,  # 关键修复：ViT-Huge的中间层维度是5120，不是3072
                'num_hidden_layers': 32,
                'num_attention_heads': 16,
                'patch_size': 14,
                'lr': 5e-6
            }
        }

    def get_local_weight_path(self, model_size):
        """获取本地权重文件路径"""
        hf_files = [
            self.weights_dir / "pytorch_model.bin",
            self.weights_dir / "model.safetensors"
        ]

        for file_path in hf_files:
            if file_path.exists():
                file_size = file_path.stat().st_size / (1024 ** 3)
                print(f"✅ 找到权重文件: {file_path} ({file_size:.2f} GB)")
                return file_path

        print("❌ 未找到本地权重文件")
        return None

    def create_huggingface_model(self, model_size, num_classes=7):
        """创建HuggingFace模型 - 修复版本"""
        from transformers import ViTForImageClassification, ViTConfig
        
        if model_size not in self.model_configs:
            raise ValueError(f"不支持的模型大小: {model_size}")
            
        model_config = self.model_configs[model_size]
        hf_repo = model_config['hf_repo']
        
        print(f"🔄 创建HuggingFace模型: {hf_repo}")
        print(f"🎯 关键配置验证:")
        print(f"  隐藏层维度: {model_config['hidden_size']}")
        print(f"  中间层维度: {model_config['intermediate_size']}")  # 关键修复
        print(f"  层数: {model_config['num_hidden_layers']}")
        print(f"  注意力头数: {model_config['num_attention_heads']}")
        print(f"  Patch大小: {model_config['patch_size']}")

        try:
            # 明确指定所有关键参数
            vit_config = ViTConfig(
                image_size=224,
                patch_size=model_config['patch_size'],
                num_channels=3,
                hidden_size=model_config['hidden_size'],
                num_hidden_layers=model_config['num_hidden_layers'],
                num_attention_heads=model_config['num_attention_heads'],
                intermediate_size=model_config['intermediate_size'],  # 关键修复！
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                num_labels=num_classes
            )
            
            # 首先尝试从本地加载
            local_path = self.get_local_weight_path(model_size)
            if local_path and not self.force_download:
                print(f"📁 从本地文件加载: {local_path}")
                model = ViTForImageClassification.from_pretrained(
                    str(local_path.parent),
                    config=vit_config,
                    ignore_mismatched_sizes=True
                )
            else:
                # 从HuggingFace Hub下载
                print(f"🌐 从HuggingFace Hub下载: {hf_repo}")
                model = ViTForImageClassification.from_pretrained(
                    hf_repo,
                    config=vit_config,
                    ignore_mismatched_sizes=True
                )
                
                # 保存到本地
                if local_path:
                    model.save_pretrained(self.weights_dir)
                    print(f"💾 权重已保存到: {self.weights_dir}")

            # 验证最终配置
            actual_config = model.config
            print(f"✅ 模型创建成功")
            print(f"🔍 最终配置验证:")
            print(f"  实际隐藏层: {actual_config.hidden_size}")
            print(f"  实际中间层: {actual_config.intermediate_size}")
            print(f"  实际层数: {actual_config.num_hidden_layers}")
            
            # 检查维度匹配
            if (actual_config.hidden_size == model_config['hidden_size'] and 
                actual_config.intermediate_size == model_config['intermediate_size']):
                print("🎉 所有维度匹配成功!")
            else:
                print(f"⚠️ 警告: 维度不匹配!")
                print(f"  期望隐藏层: {model_config['hidden_size']}, 实际: {actual_config.hidden_size}")
                print(f"  期望中间层: {model_config['intermediate_size']}, 实际: {actual_config.intermediate_size}")

            return model

        except Exception as e:
            print(f"❌ HuggingFace模型创建失败: {e}")
            # 回退方案：创建随机初始化的正确模型
            print("🔄 创建随机初始化的正确模型...")
            model = ViTForImageClassification(vit_config)
            print("✅ 随机初始化模型创建成功")
            return model

    def diagnostic_check(self, model, train_loader, device):
        """诊断模型和数据"""
        print("\n" + "="*60)
        print("🔍 模型诊断检查")
        print("="*60)
        
        # 1. 检查模型配置
        print("📊 模型配置:")
        print(f"  隐藏层维度: {model.config.hidden_size}")
        print(f"  中间层维度: {model.config.intermediate_size}")  # 关键！
        print(f"  层数: {model.config.num_hidden_layers}")
        print(f"  注意力头数: {model.config.num_attention_heads}")
        
        # 2. 检查参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📈 参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        # 3. 检查一个batch的数据
        model.eval()
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                print(f"🔍 输入数据检查:")
                print(f"  输入形状: {images.shape}")
                print(f"  标签形状: {labels.shape}")
                print(f"  像素范围: [{images.min():.3f}, {images.max():.3f}]")
                print(f"  均值: {images.mean():.3f}, 标准差: {images.std():.3f}")
                
                # 前向传播测试
                outputs = model(pixel_values=images)
                logits = outputs.logits
                    
                print(f"🎯 输出检查:")
                print(f"  输出形状: {logits.shape}")
                print(f"  输出范围: [{logits.min():.3f}, {logits.max():.3f}]")
                
                # 计算初始准确率
                _, predicted = torch.max(logits, 1)
                initial_acc = (predicted == labels).float().mean()
                print(f"🎯 初始准确率: {initial_acc.item()*100:.2f}%")
                
                break
        
        return initial_acc.item()


# === 安全模式检查器 ===
class SafeModeChecker:
    def __init__(self, memory_limit_gb=18, check_interval=300):
        self.memory_limit_gb = memory_limit_gb
        self.check_interval = check_interval
        self.last_check_time = 0

    def is_safe_to_train(self):
        """安全检查逻辑"""
        current_time = time.time()
        if current_time - self.last_check_time < self.check_interval:
            return True

        if not torch.cuda.is_available():
            return True

        try:
            used_memory = torch.cuda.memory_allocated() / 1024 ** 3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            free_memory = total_memory - used_memory

            if used_memory > self.memory_limit_gb:
                print(f"⚠️ GPU显存超限: {used_memory:.1f}GB > {self.memory_limit_gb}GB限制")
                return False

            safety_margin = 2.0
            if free_memory < safety_margin:
                print(f"⚠️ GPU显存不足: 剩余 {free_memory:.1f}GB < {safety_margin}GB安全边际")
                return False

            self.last_check_time = current_time
            return True

        except Exception as e:
            print(f"❌ 安全检查失败: {e}")
            return True


# === 早停机制 ===
class AdaptiveEarlyStopping:
    def __init__(self, patience=10, delta=0.001, warmup_epochs=5):
        self.patience = patience
        self.delta = delta
        self.warmup_epochs = warmup_epochs
        self.best_acc = 0
        self.counter = 0
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_acc, epoch, model):
        if epoch < self.warmup_epochs:
            return False

        if val_acc > self.best_acc + self.delta:
            self.best_acc = val_acc
            self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print("✅ 已恢复最佳模型权重")
        return self.early_stop


# === 标签平滑 ===
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# === 模型EMA ===
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# === 学习率调度 ===
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# === 设备设置 ===
def setup_device():
    """设置GPU设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)

        print(f"✅ 发现 {gpu_count} 个GPU:")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.1f} GB")

        if gpu_count > 1:
            print(f"🎯 使用多GPU训练: {gpu_count}个GPU")
        else:
            print(f"🎯 使用GPU: {device_name}")

        torch.cuda.empty_cache()
        return device
    else:
        print("⚠️ 未发现GPU，使用CPU")
        return torch.device('cpu')


def set_gpu_memory_limit(limit_gb=12):
    """设置GPU显存限制"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        limit_bytes = int(limit_gb * 1024 ** 3)

        torch.cuda.set_per_process_memory_fraction(limit_bytes / total_memory)
        print(f"✅ 设置显存限制: {limit_gb}GB / {total_memory / 1024 ** 3:.1f}GB")


def setup_multi_gpu(model, device):
    """设置多GPU训练"""
    if torch.cuda.device_count() > 1:
        print(f"🚀 使用 {torch.cuda.device_count()} 个GPU进行数据并行训练")
        model = nn.DataParallel(model)

    model = model.to(device)
    return model


def main():
    print("=" * 60)
    print("🎭 表情识别训练脚本 - 修复版本")
    print("=" * 60)

    # 打印配置
    print("📋 训练配置:")
    print(f"  模型大小: {config.model_size}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  使用预训练: {'是' if config.use_pretrained else '否'}")

    # === 初始化权重加载器 ===
    weight_loader = SmartWeightLoader(
        weights_dir=config.weights_dir,
        use_pretrained=config.use_pretrained,
        force_download=config.force_download
    )

    # === 设备设置 ===
    device = setup_device()

    # === 设置显存限制 ===
    if config.safe_mode and torch.cuda.is_available():
        set_gpu_memory_limit(config.memory_limit)

    # === 数据预处理 ===
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dir = './data/train'
    test_dir = './data/test'

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    num_classes = len(train_dataset.classes)
    print(f"🎯 数据集: {num_classes}个类别")

    # === 创建模型 ===
    print(f"\n🔄 创建模型: {config.model_size}")
    model = weight_loader.create_huggingface_model(config.model_size, num_classes=num_classes)
    model = model.to(device)

    # === 诊断检查 ===
    batch_size = config.batch_size
    num_workers = min(4, os.cpu_count())
    train_loader_diag = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
    
    initial_acc = weight_loader.diagnostic_check(model, train_loader_diag, device)
    
    # 评估初始准确率
    print(f"\n🎯 初始准确率评估:")
    if initial_acc > 0.7:
        print(f"✅ 优秀! 初始准确率: {initial_acc*100:.2f}% (预期范围: 70-85%)")
    elif initial_acc > 0.5:
        print(f"⚠️ 一般! 初始准确率: {initial_acc*100:.2f}% (低于预期)")
    else:
        print(f"❌ 较差! 初始准确率: {initial_acc*100:.2f}% (可能存在配置问题)")

    # === 设置学习率 ===
    learning_rate = config.lr if config.lr > 0 else weight_loader.model_configs[config.model_size]['lr']
    print(f"🎯 学习率: {learning_rate:.2e}")

    # === 优化器 ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    # === 损失函数 ===
    simple_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    train_dataset_simple = datasets.ImageFolder(train_dir, transform=simple_transform)
    train_labels = [label for _, label in train_dataset_simple]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # === 数据加载器 ===
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)

    print(f"📊 数据加载器:")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    print(f"  Batch Size: {batch_size}")
    print(f"  数据加载进程: {num_workers}")

    # === 学习率调度 ===
    num_training_steps = len(train_loader) * config.epochs // config.grad_accum_steps
    num_warmup_steps = len(train_loader) * 5 // config.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # === 初始化训练组件 ===
    ema = ModelEMA(model)
    early_stopping = AdaptiveEarlyStopping(patience=10)
    safe_checker = SafeModeChecker() if config.safe_mode else None
    scaler = GradScaler() if AMP_AVAILABLE else None

    # === 训练函数 ===
    def train_epoch(model, dataloader, criterion, optimizer, scaler, accumulation_steps=1):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(tqdm(dataloader, desc="训练中")):
            # 安全检查
            if safe_checker and not safe_checker.is_safe_to_train():
                time.sleep(10)
                continue

            images, labels = images.to(device), labels.to(device)

            if AMP_AVAILABLE:
                with autocast():
                    outputs = model(pixel_values=images, labels=labels)
                    loss = outputs.loss / accumulation_steps

                scaler.scale(loss).backward()
            else:
                outputs = model(pixel_values=images, labels=labels)
                loss = outputs.loss / accumulation_steps
                loss.backward()

            total_loss += loss.item() * accumulation_steps

            if (i + 1) % accumulation_steps == 0:
                if AMP_AVAILABLE:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                ema.update()

        return total_loss / len(dataloader)

    def evaluate(model, dataloader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="验证中"):
                images, labels = images.to(device), labels.to(device)

                if AMP_AVAILABLE:
                    with autocast():
                        outputs = model(pixel_values=images)
                        logits = outputs.logits
                else:
                    outputs = model(pixel_values=images)
                    logits = outputs.logits

                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    # === 训练循环 ===
    print(f"\n🚀 开始训练")
    best_acc = 0.0
    training_history = {'train_loss': [], 'val_acc': []}

    for epoch in range(config.epochs):
        print(f"\n📊 Epoch [{epoch+1}/{config.epochs}]")

        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, config.grad_accum_steps)

        # 验证
        ema.apply_shadow()
        val_acc = evaluate(model, val_loader)
        ema.restore()

        training_history['train_loss'].append(train_loss)
        training_history['val_acc'].append(val_acc)

        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证准确率: {val_acc*100:.2f}%")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")

        # 更新最佳准确率
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存最佳模型
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch,
                'config': vars(config)
            }, f'best_model_{config.model_size}.pth')
            print(f"🎉 新的最佳准确率: {best_acc*100:.2f}%")

        # 早停检查
        if early_stopping(val_acc, epoch, model):
            print("🛑 早停触发")
            break

    print(f"\n🏁 训练完成! 最终最佳准确率: {best_acc*100:.2f}%")


if __name__ == '__main__':
    main()
