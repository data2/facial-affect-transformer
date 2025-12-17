import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """带热身的余弦退火调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    print("开始进入训练 - 修复版 (目标: 80%准确率)")
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
        'learning_rate': 5e-5,  # 增大学习率
        'weight_decay': 0.05,
        'cutmix_alpha': 1.0,
        'label_smoothing': 0.1,
        'drop_rate': 0.2,
        'grad_accum_steps': 2,  # 减少累积步数
        'warmup_epochs': 10,    # 延长热身期
    }
    
    MODEL_SIZE = config['model_size']
    print(f"🎯 目标: 80%验证准确率 | 模型: ViT-{MODEL_SIZE.capitalize()}")

    # === 1. 修复数据预处理 ===
    print("\n🔄 配置修复版数据增强...")
    
    # 训练时使用的增强transform (修复版)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 先放大
        transforms.RandomCrop((224, 224)),  # 随机裁剪
        transforms.Grayscale(num_output_channels=3),  # 灰度转3通道
        
        # 空间变换
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        
        # 颜色变换
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        
        # 转换为tensor
        transforms.ToTensor(),
        
        # 归一化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # tensor上的变换（必须在ToTensor之后）
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])

    # 验证集transform (简化版)
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

    # 检查数据路径
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

    # 数据加载器
    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print(f"  Batch Size: {config['batch_size']} (累积步数: {config['grad_accum_steps']})")

    # === 3. 计算类别权重 (修复版) ===
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # 适度增强困难类别的权重 (避免过度加权)
    class_weights[0] *= 1.5  # angry (原2.0)
    class_weights[1] *= 2.0  # fear (原3.0)
    class_weights[5] *= 1.2  # sad (原1.5)
    
    print("📈 调整后的类别权重:", class_weights.cpu().numpy())

    # === 4. 修复模型创建 ===
    def create_fixed_model(model_size='base', num_classes=7, weights_dir='./weights'):
        """修复版模型创建函数"""
        model_configs = {
            'base': {
                'name': 'vit_base_patch16_224',
                'local_file': 'vit_base_patch16_224.pth',
            }
        }

        if model_size not in model_configs:
            raise ValueError(f"不支持的模型大小: {model_size}")

        config = model_configs[model_size]
        local_weight_path = os.path.join(weights_dir, config['local_file'])

        if not os.path.exists(local_weight_path):
            print(f"⚠️ 未找到本地权重文件: {local_weight_path}")
            print("🔄 使用随机初始化模型...")
            model = timm.create_model(config['name'], pretrained=False, num_classes=num_classes)
            return model

        print(f"🔄 从本地加载预训练权重: {local_weight_path}")

        # 创建基础模型
        model = timm.create_model(config['name'], pretrained=False, num_classes=num_classes)

        try:
            # 加载权重文件
            checkpoint = torch.load(local_weight_path, map_location='cpu')
            print(f"✅ 权重文件加载成功，大小: {os.path.getsize(local_weight_path) / 1024**3:.2f} GB")

            # 提取状态字典
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            print(f"🔍 权重文件包含的键数量: {len(state_dict)}")

            # 过滤分类头权重
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if not any(key.startswith(prefix) for prefix in ['head.', 'fc.', 'classifier.']):
                    filtered_state_dict[key] = value
                else:
                    print(f"⚠️ 跳过分类头权重: {key}")

            print(f"🔍 过滤后保留的键数量: {len(filtered_state_dict)}")

            # 加载权重
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            
            print("✅ 预训练权重加载成功！")
            print(f"📊 权重加载报告:")
            print(f"  缺失的键: {len(missing_keys)}个")
            print(f"  意外的键: {len(unexpected_keys)}个")

            # 验证权重加载
            total_params = sum(p.numel() for p in model.parameters())
            pretrained_params = total_params - sum(p.numel() for n, p in model.named_parameters() 
                                                 if n in missing_keys)
            pretrained_ratio = pretrained_params / total_params
            print(f"🎯 预训练参数比例: {pretrained_ratio*100:.1f}%")

            if pretrained_ratio < 0.9:
                print("⚠️ 预训练权重加载比例较低，建议检查权重文件")

        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
            print("🔄 使用随机初始化模型...")
            model = timm.create_model(config['name'], pretrained=False, num_classes=num_classes)

        return model

    # 创建模型
    num_classes = 7
    model = create_fixed_model(MODEL_SIZE, num_classes=num_classes)
    model = model.to(device)
    print("✅ 模型创建成功")

    # === 5. 优化器配置 (修复版) ===
    # 使用统一学习率，避免分层学习率导致的问题
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度
    num_training_steps = len(train_loader) * config['num_epochs'] // config['grad_accum_steps']
    num_warmup_steps = len(train_loader) * config['warmup_epochs'] // config['grad_accum_steps']
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # 损失函数 (标签平滑)
    criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])
    criterion.weight = class_weights

    # 早停机制
    early_stopping = EarlyStopping(patience=12, delta=0.002, min_epochs=20)

    print(f"🎯 优化配置:")
    print(f"  学习率: {config['learning_rate']:.1e}")
    print(f"  CutMix Alpha: {config['cutmix_alpha']}")
    print(f"  标签平滑: {config['label_smoothing']}")
    print(f"  热身轮次: {config['warmup_epochs']}")

    # === 6. 训练函数 ===
    def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, grad_accum_steps=2):
        """修复版训练函数"""
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(tqdm(train_loader, desc="训练中")):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # 30%概率使用CutMix (降低概率，避免早期训练不稳定)
            use_cutmix = np.random.rand() < 0.3
            if use_cutmix and i > 10:  # 前10个batch不使用CutMix
                images, targets_a, targets_b, lam = cutmix_data(images, labels, config['cutmix_alpha'])
                outputs = model(images)
                loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

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

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="验证中", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        acc = correct / total
        avg_loss = total_loss / len(dataloader)
        
        # 计算各类别准确率
        class_acc = []
        for i in range(len(val_dataset.classes)):
            class_mask = np.array(all_targets) == i
            if class_mask.sum() > 0:
                class_acc.append((np.array(all_preds)[class_mask] == i).mean())
            else:
                class_acc.append(0.0)
                
        return acc, avg_loss, class_acc

    # === 7. 数据验证 ===
    print("\n🔍 验证数据加载和模型...")
    
    # 验证数据加载
    sample_batch = next(iter(train_loader))
    images, labels = sample_batch
    print(f"✅ 数据加载验证:")
    print(f"  输入形状: {images.shape}")
    print(f"  标签形状: {labels.shape}")
    print(f"  像素范围: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  均值: {images.mean():.3f}, 标准差: {images.std():.3f}")
    
    # 验证模型前向传播
    model.eval()
    with torch.no_grad():
        sample_output = model(images[:2].to(device))  # 只测试2个样本
        print(f"✅ 模型前向验证:")
        print(f"  输出形状: {sample_output.shape}")
        print(f"  输出范围: [{sample_output.min():.3f}, {sample_output.max():.3f}]")

    # === 8. 训练循环 ===
    print(f"\n🚀 开始修复版训练")
    print("=" * 60)

    best_acc = 0.0
    training_history = {
        'train_loss': [], 'val_acc': [], 'val_loss': [], 'learning_rates': []
    }

    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        print(f"\n📊 Epoch [{epoch+1}/{config['num_epochs']}]")

        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, 
                                device, config['grad_accum_steps'])
        training_history['train_loss'].append(train_loss)

        # 验证
        val_acc, val_loss, class_acc = evaluate(model, val_loader, criterion)
        training_history['val_acc'].append(val_acc)
        training_history['val_loss'].append(val_loss)
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
            print(f"  {cls_name}: {class_acc[i]*100:5.1f}%")

        # 保存最佳模型
        if val_acc > best_acc + 0.001:
            best_acc = val_acc
            best_model_path = f'best_vit_{MODEL_SIZE}_fixed.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'config': config
            }, best_model_path)
            print(f"🎉 新的最佳准确率: {best_acc*100:.2f}% -> {best_model_path}")

        # 早停检查
        if early_stopping(val_acc, epoch):
            print("🛑 早停触发，训练结束")
            break

        print("-" * 60)

        # 每5个epoch保存一次checkpoint
        if (epoch + 1) % 5 == 0:
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

    # === 9. 训练总结 ===
    print("\n" + "=" * 60)
    print("🎯 训练总结")
    print("=" * 60)
    print(f"📊 最终最佳准确率: {best_acc*100:.2f}%")
    print(f"💾 最佳模型: best_vit_{MODEL_SIZE}_fixed.pth")
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
        plt.savefig('training_curves_fixed.png', dpi=300, bbox_inches='tight')
        print("📈 训练曲线已保存: training_curves_fixed.png")
    except:
        print("⚠️ 无法绘制训练曲线，请安装matplotlib")

    # === 10. 最终验证 ===
    print("\n🔍 最终模型验证...")
    try:
        checkpoint = torch.load(f'best_vit_{MODEL_SIZE}_fixed.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        final_acc, final_loss, class_acc = evaluate(model, val_loader, criterion)
        
        print(f"✅ 最终验证准确率: {final_acc*100:.2f}%")
        if final_acc >= 0.8:
            print("🎉 恭喜！达到80%准确率目标！")
        elif final_acc >= 0.7:
            print("✅ 良好！准确率 > 70%")
        elif final_acc >= 0.6:
            print("📈 中等！准确率 > 60%")
        else:
            print("⚠️ 需要进一步优化")
    except Exception as e:
        print(f"❌ 最终验证失败: {e}")

    # === 11. 示例预测 ===
    def predict_single_image(image_path, model, transform, class_names):
        """单图预测函数"""
        model.eval()
        try:
            img = Image.open(image_path).convert('L')  # 转为灰度
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)

            label = class_names[pred.item()]
            confidence = conf.item()

            print(f"\n🎯 单图预测结果:")
            print(f"  图片: {os.path.basename(image_path)}")
            print(f"  预测类别: {label}")
            print(f"  置信度: {confidence:.4f}")
            
            # 打印所有类别概率
            print(f"  所有类别概率:")
            for i, cls_name in enumerate(class_names):
                print(f"    {cls_name}: {probs[0][i].item():.4f}")
            
            return label, confidence
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None, None

    # 示例预测
    test_image_path = 'data/test/angry/PrivateTest_3309033.jpg'
    if os.path.exists(test_image_path):
        print("\n" + "=" * 50)
        print("🔍 示例预测")
        print("=" * 50)
        predict_single_image(test_image_path, model, val_transform, val_dataset.classes)
    else:
        # 尝试找到任意测试图片
        for emotion in val_dataset.classes:
            test_dir = os.path.join('data/test', emotion)
            if os.path.exists(test_dir):
                image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
                if image_files:
                    test_image_path = os.path.join(test_dir, image_files[0])
                    print("\n" + "=" * 50)
                    print("🔍 示例预测")
                    print("=" * 50)
                    predict_single_image(test_image_path, model, val_transform, val_dataset.classes)
                    break


if __name__ == '__main__':
    main()