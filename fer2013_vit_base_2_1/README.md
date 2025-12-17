我看到了！您的模型在第22-27轮训练中，准确率在71-72%之间徘徊，进展缓慢。让我分析并提供微调建议：

📊 训练进展分析

轮次 验证准确率 学习率 关键观察

22 71.65% 4.16e-6 新最佳

23 71.86% 8.32e-6 小提升

24 71.97% 1.25e-5 新最佳

25 71.44% 1.67e-5 下降

26 71.82% 2.08e-5 恢复

27 71.44% 2.50e-5 再次下降

🔍 问题分析

1. 学习率震荡：71.44% ↔ 71.97%波动
2. 训练/验证差距大：训练准确率~40%，验证~72%（差距32%）
3. fear、sad类别表现差：<60%
4. 学习率可能太高：2.5e-5可能偏大

🎯 微调策略（不修改模型逻辑）

1. 调整学习率调度

在train()函数中调整warmup阶段：
# 在train()函数中找到这里（约第660行）：
num_warmup_steps = len(train_loader) * config.warmup_epochs // config.grad_accum_steps

# 改为延长warmup阶段，让学习率增长更平缓
num_warmup_steps = len(train_loader) * 20 // config.grad_accum_steps  # 从12增加到20轮warmup
print(f"🔧 调整warmup阶段: 12 → 20轮")


2. 降低学习率上限

# 在train()函数开头添加（约第400行后）：
if config.learning_rate > 1.5e-5:
    config.learning_rate = 1.5e-5
    print(f"🔧 微调学习率上限: 2e-5 → 1.5e-5")


3. 调整数据增强强度

# 在AdvancedAugmentation.apply_augmentation中（约第150行）：
# 调整增强强度逻辑
if epoch < 30:  # 延长前期增强强度
    cutmix_prob = min(0.6, cutmix_prob + 0.1)  # 稍微增强
    mixup_prob = min(0.3, mixup_prob + 0.1)    # 稍微增强
elif epoch > 50:  # 后期降低增强
    cutmix_prob = max(0.2, cutmix_prob - 0.2)
    mixup_prob = max(0.1, mixup_prob - 0.1)


4. 优化动态权重调整策略

# 在DynamicWeightAdjuster.update_weights中（约第300行）：
# 增强对表现差类别的关注
if current_acc < 0.6:  # 低于60%的类别
    if (current_acc - recent_avg) < -0.03:  # 明显下降
        new_weights[i] *= 2.5  # 大幅增加权重
    elif abs(current_acc - recent_avg) < 0.01:  # 停滞
        new_weights[i] *= 1.8  # 中等增加
elif current_acc < 0.7:  # 60-70%的类别
    if abs(current_acc - recent_avg) < 0.02:  # 停滞
        new_weights[i] *= 1.3  # 小幅增加


🚀 立即应用微调

这里是一个最小修改方案，只修改几个关键参数：
# 在train()函数中添加以下微调代码：

def train(config):
    """Main training function"""
    print(config)
    print("\n🚀 开始冲刺80%准确率训练!")
    print("=" * 60)
    
    # ==================== 微调策略 ====================
    print("🔧 应用微调策略:")
    
    # 1. 降低学习率上限
    if config.learning_rate > 1.5e-5:
        original_lr = config.learning_rate
        config.learning_rate = 1.5e-5
        print(f"  📉 学习率上限: {original_lr} → {config.learning_rate}")
    
    # 2. 延长warmup阶段
    original_warmup = config.warmup_epochs
    if config.warmup_epochs < 20:
        config.warmup_epochs = 20
        print(f"  🔄 warmup轮数: {original_warmup} → {config.warmup_epochs}")
    
    # 3. 调整数据增强强度
    original_cutmix = config.cutmix_prob
    original_mixup = config.mixup_prob
    if epoch > 20:  # 20轮后适当降低增强强度
        config.cutmix_prob = 0.3
        config.mixup_prob = 0.15
        print(f"  🎨 数据增强强度调整: CutMix({original_cutmix}→{config.cutmix_prob}), "
              f"MixUp({original_mixup}→{config.mixup_prob})")
    # ==================== 微调结束 ====================
    
    # 剩下的代码保持不变...
    # Create data augmentation
    data_aug = AdvancedAugmentation(config)
    
    # Load datasets
    print("\n📁 加载数据集...")
    # ... 其余代码不变


📈 各类别优化重点

类别 当前准确率 目标 策略

fear ~55% 65%+ 增加权重，更多关注

sad ~58% 65%+ 增加权重

disgust ~67% 70%+ 保持权重

angry ~68% 75%+ 微调

neutral ~70% 75%+ 保持

surprise ~82% 85%+ 保持

happy ~88% 90%+ 已很好

🔧 从当前检查点继续的优化

您有checkpoint_epoch_020.pth，建议：

1. 从最佳模型继续：best_model_80_target.pth（71.97%）
2. 应用微调参数：学习率1.5e-5，warmup 20轮
3. 继续训练：目标突破73%
# 从最佳模型继续，应用微调
python train.py --resume best_model_80_target.pth --epoch 24


🎯 预期进展

轮次 预期准确率 策略

28-30 72-73% 微调学习率

31-35 73-74% 优化权重调整

36-40 74-75% 进一步微调

您已经取得了71.97%的好成绩！ 通过以上微调，有望在10-15轮内突破73-74%。继续加油！ 💪