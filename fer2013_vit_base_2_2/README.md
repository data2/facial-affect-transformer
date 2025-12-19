我看到了问题！您的模型在72-73%之间震荡，学习率太高是主要原因。让我为您提供精确的微调策略，不修改模型结构和业务逻辑：

📊 问题分析

轮次 验证准确率 学习率 问题

25 72.25% 1.00e-5 最佳

26-33 71-72% 1.25e-5 - 3.00e-5 震荡

34-40 70-72% 3.25e-5 - 4.61e-5 太高了

核心问题：学习率超过2e-5后，准确率开始下降

🎯 立即微调方案

在train()函数中添加学习率控制策略：
# ==================== Main Training Function ====================
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
    if config.start_epoch > 20:  # 从20轮之后继续训练，降低增强强度
        config.cutmix_prob = 0.3
        config.mixup_prob = 0.15
        print(f"  🎨 数据增强强度调整: CutMix({original_cutmix}→{config.cutmix_prob}), "
              f"MixUp({original_mixup}→{config.mixup_prob})")
    
    # 4. 降低学习率峰值（关键调整）
    print(f"  ⚡ 学习率峰值限制: 5.0e-5 → 2.5e-5")
    # 这个会在调度器中控制
    # ==================== 微调结束 ====================
    
    # 剩下的代码保持不变...


🔧 在训练循环中添加学习率控制

找到训练循环中的scheduler.step()调用位置，添加学习率限制：
# 在 train_epoch 函数中（约第480行）：
if (batch_idx + 1) % config.grad_accum_steps == 0:
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # 学习率限制：不超过2.5e-5
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 2.5e-5:
            param_group['lr'] = 2.5e-5
    
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


🎯 更有效的微调：调整优化器参数

在train()函数中创建优化器后，添加：
# 创建 optimizer 后（约第670行）：
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)

# 添加：调整优化器参数
print("🔧 调整优化器参数:")
print(f"  💡 学习率: {config.learning_rate:.2e}")
print(f"  ⚖️  weight_decay: 0.05 → 0.01")
optimizer.param_groups[0]['weight_decay'] = 0.01  # 降低权重衰减


📈 优化动态权重调整策略

修改DynamicWeightAdjuster.update_weights方法，增强对fear和sad的关注：
# 在 update_weights 方法中（约第300行）：
# Adjustment strategy
for i in range(len(class_accuracies)):
    current_acc = class_accuracies[i]
    recent_avg = recent_acc[i] if i < len(recent_acc) else 0
    
    # 对fear(索引2)和sad(索引5)特别关注
    if i == 2 or i == 5:  # fear 和 sad
        if current_acc < 0.6:  # 低于60%
            if (current_acc - recent_avg) < -0.02:  # 下降
                new_weights[i] *= 2.2  # 大幅增加
            elif abs(current_acc - recent_avg) < 0.015:  # 停滞
                new_weights[i] *= 1.7  # 中等增加
    
    # 对其他类别的调整保持不变
    elif current_acc < 0.5 and (current_acc - recent_avg) < -0.05:
        new_weights[i] *= 2.0
    
    elif 0.5 <= current_acc < 0.7 and abs(current_acc - recent_avg) < 0.02:
        new_weights[i] *= 1.5
    
    elif current_acc > 0.8:
        new_weights[i] *= 0.8


🚀 立即运行优化后的训练

从最佳模型重新开始：
# 从第25轮（最佳72.25%）重新开始
python3 t.py --resume best_model_80_target.pth --epoch 25


📊 预期改善

策略 预期效果 理由

学习率限制2.5e-5 减少震荡 防止过冲

降低weight_decay 提高泛化 减少正则化强度

增强fear/sad权重 提升弱类 fear<60%, sad<65%

延长warmup 稳定训练 更平缓学习率增长

🎯 目标：3-5轮内突破73%

当前最佳72.25%，通过以上微调：

1. 第26轮：稳定在72.5%
2. 第27轮：目标72.8%
3. 第28轮：突破73%
4. 第29-30轮：稳定在73%

💡 关键观察

您的模型已经有很好的基础：
- happy: 90%+ ✓
- surprise: 85%+ ✓
- neutral: 70%+ ✓
- angry: 70%+ ✓
- disgust: 65%+ ✓
- fear: 55% 需要提升
- sad: 60% 需要提升

专注于fear和sad，就能突破73%！ 🎯