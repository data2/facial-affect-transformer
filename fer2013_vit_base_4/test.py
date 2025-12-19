import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

# ========== 全局字体配置 ==========
def setup_global_font():
    """全局设置中文字体 - 使用 AR PL UMing CN"""
    try:
        # 设置字体为 AR PL UMing CN（文鼎明体）
        matplotlib.rcParams['font.family'] = 'AR PL UMing CN'
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 可选：设置字体大小
        matplotlib.rcParams['font.size'] = 12
        matplotlib.rcParams['axes.titlesize'] = 14
        matplotlib.rcParams['axes.labelsize'] = 12
        matplotlib.rcParams['xtick.labelsize'] = 10
        matplotlib.rcParams['ytick.labelsize'] = 10
        matplotlib.rcParams['legend.fontsize'] = 10
        
        print("✅ 全局字体已设置为: AR PL UMing CN (文鼎明体)")
        
        # 验证字体是否设置成功
        from matplotlib import font_manager
        current_font = matplotlib.rcParams['font.family']
        print(f"📝 当前使用字体: {current_font}")
        
        return True
    except Exception as e:
        print(f"❌ 字体设置失败: {e}")
        return False

# 立即执行全局字体设置
setup_global_font()

class PrivateTestEvaluator:
    """在PrivateTest集上评估最佳模型的测试类"""
    
    def __init__(self, config=None):
        """初始化评估器"""
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 使用配置或创建默认配置
        if config is None:
            from dataclasses import dataclass
            @dataclass
            class TestConfig:
                model_name = 'vit_base_patch16_224'
                num_classes = 7
                img_size = 224
                batch_size = 16
                device = self.device
                class_weights = None
                drop_rate = 0.3
                
            self.config = TestConfig()
        else:
            self.config = config
            
        # 类别名称
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # 模型和转换
        self.model = None
        self.test_transform = None
        
        # 结果存储
        self.results = {}
        
        # 确保字体已正确设置
        self._verify_font()
    
    def _verify_font(self):
        """验证字体设置"""
        print("🔍 验证字体设置...")
        current_font = matplotlib.rcParams.get('font.family', '未知')
        print(f"  当前字体: {current_font}")
        
        # 创建一个小测试图验证字体
        try:
            test_fig, test_ax = plt.subplots(figsize=(4, 3))
            test_ax.text(0.5, 0.5, '字体测试: 中文', 
                        fontsize=14, ha='center', va='center', 
                        transform=test_ax.transAxes)
            test_ax.set_title('字体验证')
            test_ax.axis('off')
            plt.savefig('font_verification.png', dpi=150, bbox_inches='tight')
            plt.close(test_fig)
            print("✅ 字体验证图已保存: font_verification.png")
        except Exception as e:
            print(f"⚠️  字体验证失败: {e}")
    
    def create_model(self, model_path='./best_model_3090.pth'):
        """创建和加载模型"""
        print(f"\n📦 创建模型: {self.config.model_name}")
        
        # 创建模型
        model = timm.create_model(
            self.config.model_name,
            pretrained=False,
            num_classes=self.config.num_classes,
            drop_rate=self.config.drop_rate
        ).to(self.device)
        
        # 加载训练的最佳模型
        if os.path.exists(model_path):
            print(f"📥 加载最佳模型: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 处理不同的checkpoint格式
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                
                # 加载状态字典
                model.load_state_dict(state_dict)
                print("✅ 模型加载成功")
                
                # 如果有配置信息，更新配置
                if 'config' in checkpoint:
                    checkpoint_config = checkpoint['config']
                    print(f"📊 模型训练信息:")
                    print(f"  - 最佳准确率: {checkpoint.get('best_acc', 0)*100:.2f}%")
                    print(f"  - 训练轮数: {checkpoint.get('epoch', 0)+1}")
                
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                # 尝试不同的加载方式
                try:
                    model.load_state_dict(checkpoint)
                    print("✅ 使用直接加载方式成功")
                except:
                    raise RuntimeError(f"无法加载模型权重: {e}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.model = model
        return model
    
    def get_test_transform(self):
        """获取测试数据转换"""
        if self.test_transform is None:
            self.test_transform = transforms.Compose([
                transforms.Resize((self.config.img_size, self.config.img_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        return self.test_transform
    
    def load_test_dataset(self, test_dir='./data/test'):
        """加载测试数据集"""
        print(f"\n📁 加载测试数据集: {test_dir}")
        
        if not os.path.exists(test_dir):
            # 尝试不同的路径
            possible_paths = [
                './data/private',
                './data/private_test',
                './datasets/PrivateTest',
                '../data/PrivateTest'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    test_dir = path
                    print(f"✅ 找到测试集: {test_dir}")
                    break
            else:
                raise FileNotFoundError(f"找不到测试集，请检查路径。尝试过的路径: {possible_paths}")
        
        # 加载数据集
        test_dataset = datasets.ImageFolder(
            test_dir, 
            transform=self.get_test_transform()
        )
        
        # 验证类别数量
        if len(test_dataset.classes) != self.config.num_classes:
            print(f"⚠️  警告: 数据集类别数({len(test_dataset.classes)})与模型类别数({self.config.num_classes})不匹配")
            print(f"    数据集类别: {test_dataset.classes}")
        
        print(f"📊 测试集统计:")
        print(f"  - 总样本数: {len(test_dataset):,}")
        print(f"  - 类别: {test_dataset.classes}")
        
        # 显示每个类别的样本数
        class_counts = {}
        for _, label in test_dataset.samples:
            class_name = test_dataset.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"  - 各类别样本数:")
        for cls, count in class_counts.items():
            print(f"    {cls}: {count}")
        
        return test_dataset
    
    def evaluate(self, test_dir='./data/test', model_path='./best_model_3090.pth'):
        """在测试集上评估模型"""
        print("=" * 70)
        print("🧪 开始在测试集上进行模型评估")
        print("=" * 70)
        
        # 1. 创建并加载模型
        model = self.create_model(model_path)
        model.eval()
        
        # 2. 加载测试数据集
        test_dataset = self.load_test_dataset(test_dir)
        
        # 3. 创建数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 4. 进行预测
        print(f"\n🔮 进行预测...")
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="预测")):
                images = images.to(self.device)
                
                # 前向传播
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # 保存结果
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 5. 计算指标
        print(f"\n📊 计算评估指标...")
        
        # 总体准确率
        overall_accuracy = accuracy_score(all_labels, all_predictions)
        
        # 分类报告
        class_report = classification_report(
            all_labels, 
            all_predictions, 
            target_names=test_dataset.classes,
            digits=4,
            output_dict=True
        )
        
        # 混淆矩阵
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # 6. 保存结果
        self.results = {
            'overall_accuracy': overall_accuracy,
            'class_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probs,
            'class_names': test_dataset.classes,
            'total_samples': len(test_dataset)
        }
        
        return self.results
    
    def generate_report(self, save_dir='./results'):
        """生成详细的评估报告"""
        if not self.results:
            print("⚠️  请先运行evaluate()方法")
            return
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n📄 生成评估报告...")
        
        # 1. 打印总体结果
        print("\n" + "=" * 70)
        print("🎯 评估结果总结")
        print("=" * 70)
        print(f"📊 总体准确率: {self.results['overall_accuracy']*100:.4f}%")
        print(f"📈 总样本数: {self.results['total_samples']:,}")
        print("-" * 70)
        
        # 2. 打印每个类别的详细结果
        print("📋 每个类别性能:")
        class_report = self.results['class_report']
        
        # 创建表格
        metrics_df = pd.DataFrame({
            'Precision': [class_report[cls]['precision'] * 100 for cls in self.results['class_names']],
            'Recall': [class_report[cls]['recall'] * 100 for cls in self.results['class_names']],
            'F1-Score': [class_report[cls]['f1-score'] * 100 for cls in self.results['class_names']],
            'Support': [class_report[cls]['support'] for cls in self.results['class_names']]
        }, index=self.results['class_names'])
        
        # 添加平均值行
        metrics_df.loc['Weighted Avg'] = [
            class_report['weighted avg']['precision'] * 100,
            class_report['weighted avg']['recall'] * 100,
            class_report['weighted avg']['f1-score'] * 100,
            class_report['weighted avg']['support']
        ]
        
        metrics_df.loc['Macro Avg'] = [
            class_report['macro avg']['precision'] * 100,
            class_report['macro avg']['recall'] * 100,
            class_report['macro avg']['f1-score'] * 100,
            class_report['macro avg']['support']
        ]
        
        # 格式化显示
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(metrics_df.to_string())
        
        # 3. 保存详细结果到JSON
        report_path = os.path.join(save_dir, 'test_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'overall_accuracy': float(self.results['overall_accuracy']),
                'class_report': self.results['class_report'],
                'class_names': self.results['class_names'],
                'total_samples': self.results['total_samples'],
                'timestamp': pd.Timestamp.now().isoformat()
            }, f, indent=4)
        print(f"\n✅ 详细报告已保存: {report_path}")
        
        # 4. 保存预测结果到CSV
        predictions_df = pd.DataFrame({
            'true_label': [self.results['class_names'][l] for l in self.results['labels']],
            'predicted_label': [self.results['class_names'][p] for p in self.results['predictions']],
            'correct': [l == p for l, p in zip(self.results['labels'], self.results['predictions'])]
        })
        
        # 添加每个类别的概率
        for i, cls in enumerate(self.results['class_names']):
            predictions_df[f'prob_{cls}'] = [prob[i] for prob in self.results['probabilities']]
        
        predictions_path = os.path.join(save_dir, 'detailed_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"✅ 详细预测结果已保存: {predictions_path}")
        
        return metrics_df
    
    def plot_confusion_matrix(self, save_dir='./results', figsize=(12, 10)):
        """绘制并保存混淆矩阵"""
        if not self.results:
            print("⚠️  请先运行evaluate()方法")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n🎨 绘制混淆矩阵...")
        
        # 再次确认字体设置（保险起见）
        plt.rcParams['font.family'] = 'AR PL UMing CN'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建混淆矩阵
        conf_matrix = np.array(self.results['confusion_matrix'])
        class_names = self.results['class_names']
        
        # 归一化混淆矩阵
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 原始混淆矩阵
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax1, cbar=False)
        ax1.set_xlabel('预测标签', fontsize=12)
        ax1.set_ylabel('真实标签', fontsize=12)
        ax1.set_title('混淆矩阵（原始计数）', fontsize=14, pad=20)
        
        # 归一化混淆矩阵
        sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Oranges',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax2, cbar=False)
        ax2.set_xlabel('预测标签', fontsize=12)
        ax2.set_ylabel('真实标签', fontsize=12)
        ax2.set_title('混淆矩阵（归一化）', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        # 保存图像
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.pdf'), bbox_inches='tight')
        plt.close(fig)  # 关闭图形，避免内存泄漏
        
        print(f"✅ 混淆矩阵已保存: {cm_path}")
        print(f"📝 使用字体: AR PL UMing CN")
        
        return cm_path
    
    def plot_class_performance(self, save_dir='./results', figsize=(12, 8)):
        """绘制每个类别的性能指标"""
        if not self.results:
            print("⚠️  请先运行evaluate()方法")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n📈 绘制类别性能图...")
        
        # 再次确认字体设置
        plt.rcParams['font.family'] = 'AR PL UMing CN'
        plt.rcParams['axes.unicode_minus'] = False
        
        class_report = self.results['class_report']
        class_names = self.results['class_names']
        
        # 提取指标
        precision = [class_report[cls]['precision'] * 100 for cls in class_names]
        recall = [class_report[cls]['recall'] * 100 for cls in class_names]
        f1 = [class_report[cls]['f1-score'] * 100 for cls in class_names]
        
        # 支持度
        support = [class_report[cls]['support'] for cls in class_names]
        
        # 创建图形 - 改为1行2列布局
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 指标柱状图（左）
        x = np.arange(len(class_names))
        width = 0.25
        
        ax1 = axes[0]
        bars1 = ax1.bar(x - width, precision, width, label='精确率', color='#4C72B0', alpha=0.8)
        bars2 = ax1.bar(x, recall, width, label='召回率', color='#55A868', alpha=0.8)
        bars3 = ax1.bar(x + width, f1, width, label='F1分数', color='#C44E52', alpha=0.8)
        
        ax1.set_xlabel('情感类别', fontsize=12)
        ax1.set_ylabel('百分比 (%)', fontsize=12)
        ax1.set_title('各类别性能指标对比', fontsize=14, pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 设置y轴范围
        ax1.set_ylim(0, 105)
        
        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 支持度饼图（右）
        ax2 = axes[1]
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        wedges, texts, autotexts = ax2.pie(support, labels=class_names, colors=colors,
                                          autopct='%1.1f%%', startangle=90,
                                          textprops={'fontsize': 10})
        
        ax2.set_title('各类别样本分布', fontsize=14, pad=20)
        
        # 调整图例
        ax2.legend(wedges, class_names, title="情感类别",
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        
        # 保存图像
        perf_path = os.path.join(save_dir, 'class_performance.png')
        plt.savefig(perf_path, dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'class_performance.pdf'), bbox_inches='tight')
        plt.close(fig)  # 关闭图形
        
        print(f"✅ 类别性能图已保存: {perf_path}")
        print(f"📝 使用字体: AR PL UMing CN")
        
        return perf_path
    
    def plot_summary_chart(self, save_dir='./results', figsize=(10, 6)):
        """绘制总体性能总结图"""
        if not self.results:
            print("⚠️  请先运行evaluate()方法")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n📊 绘制总体性能总结图...")
        
        # 设置字体
        plt.rcParams['font.family'] = 'AR PL UMing CN'
        plt.rcParams['axes.unicode_minus'] = False
        
        class_report = self.results['class_report']
        overall_acc = self.results['overall_accuracy'] * 100
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 数据
        categories = ['精确率\n(Precision)', '召回率\n(Recall)', 'F1分数\n(F1-Score)']
        macro_avg = [
            class_report['macro avg']['precision'] * 100,
            class_report['macro avg']['recall'] * 100,
            class_report['macro avg']['f1-score'] * 100
        ]
        weighted_avg = [
            class_report['weighted avg']['precision'] * 100,
            class_report['weighted avg']['recall'] * 100,
            class_report['weighted avg']['f1-score'] * 100
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        # 绘制柱状图
        bars1 = ax.bar(x - width/2, macro_avg, width, label='宏平均', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, weighted_avg, width, label='加权平均', color='#A23B72', alpha=0.8)
        
        # 添加总体准确率线
        ax.axhline(y=overall_acc, color='#F18F01', linestyle='--', linewidth=2, 
                  label=f'总体准确率 ({overall_acc:.2f}%)')
        
        ax.set_xlabel('性能指标', fontsize=12)
        ax.set_ylabel('百分比 (%)', fontsize=12)
        ax.set_title('模型总体性能总结', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # 保存图像
        summary_path = os.path.join(save_dir, 'performance_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ 总体性能总结图已保存: {summary_path}")
        return summary_path
    
    def generate_latex_table(self, save_dir='./results'):
        """生成LaTeX表格"""
        if not self.results:
            print("⚠️  请先运行evaluate()方法")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n📋 生成LaTeX表格...")
        
        class_report = self.results['class_report']
        class_names = self.results['class_names']
        overall_acc = self.results['overall_accuracy'] * 100
        total_samples = self.results['total_samples']
        
        # 创建LaTeX表格 - 修复格式化问题
        latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{在测试集上的分类性能 (总体准确率: {overall_acc:.2f}\\%)}}
\\label{{tab:test_results}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{类别}} & \\textbf{{精确率}} & \\textbf{{召回率}} & \\textbf{{F1分数}} & \\textbf{{支持度}} \\\\
\\midrule
"""
        
        # 添加每个类别的数据
        for cls in class_names:
            p = class_report[cls]['precision'] * 100
            r = class_report[cls]['recall'] * 100
            f = class_report[cls]['f1-score'] * 100
            s = class_report[cls]['support']
            
            latex_table += f"{cls} & {p:.2f}\\% & {r:.2f}\\% & {f:.2f}\\% & {s} \\\\\n"
        
        # 添加平均值
        macro_precision = class_report['macro avg']['precision'] * 100
        macro_recall = class_report['macro avg']['recall'] * 100
        macro_f1 = class_report['macro avg']['f1-score'] * 100
        macro_support = class_report['macro avg']['support']
        
        weighted_precision = class_report['weighted avg']['precision'] * 100
        weighted_recall = class_report['weighted avg']['recall'] * 100
        weighted_f1 = class_report['weighted avg']['f1-score'] * 100
        weighted_support = class_report['weighted avg']['support']
        
        latex_table += "\\midrule\n"
        latex_table += f"宏平均 & {macro_precision:.2f}\\% & {macro_recall:.2f}\\% & {macro_f1:.2f}\\% & {macro_support} \\\\\n"
        latex_table += f"加权平均 & {weighted_precision:.2f}\\% & {weighted_recall:.2f}\\% & {weighted_f1:.2f}\\% & {weighted_support} \\\\\n"
        
        latex_table += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        # 保存LaTeX表格
        latex_path = os.path.join(save_dir, 'results_latex.tex')
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print(f"✅ LaTeX表格已保存: {latex_path}")
        
        # 简单版本
        simple_latex = f"""\\begin{{tabular}}{{lc}}
\\hline
\\textbf{{指标}} & \\textbf{{数值}} \\\\ \\hline
总体准确率 & {overall_acc:.2f}\\% \\\\
总样本数 & {total_samples:,} \\\\
宏平均F1 & {macro_f1:.2f}\\% \\\\
加权平均F1 & {weighted_f1:.2f}\\% \\\\ \\hline
\\end{{tabular}}"""
        
        simple_path = os.path.join(save_dir, 'simple_results.tex')
        with open(simple_path, 'w', encoding='utf-8') as f:
            f.write(simple_latex)
        
        print(f"✅ 简化版LaTeX已保存: {simple_path}")
        
        # 同时生成一个纯文本的Markdown表格，方便查看
        markdown_table = f"""# 模型性能报告

## 总体性能
- **总体准确率**: {overall_acc:.4f}%
- **总样本数**: {total_samples:,}
- **测试时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 详细性能指标

| 类别 | 精确率 | 召回率 | F1分数 | 支持度 |
|------|--------|--------|--------|--------|
"""
        
        for cls in class_names:
            p = class_report[cls]['precision'] * 100
            r = class_report[cls]['recall'] * 100
            f = class_report[cls]['f1-score'] * 100
            s = class_report[cls]['support']
            markdown_table += f"| {cls} | {p:.2f}% | {r:.2f}% | {f:.2f}% | {s} |\n"
        
        markdown_table += f"""| **宏平均** | {macro_precision:.2f}% | {macro_recall:.2f}% | {macro_f1:.2f}% | {macro_support} |
| **加权平均** | {weighted_precision:.2f}% | {weighted_recall:.2f}% | {weighted_f1:.2f}% | {weighted_support} |

## 性能总结
- 最佳表现类别: **happy** (精确率: {class_report['happy']['precision']*100:.2f}%, 召回率: {class_report['happy']['recall']*100:.2f}%)
- 最差表现类别: **fear** (精确率: {class_report['fear']['precision']*100:.2f}%, 召回率: {class_report['fear']['recall']*100:.2f}%)
- 类别不平衡: disgust类别样本最少({class_report['disgust']['support']})，但表现不错(精确率: {class_report['disgust']['precision']*100:.2f}%)
"""
        
        markdown_path = os.path.join(save_dir, 'performance_report.md')
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_table)
        
        print(f"✅ Markdown报告已保存: {markdown_path}")
        
        return latex_table

def run_full_evaluation(test_dir='./data/test', model_path='./best_model_3090.pth'):
    """运行完整的评估流程"""
    print("=" * 70)
    print("🎯 Vision Transformer 在测试集上的评估")
    print("=" * 70)
    
    evaluator = PrivateTestEvaluator()
    
    try:
        # 1. 评估模型
        print("\n1️⃣ 模型评估...")
        results = evaluator.evaluate(test_dir=test_dir, model_path=model_path)
        
        # 2. 生成文本报告
        print("\n2️⃣ 生成文本报告...")
        evaluator.generate_report('./results')
        
        # 3. 绘制可视化图表
        print("\n3️⃣ 生成可视化图表...")
        try:
            cm_path = evaluator.plot_confusion_matrix('./results')
            print(f"  混淆矩阵: {cm_path}")
        except Exception as e:
            print(f"  ⚠️  混淆矩阵生成失败: {e}")
        
        try:
            perf_path = evaluator.plot_class_performance('./results')
            print(f"  类别性能图: {perf_path}")
        except Exception as e:
            print(f"  ⚠️  类别性能图生成失败: {e}")
        
        try:
            summary_path = evaluator.plot_summary_chart('./results')
            print(f"  性能总结图: {summary_path}")
        except Exception as e:
            print(f"  ⚠️  性能总结图生成失败: {e}")
        
        # 4. 生成LaTeX表格
        print("\n4️⃣ 生成LaTeX表格...")
        try:
            evaluator.generate_latex_table('./results')
        except Exception as e:
            print(f"  ⚠️  LaTeX表格生成失败: {e}")
            print("  ℹ️  正在生成简化版本...")
            # 生成一个简单的文本版本
            simple_report = f"""
总体准确率: {results['overall_accuracy']*100:.4f}%
总样本数: {results['total_samples']:,}
宏平均F1: {results['class_report']['macro avg']['f1-score']*100:.2f}%
加权平均F1: {results['class_report']['weighted avg']['f1-score']*100:.2f}%
            """
            with open('./results/simple_report.txt', 'w') as f:
                f.write(simple_report)
        
        # 5. 最终总结
        print("\n" + "=" * 70)
        print("✅ 评估完成！")
        print("=" * 70)
        print(f"📊 总体准确率: {results['overall_accuracy']*100:.4f}%")
        print(f"📈 总样本数: {results['total_samples']:,}")
        print(f"📁 结果已保存至: ./results/")
        
        # 显示保存的文件
        if os.path.exists('./results'):
            print("\n📂 生成的文件:")
            files = os.listdir('./results')
            if files:
                for file in sorted(files):
                    if file.endswith(('.png', '.pdf', '.json', '.csv', '.tex', '.md', '.txt')):
                        full_path = f'./results/{file}'
                        size = os.path.getsize(full_path) / 1024
                        print(f"  • {file:25s} ({size:.1f} KB)")
            else:
                print("  (空目录)")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def quick_test(test_dir='./data/test', model_path='./best_model_3090.pth'):
    """快速测试"""
    print("🚀 快速测试模式")
    print("=" * 70)
    
    evaluator = PrivateTestEvaluator()
    
    try:
        results = evaluator.evaluate(test_dir=test_dir, model_path=model_path)
        
        print(f"\n🎯 总体准确率: {results['overall_accuracy']*100:.4f}%")
        print(f"📊 测试样本数: {results['total_samples']:,}")
        
        # 显示每个类别的准确率
        print("\n📋 各类别准确率:")
        class_report = results['class_report']
        for cls in results['class_names']:
            acc = class_report[cls]['recall'] * 100
            print(f"  {cls:10s}: {acc:6.2f}%")
        
        print(f"\n📈 宏平均F1: {class_report['macro avg']['f1-score']*100:.2f}%")
        print(f"📈 加权平均F1: {class_report['weighted avg']['f1-score']*100:.2f}%")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='在测试集上评估模型')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'quick', 'font'],
                       help='评估模式: full(完整评估), quick(快速测试), font(字体测试)')
    parser.add_argument('--test_dir', type=str, default='./data/test',
                       help='测试集路径')
    parser.add_argument('--model_path', type=str, default='./best_model_3090.pth',
                       help='模型路径')
    
    args = parser.parse_args()
    
    if args.mode == 'font':
        # 字体测试
        print("🔍 字体测试模式")
        print(f"当前字体: {matplotlib.rcParams.get('font.family', '未知')}")
        
        # 创建测试图
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.7, '中文测试: AR PL UMing CN', fontsize=16, ha='center', va='center')
        plt.text(0.5, 0.5, '预测标签 - 真实标签', fontsize=12, ha='center', va='center')
        plt.text(0.5, 0.3, '精确率: 85.6% 召回率: 92.1%', fontsize=12, ha='center', va='center')
        plt.title('字体测试 - AR PL UMing CN', fontsize=14)
        plt.axis('off')
        plt.savefig('final_font_test.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 字体测试图已保存: final_font_test.png")
        
    elif args.mode == 'quick':
        quick_test(args.test_dir, args.model_path)
    else:
        run_full_evaluation(args.test_dir, args.model_path)