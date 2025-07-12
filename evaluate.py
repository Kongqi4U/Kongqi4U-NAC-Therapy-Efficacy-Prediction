#!/usr/bin/env python3
"""
模型评估脚本
对训练好的MAFM-AttU_Net模型进行全面评估
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sns

from model.Models import AttU_Net_with_MAFM
from utils.dataset import ISBI_Loader
from utils.utils_metrics import *
from torch.utils.data import DataLoader

class ModelEvaluator:
    def __init__(self, model_path, data_path, device='cuda'):
        self.model_path = model_path
        self.data_path = data_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = AttU_Net_with_MAFM(img_ch=1, output_ch=1)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 模型已加载: {model_path}")
        print(f"✅ 使用设备: {self.device}")
    
    def evaluate_dataset(self, batch_size=16):
        """在整个数据集上评估模型"""
        # 加载测试数据
        dataset = ISBI_Loader(self.data_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        all_dice_scores = []
        all_iou_scores = []
        all_pixel_acc = []
        all_precision = []
        all_recall = []
        
        print("🔍 开始评估模型性能...")
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 模型预测
                outputs = self.model(images)
                outputs = torch.sigmoid(outputs)
                
                # 转换为二值图像
                preds = (outputs > 0.5).float()
                
                # 计算指标
                for j in range(images.size(0)):
                    pred = preds[j].cpu().numpy()
                    label = labels[j].cpu().numpy()
                    
                    # 计算各种指标
                    dice = calculate_dice_score(pred, label)
                    iou = calculate_iou(pred, label)
                    pixel_acc = calculate_pixel_accuracy(pred, label)
                    precision = calculate_precision(pred, label)
                    recall = calculate_recall(pred, label)
                    
                    all_dice_scores.append(dice)
                    all_iou_scores.append(iou)
                    all_pixel_acc.append(pixel_acc)
                    all_precision.append(precision)
                    all_recall.append(recall)
        
        # 计算平均指标
        results = {
            'dice_score': np.mean(all_dice_scores),
            'iou': np.mean(all_iou_scores),
            'pixel_accuracy': np.mean(all_pixel_acc),
            'precision': np.mean(all_precision),
            'recall': np.mean(all_recall),
            'std_dice': np.std(all_dice_scores),
            'std_iou': np.std(all_iou_scores)
        }
        
        return results, all_dice_scores, all_iou_scores
    
    def evaluate_single_image(self, image_path, save_result=True):
        """评估单张图像"""
        # 加载图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"❌ 无法加载图像: {image_path}")
            return
        
        # 预处理
        image = cv2.resize(image, (256, 256))
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)
            output = torch.sigmoid(output)
            pred = (output > 0.5).float()
        
        # 转换为numpy
        pred_np = pred.cpu().numpy().squeeze()
        
        if save_result:
            # 保存结果
            result_dir = "results/single_predictions"
            os.makedirs(result_dir, exist_ok=True)
            
            filename = os.path.basename(image_path)
            result_path = os.path.join(result_dir, f"pred_{filename}")
            
            # 保存预测结果
            pred_image = (pred_np * 255).astype(np.uint8)
            cv2.imwrite(result_path, pred_image)
            
            print(f"✅ 预测结果已保存: {result_path}")
        
        return pred_np
    
    def visualize_predictions(self, num_samples=6):
        """可视化预测结果"""
        dataset = ISBI_Loader(self.data_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))
        fig.suptitle('MAFM-AttU_Net 预测结果可视化', fontsize=16)
        
        count = 0
        with torch.no_grad():
            for images, labels in dataloader:
                if count >= num_samples:
                    break
                
                images = images.to(self.device)
                outputs = self.model(images)
                outputs = torch.sigmoid(outputs)
                preds = (outputs > 0.5).float()
                
                # 转换为numpy
                image = images[0].cpu().numpy().squeeze()
                label = labels[0].cpu().numpy().squeeze()
                pred = preds[0].cpu().numpy().squeeze()
                
                # 绘制图像
                axes[0, count].imshow(image, cmap='gray')
                axes[0, count].set_title('原始图像')
                axes[0, count].axis('off')
                
                axes[1, count].imshow(label, cmap='gray')
                axes[1, count].set_title('真实标签')
                axes[1, count].axis('off')
                
                axes[2, count].imshow(pred, cmap='gray')
                axes[2, count].set_title('预测结果')
                axes[2, count].axis('off')
                
                count += 1
        
        plt.tight_layout()
        
        # 保存可视化结果
        vis_path = "results/visualization.png"
        os.makedirs("results", exist_ok=True)
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        print(f"✅ 可视化结果已保存: {vis_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, results):
        """生成评估报告"""
        report = f"""
# MAFM-AttU_Net 模型评估报告

## 模型信息
- 模型路径: {self.model_path}
- 数据集路径: {self.data_path}
- 评估设备: {self.device}

## 性能指标

### 主要指标
- **Dice系数**: {results['dice_score']:.4f} ± {results['std_dice']:.4f}
- **IoU**: {results['iou']:.4f} ± {results['std_iou']:.4f}
- **像素精度**: {results['pixel_accuracy']:.4f}
- **精确率**: {results['precision']:.4f}
- **召回率**: {results['recall']:.4f}

### 性能解读
- Dice系数 > 0.7: {'✅ 优秀' if results['dice_score'] > 0.7 else '⚠️ 需要改进'}
- IoU > 0.5: {'✅ 良好' if results['iou'] > 0.5 else '⚠️ 需要改进'}
- 精确率与召回率平衡: {'✅ 平衡' if abs(results['precision'] - results['recall']) < 0.1 else '⚠️ 不平衡'}

## 建议
"""
        
        if results['dice_score'] > 0.7:
            report += "- 模型表现优秀，可以用于实际应用\n"
        else:
            report += "- 建议继续训练或调整模型参数\n"
            
        if results['std_dice'] > 0.1:
            report += "- Dice系数标准差较大，建议增加训练数据或数据增强\n"
        
        # 保存报告
        report_path = "results/evaluation_report.md"
        os.makedirs("results", exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 评估报告已保存: {report_path}")
        print(report)

def main():
    parser = argparse.ArgumentParser(description="MAFM-AttU_Net 模型评估工具")
    parser.add_argument('--model-path', required=True, help='模型权重文件路径')
    parser.add_argument('--data-path', required=True, help='测试数据集路径')
    parser.add_argument('--single-image', help='单张图像评估路径')
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--device', default='cuda', help='计算设备')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"❌ 模型文件不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.data_path):
        print(f"❌ 数据集路径不存在: {args.data_path}")
        return
    
    # 创建评估器
    evaluator = ModelEvaluator(args.model_path, args.data_path, args.device)
    
    if args.single_image:
        # 单张图像评估
        print(f"🔍 评估单张图像: {args.single_image}")
        evaluator.evaluate_single_image(args.single_image)
    else:
        # 数据集评估
        print("🔍 开始数据集评估...")
        results, dice_scores, iou_scores = evaluator.evaluate_dataset(args.batch_size)
        
        # 生成报告
        evaluator.generate_evaluation_report(results)
        
        # 可视化
        if args.visualize:
            print("🎨 生成可视化结果...")
            evaluator.visualize_predictions()

if __name__ == "__main__":
    main()