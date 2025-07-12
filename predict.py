#!/usr/bin/env python3
"""
单图像预测脚本
使用训练好的MAFM-AttU_Net模型对单张或多张图像进行分割预测
"""

import argparse
import os
import glob
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from model.Models import AttU_Net_with_MAFM

def load_model(model_path, device):
    """加载训练好的模型"""
    model = AttU_Net_with_MAFM(img_ch=1, output_ch=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, target_size=(256, 256)):
    """预处理输入图像"""
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 保存原始尺寸
    original_shape = image.shape
    
    # 调整尺寸
    image_resized = cv2.resize(image, target_size)
    
    # 转换为tensor
    image_tensor = torch.from_numpy(image_resized).float()
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
    image_tensor = image_tensor / 255.0  # 归一化到[0,1]
    
    return image_tensor, original_shape

def postprocess_prediction(prediction, original_shape):
    """后处理预测结果"""
    # 转换为numpy
    pred_np = prediction.cpu().numpy().squeeze()
    
    # 调整回原始尺寸
    pred_resized = cv2.resize(pred_np, (original_shape[1], original_shape[0]))
    
    # 转换为0-255范围的整数
    pred_uint8 = (pred_resized * 255).astype(np.uint8)
    
    return pred_uint8

def predict_single_image(model, image_path, output_path, device, visualize=False):
    """预测单张图像"""
    print(f"正在处理: {image_path}")
    
    # 预处理
    image_tensor, original_shape = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # 预测
    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction)
        prediction_binary = (prediction > 0.5).float()
    
    # 后处理
    result = postprocess_prediction(prediction_binary, original_shape)
    
    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"结果已保存到: {output_path}")
    
    # 可视化
    if visualize:
        visualize_result(image_path, result, output_path)
    
    return result

def visualize_result(image_path, prediction, output_path):
    """可视化预测结果"""
    # 读取原始图像
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    axes[1].imshow(prediction, cmap='gray')
    axes[1].set_title('预测结果')
    axes[1].axis('off')
    
    # 叠加显示
    overlay = cv2.addWeighted(original, 0.7, prediction, 0.3, 0)
    axes[2].imshow(overlay, cmap='gray')
    axes[2].set_title('叠加显示')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存可视化结果
    vis_path = output_path.replace('.png', '_visualization.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存到: {vis_path}")
    
    plt.show()

def predict_batch(model, input_dir, output_dir, device, pattern="*.jpg"):
    """批量预测多张图像"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像路径
    image_paths = glob.glob(os.path.join(input_dir, pattern))
    
    if not image_paths:
        print(f"在 {input_dir} 中没有找到匹配 {pattern} 的图像")
        return
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 批量处理
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_pred.png")
        
        try:
            predict_single_image(model, image_path, output_path, device)
        except Exception as e:
            print(f"处理 {image_path} 时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="MAFM-AttU_Net 图像分割预测工具")
    parser.add_argument('--model', required=True, help='模型权重文件路径')
    parser.add_argument('--input', required=True, help='输入图像路径或目录')
    parser.add_argument('--output', help='输出路径或目录')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    parser.add_argument('--pattern', default='*.jpg', help='批量模式下的文件匹配模式')
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果')
    parser.add_argument('--device', default='cuda', help='计算设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 检查设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在 {args.model}")
        return
    
    # 加载模型
    print("正在加载模型...")
    try:
        model = load_model(args.model, device)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 检查输入
    if not os.path.exists(args.input):
        print(f"错误: 输入路径不存在 {args.input}")
        return
    
    if args.batch:
        # 批量处理模式
        if not os.path.isdir(args.input):
            print("错误: 批量模式需要输入目录")
            return
        
        output_dir = args.output or f"{args.input}_predictions"
        predict_batch(model, args.input, output_dir, device, args.pattern)
        
    else:
        # 单图像处理模式
        if not os.path.isfile(args.input):
            print("错误: 单图像模式需要输入文件")
            return
        
        if args.output is None:
            name, ext = os.path.splitext(args.input)
            args.output = f"{name}_pred.png"
        
        predict_single_image(model, args.input, args.output, device, args.visualize)

if __name__ == "__main__":
    main()
