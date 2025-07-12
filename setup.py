#!/usr/bin/env python3
"""
MAFM-AttU_Net 项目安装脚本
用于检查环境依赖和数据集准备
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，请使用Python 3.8或更高版本")
        return False
    print(f"✅ Python版本: {sys.version}")
    return True

def check_pytorch():
    """检查PyTorch安装"""
    try:
        import torch
        import torchvision
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ TorchVision版本: {torchvision.__version__}")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA不可用，将使用CPU训练")
        return True
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def check_dependencies():
    """检查其他依赖包"""
    required_packages = [
        'numpy', 'opencv-python', 'matplotlib', 'pandas', 
        'scikit-learn', 'tqdm', 'albumentations'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    return missing_packages

def create_directories():
    """创建必要的目录结构"""
    directories = [
        'data/BUSI/images/benign',
        'data/BUSI/images/malignant', 
        'data/BUSI/images/normal',
        'data/BUSI/masks/benign',
        'data/BUSI/masks/malignant',
        'data/BUSI/masks/normal',
        'checkpoints',
        'results'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")

def download_sample_model():
    """下载示例预训练模型（可选）"""
    model_url = "https://github.com/example/model/releases/download/v1.0/best_model.pth"
    model_path = "checkpoints/best_model.pth"
    
    if not os.path.exists(model_path):
        print("📥 正在下载示例模型...")
        try:
            # 这里可以添加实际的模型下载链接
            print("ℹ️  请手动下载预训练模型到 checkpoints/ 目录")
        except Exception as e:
            print(f"⚠️  模型下载失败: {e}")

def main():
    print("🚀 MAFM-AttU_Net 项目环境检查")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        return
    
    # 检查PyTorch
    if not check_pytorch():
        print("\n📦 安装PyTorch:")
        print("pip install torch torchvision")
        return
    
    # 检查其他依赖
    print("\n📦 检查依赖包:")
    missing = check_dependencies()
    
    if missing:
        print(f"\n❌ 缺少依赖包: {', '.join(missing)}")
        print("请运行以下命令安装:")
        print("pip install -r requirements.txt")
        print("或者:")
        print("conda env create -f environment.yml")
        return
    
    # 创建目录结构
    print("\n📁 创建目录结构:")
    create_directories()
    
    # 下载示例模型
    print("\n📥 检查预训练模型:")
    download_sample_model()
    
    print("\n✅ 环境检查完成！")
    print("\n📋 下一步:")
    print("1. 下载BUSI数据集到 data/BUSI/ 目录")
    print("2. 运行训练: python train.py")
    print("3. 运行测试: python test.py")

if __name__ == "__main__":
    main()