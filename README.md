# 基于超声影像的乳腺癌新辅助化疗疗效预测

## 项目概述

本项目是一个基于深度学习的医学影像分析研究，专注于乳腺超声图像的高精度分割技术。项目的核心贡献是提出了**MAFM-AttU_Net分割模型**，有效解决了超声图像噪声高、病灶边缘模糊的分割挑战。

> **注**: 项目原包含基于内部临床数据的pCR预测研究，但由于医疗数据隐私保护条例限制，相关数据集和代码不包含在此开源版本中。本开源项目专注于分割模型的技术创新部分。

## 🚀 核心特性

### 1. MAFM-AttU_Net 分割模型
- **解决痛点**: 超声图像噪声高、病灶边缘模糊的分割挑战
- **核心创新**: **MAFM**(多尺度感知融合模块) - Multi-scale Awareness Fusion Module
- **技术优势**: 
  - 基于Attention U-Net架构设计
  - 解码器集成MAFM模块，显著增强边缘细节感知能力
  - 在模糊和不规则病灶边缘处理方面表现卓越

### 2. pCR预测应用潜力
- **研究价值**: 分割模型可为下游pCR预测任务提供高质量特征
- **技术优势**: MAFM模块提取的多尺度特征具有强表征能力
- **应用前景**: 为临床疗效预测提供可靠的技术基础

> **数据说明**: pCR预测相关的临床数据因隐私保护要求未包含在开源版本中

## 📊 性能表现

### 分割任务
- **MAFM-AttU_Net**: Dice系数达到 **73.07%**
- **对比基线**: 相比标准AttU_Net(69.5%)提升明显
- **优势领域**: 在处理边缘模糊的病灶区域表现突出

### 特征提取能力
- **特征质量**: MAFM模块提取的深层特征具有强判别性
- **多尺度融合**: 有效整合不同分辨率的图像信息
- **边缘增强**: 在模糊边缘区域提供更丰富的特征表示
- **应用验证**: 在内部pCR预测任务中表现优异(详情见相关研究文献)

## 🏗️ 项目结构

```
项目根目录/
├── model/
│   └── Models.py              # 核心网络架构
│       ├── MAFM               # 多尺度感知融合模块
│       ├── AttU_Net_with_MAFM # 主分割网络
│       └── MHMC, COI          # MAFM核心组件
├── data/                      # 数据目录(需用户自行准备)
│   └── BUSI/                  # BUSI乳腺超声数据集
│       ├── images/            # 超声图像
│       └── masks/             # 分割标注
├── utils/
│   ├── dataset.py             # 分割任务数据处理
│   ├── loss.py                # 损失函数定义
│   └── utils_metrics.py       # 评估指标计算
├── train.py                   # 分割模型训练脚本
├── test.py                   # 模型测试评估
├── predict.py                # 单图像预测脚本
└── environment.yml           # 环境依赖配置
```

## 🔬 技术架构

### MAFM模块设计
- **MHMC**(多头多卷积注意力): 实现多尺度特征感知与融合
- **COI**(通道优化交互): 强化通道间信息交换机制
- **预处理注意力**: 采用深度可分离卷积优化特征表示

### 分割网络特点
- **编码器-解码器架构**: 基于AttU_Net的对称设计
- **注意力机制**: 增强重要特征区域的表示能力
- **多尺度融合**: MAFM模块实现不同层级特征的有效整合
- **边缘优化**: 专门针对超声图像模糊边缘的处理策略

## 🛠️ 环境配置

### 系统要求
- Python 3.8+
- CUDA兼容GPU (推荐配置)
- 内存 8GB+

### 依赖安装
```bash
# 方式1: 使用conda环境文件
conda env create -f environment.yml
conda activate APCGAN_AttuNet

# 方式2: 手动安装核心包
pip install torch torchvision
pip install opencv-python albumentations scikit-learn
pip install pandas numpy matplotlib seaborn
pip install pyradiomics SimpleITK tqdm
```

## 🚀 使用指南

### 主要功能: 图像分割
```bash
# 1. 准备数据
# 下载BUSI数据集并放置到data/BUSI/目录

# 2. 配置参数 (修改 train.py)
data_path = "./data/BUSI"       # 数据集路径
net = AttU_Net_with_MAFM(img_ch=1, output_ch=1)  # 选择模型

# 3. 开始训练
python train.py
```

### 模型评估与预测
```bash
# 分割性能评估
python test.py

# 单张图像预测
python predict.py --input /path/to/image.png --output /path/to/result.png

# 可视化对比(如果有对比数据)
python comparison.py
```

## 📁 数据格式

### BUSI数据集结构
```
data/BUSI/
├── images/
│   ├── benign/               # 良性病例图像
│   ├── malignant/           # 恶性病例图像
│   └── normal/              # 正常病例图像
└── masks/
    ├── benign/               # 良性病例分割标注
    ├── malignant/           # 恶性病例分割标注
    └── normal/              # 正常病例分割标注
```

### 图像要求
- **格式**: PNG/JPG
- **尺寸**: 建议256x256或更高分辨率
- **通道**: 灰度图像(单通道)或彩色图像(会自动转换为灰度)

## ⚙️ 实验配置

### 核心超参数
```python
# 训练配置
BATCH_SIZE = 16               # 批处理大小
EPOCHS = 300                  # 最大训练轮数
LEARNING_RATE = 0.00001       # 初始学习率
VAL_PERCENT = 0.2             # 验证集比例
PATIENCE = 20                 # 早停策略耐心值

# MAFM模块配置
mhmc_ca_num_heads = 4         # 多头注意力头数
mhmc_expand_ratio = 2         # 通道扩展倍数
```

### 评估指标
- **Dice系数**: 分割重叠度量
- **IoU**: 交并比
- **像素精度**: 像素级分类准确率
- **召回率**: 病灶区域召回能力

## 🎯 复现流程

### 完整实验复现
1. **环境准备**: 按照上述步骤配置Python环境
2. **数据准备**: 
   - 下载BUSI乳腺超声数据集
   - 按照上述格式组织数据目录结构
3. **模型训练**: 
   ```bash
   # 训练MAFM-AttU_Net分割模型
   python train.py
   ```
4. **模型评估**: 
   ```bash
   # 在测试集上评估模型性能
   python test.py
   ```
5. **结果分析**: 查看results/目录下生成的评估报告和可视化结果

### 预训练权重
- 分割模型: `checkpoints/best_model.pth`
- 可使用预训练权重进行fine-tuning或直接预测

## 🤝 贡献指南

欢迎研究人员和开发者为项目贡献代码或提出改进建议：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 开源协议

本项目基于MIT协议开源，详见 [LICENSE](LICENSE) 文件。

## 🔗 相关资源

- [Attention U-Net原论文](https://arxiv.org/abs/1804.03999)
- [Pyradiomics库文档](https://pyradiomics.readthedocs.io/)
- [BUSI数据集](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

## ⚠️ 重要说明

### 数据隐私
本开源版本不包含任何患者医疗数据。如需使用临床数据进行研究，请：
- 遵守当地医疗数据保护法规
- 获得必要的伦理委员会批准
- 确保患者隐私和数据安全

### 免责声明
本项目仅供学术研究和技术验证使用，不能替代专业医学诊断。在临床应用前需要经过严格的医学验证和监管审批。

---

**注**: 本项目专注于医学影像分析的创新研究，致力于为乳腺癌诊疗提供智能化辅助工具。