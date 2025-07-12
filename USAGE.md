# MAFM-AttU_Net 使用指南

## 快速开始

### 1. 环境配置

#### 方式一：使用Conda (推荐)
```bash
# 创建环境
conda env create -f environment.yml
conda activate APCGAN_AttuNet
```

#### 方式二：使用pip
```bash
pip install -r requirements.txt
```

#### 方式三：自动检查环境
```bash
python setup.py
```

### 2. 数据准备

#### 下载BUSI数据集
1. 访问 [BUSI数据集](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
2. 下载数据集到本地
3. 按照以下结构组织数据：

```
data/BUSI/
├── images/
│   ├── benign/        # 良性病例
│   ├── malignant/     # 恶性病例
│   └── normal/        # 正常病例
└── masks/
    ├── benign/        # 良性病例标注
    ├── malignant/     # 恶性病例标注
    └── normal/        # 正常病例标注
```

#### 数据格式要求
- 图像格式：PNG, JPG
- 图像尺寸：任意（会自动resize到256x256）
- 标注要求：二值图像（0=背景，255=前景）

### 3. 模型训练

#### 基础训练
```bash
python train.py
```

#### 自定义参数训练
```python
# 修改 train.py 中的参数
EPOCHS = 300           # 训练轮数
BATCH_SIZE = 16        # 批次大小
LEARNING_RATE = 0.00001  # 学习率
VALIDATION_PERCENT = 0.2  # 验证集比例
```

#### 选择不同的模型
```python
# 在 train.py 中选择模型
net = AttU_Net_with_MAFM(img_ch=1, output_ch=1)  # 推荐：MAFM模型
# net = AttU_Net(img_ch=1, output_ch=1)          # 基础注意力U-Net
# net = UNet(n_channels=1, n_classes=1)          # 标准U-Net
```

### 4. 训练监控

#### 实时监控训练过程
```bash
python monitor.py --action watch --interval 30
```

#### 查看训练曲线
```bash
python monitor.py --action plot
```

#### 查看当前指标
```bash
python monitor.py --action metrics
```

#### 生成训练报告
```bash
python monitor.py --action report
```

### 5. 模型评估

#### 全面评估
```bash
python evaluate.py --model-path checkpoints/best_model.pth --data-path data/BUSI
```

#### 评估并生成可视化
```bash
python evaluate.py --model-path checkpoints/best_model.pth --data-path data/BUSI --visualize
```

#### 单张图像预测
```bash
python evaluate.py --model-path checkpoints/best_model.pth --single-image path/to/image.png
```

### 6. 模型测试

#### 标准测试
```bash
python test.py
```

#### 单张图像预测
```bash
python predict.py --input path/to/image.png --output path/to/result.png
```

## 高级用法

### 1. 自定义数据集

如果要使用自己的数据集，需要修改 `utils/dataset.py`：

```python
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # 实现自定义数据加载逻辑
        pass
    
    def __getitem__(self, index):
        # 返回 (image, mask) 元组
        pass
```

### 2. 调整MAFM参数

```python
# 在模型初始化时调整MAFM参数
net = AttU_Net_with_MAFM(
    img_ch=1, 
    output_ch=1,
    mhmc_ca_num_heads=4,     # 多头注意力头数
    mhmc_expand_ratio=2      # 通道扩展比例
)
```

### 3. 损失函数配置

```python
# 在 train.py 中调整损失函数权重
LOSS_WEIGHTS = (0.5, 0.5)  # (BCE权重, Dice权重)
```

### 4. 数据增强

```python
# 在 train.py 中启用数据增强
transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
```

## 常见问题

### 1. 内存不足
- 减小batch_size
- 使用gradient checkpointing
- 减少num_workers

### 2. 训练过慢
- 检查CUDA是否可用
- 增加batch_size（在内存允许的情况下）
- 使用更强的GPU

### 3. 模型不收敛
- 调整学习率（降低或使用学习率调度）
- 检查数据质量
- 增加训练数据
- 调整损失函数权重

### 4. 分割效果差
- 检查数据标注质量
- 尝试不同的预处理方法
- 调整模型参数
- 增加训练轮数

## 性能基准

### BUSI数据集上的性能
- **MAFM-AttU_Net**: Dice系数 73.07%
- **AttU_Net**: Dice系数 69.5%
- **U-Net**: Dice系数 ~65%

### 训练时间参考
- GTX 1080 Ti: ~2小时 (300 epochs)
- RTX 3080: ~1小时 (300 epochs)
- CPU: ~12小时 (300 epochs)

## 文件说明

### 核心文件
- `train.py`: 主训练脚本
- `test.py`: 模型测试脚本
- `predict.py`: 单图像预测脚本
- `evaluate.py`: 模型评估脚本
- `monitor.py`: 训练监控脚本

### 模型文件
- `model/Models.py`: 包含MAFM-AttU_Net等模型定义
- `model/unet_model.py`: 标准U-Net实现
- `model/unet_parts.py`: U-Net组件

### 工具文件
- `utils/dataset.py`: 数据加载器
- `utils/loss.py`: 损失函数定义
- `utils/utils_metrics.py`: 评估指标计算

### 配置文件
- `environment.yml`: Conda环境配置
- `requirements.txt`: pip依赖列表
- `.gitignore`: Git忽略文件配置

## 输出说明

### 训练输出
- `checkpoints/best_model.pth`: 最佳模型权重
- `results/training_log.csv`: 训练日志
- `results/loss_curves.png`: 损失曲线图

### 评估输出
- `results/evaluation_report.md`: 评估报告
- `results/visualization.png`: 预测可视化
- `results/single_predictions/`: 单图像预测结果

## 扩展开发

### 添加新的模型
1. 在 `model/Models.py` 中定义新模型
2. 在 `train.py` 中添加模型选择选项
3. 更新相关文档

### 添加新的损失函数
1. 在 `utils/loss.py` 中定义新损失函数
2. 在 `train.py` 中集成使用
3. 测试并验证效果

### 添加新的评估指标
1. 在 `utils/utils_metrics.py` 中实现新指标
2. 在 `evaluate.py` 中集成计算
3. 更新报告模板

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 创建Pull Request

详见项目README中的贡献指南部分。