# 数据集说明

## BUSI数据集结构

请按照以下结构组织您的BUSI乳腺超声数据集：

```
data/
└── BUSI/
    ├── images/
    │   ├── benign/               # 良性病例图像
    │   │   ├── benign (1).png
    │   │   ├── benign (2).png
    │   │   └── ...
    │   ├── malignant/           # 恶性病例图像
    │   │   ├── malignant (1).png
    │   │   ├── malignant (2).png
    │   │   └── ...
    │   └── normal/              # 正常病例图像
    │       ├── normal (1).png
    │       ├── normal (2).png
    │       └── ...
    └── masks/
        ├── benign/               # 良性病例分割标注
        │   ├── benign (1)_mask.png
        │   ├── benign (2)_mask.png
        │   └── ...
        ├── malignant/           # 恶性病例分割标注
        │   ├── malignant (1)_mask.png
        │   ├── malignant (2)_mask.png
        │   └── ...
        └── normal/              # 正常病例分割标注 (如果有)
            └── ...
```

## 数据集获取

1. **下载地址**: [BUSI数据集 - Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

2. **数据集描述**: 
   - 包含780张乳腺超声图像
   - 分为正常、良性、恶性三类
   - 提供病灶区域的分割标注

3. **使用说明**:
   - 下载数据集后，请按照上述目录结构重新组织文件
   - 确保图像和对应的mask文件名匹配
   - 训练脚本会自动加载这个目录结构的数据

## 数据预处理

项目中的数据加载器会自动进行以下预处理：
- 图像尺寸调整到256x256
- 转换为灰度图像
- 归一化到[0,1]范围
- 数据增强（如果启用）

## 注意事项

- 请确保有足够的磁盘空间存储数据集
- 建议使用SSD以提高数据加载速度
- 如需使用其他数据集，请参考`utils/dataset.py`中的数据加载器实现