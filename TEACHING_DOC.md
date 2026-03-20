# 乳腺癌 pCR 预测项目完全教学文档

> **文档目标**：读完本文档，不用来回翻论文和代码，也能把这个项目的主线、数据、模型、实验与高频追问讲清楚。
>
> **使用边界**：本文档同时参考了毕业论文 `D:\download\毕业论文.pdf` 和当前代码仓 `D:\work\bishe\APCGAN-AttuNet-main`。两者有少量口径不完全一致的地方，文中会单独标出。

---

## 目录

1. [项目背景与核心叙事](#1-项目背景与核心叙事)
2. [数据来源与任务定义](#2-数据来源与任务定义)
3. [整体数据流与两条主线](#3-整体数据流与两条主线)
4. [分割支线 Baseline：Attention U-Net](#4-分割支线-baselineattention-u-net)
5. [核心改进：MAFM 模块](#5-核心改进mafm-模块)
6. [分割模型：MAFM-AttU_Net](#6-分割模型mafm-attu_net)
7. [pCR 预测主线：三条特征路线 + 多种分类头](#7-pcr-预测主线三条特征路线--多种分类头)
8. [三条特征路线的代码实现](#8-三条特征路线的代码实现)
9. [实验结果与结论](#9-实验结果与结论)
10. [论文口径与当前代码的差异](#10-论文口径与当前代码的差异)
11. [关键张量维度速查表](#11-关键张量维度速查表)
12. [项目目录结构与文件职责](#12-项目目录结构与文件职责)

---

## 1. 项目背景与核心叙事

### 1.1 一句话定义

这是一个**基于超声影像预测乳腺癌患者在新辅助化疗后是否达到 pCR** 的项目，目标是为临床个体化治疗提供辅助决策。

### 1.2 项目真正的主线

这个项目的**主任务不是分割本身，而是 pCR 预测**。

- 分割是关键辅助环节，不是唯一主线
- pCR 预测才是最后要回答的核心临床问题
- 面试时要强调：我做的是“分割改进 + 多特征路线对比 + pCR 预测闭环”

### 1.3 为什么这个问题难

超声图像和自然图像相比有几个典型困难：

- 噪声多，伪影明显
- 病灶边界模糊，不规则病灶更难分
- 小样本医学数据难以支撑复杂模型稳定训练
- pCR 是治疗后的病理结局，单张超声图到结局之间存在较长的“语义鸿沟”

### 1.4 项目的总体思路

```
原始超声图像
    │
    ├── 预处理：去除文字/十字标记/无关区域
    │
    ├── 分割支线：
    │     Attention U-Net
    │         + MAFM
    │     目标：更好地刻画病灶边界与病灶区域
    │
    └── 预测支线：
          三条特征路线
          1. Ours：分割模型深层特征
          2. ResNet18：迁移学习特征
          3. Pyradiomics：影像组学特征
              ↓
          多种分类头
          MLP / XGBoost / SVM / RandomForest
              ↓
          比较 preNAC / postNAC / mixed
              ↓
          得到最优 pCR 预测方案
```

### 1.5 面试最稳的核心叙事

> 这个项目要解决的是如何利用乳腺超声图像预测 NAC 后是否达到 pCR。考虑到超声图像噪声多、病灶边界模糊，我没有直接把它当成一个简单二分类来做，而是先做图像预处理和病灶分割改进，在 Attention U-Net 中引入 MAFM 来增强边界表征。然后在预测阶段，我比较了分割特征、Pyradiomics 和 ResNet18 迁移特征三条路线，并结合多种分类头与不同数据模式做系统对比。最终发现，postNAC 模式下 ResNet18 + MLP 的表现最好，5 折交叉验证平均 AUC 达到 0.791。  

---

## 2. 数据来源与任务定义

### 2.1 两类数据来源

这个项目实际上用了两套数据：

- **BUSI 公共数据集**
  - 用于分割模型训练与对比
  - 目标是验证 MAFM-AttU_Net 的分割能力
- **内部临床超声数据**
  - 用于 pCR 预测
  - 毕业论文稳定口径为 **143 例 NAC 临床数据**

### 2.2 pCR 的定义

`pCR` 指 **pathological complete response**，也就是病理完全缓解。

论文口径中采用的是：

- `ypT0/is ypN0`

也就是经过 NAC 后，术后病理中乳腺原发灶没有浸润性残留，淋巴结也没有转移残留。

### 2.3 临床数据的组织方式

当前代码仓中的临床数据位于：

- `US_filtered/`

其中典型结构是：

```text
US_filtered/
├── clinical_filtered.csv
├── fs0004/
│   ├── preNAC/
│   │   └── us001.png
│   └── postNAC/
│       └── us001.png
├── fs0008/
│   └── ...
```

`clinical_filtered.csv` 中当前可见的关键字段有：

- `Patient ID`
- `ER_status`
- `PR_status`
- `HER2_status`
- `Ki67`
- `Ki67_status`
- `Molecular_Subtype`
- `preNAC`
- `postNAC`
- `pCR`

### 2.4 本地代码层面的实际统计

按当前仓库里的 `clinical_filtered.csv` 统计：

- 总行数：`174`
- `pCR` 非空样本：`161`
- 其中 `pCR=0` 有 `65` 例
- `pCR=1` 有 `96` 例

但当前 `utils/pCR_Dataset.py` 只读取每位患者的固定图像 `us001.png`，所以真正能被代码直接加载的样本数取决于本地文件是否齐全。按当前工作区扫描结果：

- `pre` 模式可加载：`108`
- `post` 模式可加载：`94`
- `mixed` 模式可加载：`52`

这和论文中“143 例临床数据”的最终统计并不完全一致，原因在第 10 节单独说明。

### 2.5 三种数据模式

在 pCR 预测阶段，代码和论文都采用三种数据模式：

- `preNAC`
  - 只使用治疗前超声图像
- `postNAC`
  - 只使用治疗后超声图像
- `mixed`
  - 同时融合 preNAC 与 postNAC 两个时点的信息

### 2.6 为什么 `postNAC` 常常更强

这是一个高频追问点。

最稳的解释是：

- `postNAC` 更接近最终病理结局
- pCR 本身定义的是**治疗后的病理反应**
- 因此治疗后超声中的纹理变化、病灶残余、形态变化与 pCR 的语义更接近
- `preNAC` 信息更早、更间接，预测难度更高
- `mixed` 理论上信息更多，但在小样本条件下也更容易引入噪声和过拟合

---

## 3. 整体数据流与两条主线

### 3.1 从输入到输出的完整闭环

```
输入：乳腺超声图像
    │
    ├── 图像预处理
    │     去掉文字、十字标记、扫描区域外干扰
    │
    ├── 分割支线
    │     BUSI 数据集
    │     → Attention U-Net baseline
    │     → 在解码器中引入 MAFM
    │     → 输出病灶 mask
    │     → 用 Dice / Precision / Recall / mIoU / ACC 评价
    │
    └── 预测支线
          临床超声数据
          → 按 pre / post / mixed 组织
          → 提取三类特征
              1. Ours：分割网络特征
              2. ResNet18：预训练迁移特征
              3. Pyradiomics：影像组学特征
          → 输入分类头
              MLP / XGBoost / SVM / RF
          → 5 折交叉验证
          → 用 AUC / ACC / 混淆矩阵等评价
          → 选择最优方案
```

### 3.2 为什么先做分割而不是直接分类

这个问题不能答成“如果分割不准，后面的预测就没有意义”，那样太绝对。

更稳的说法是：

- 超声图像噪声大，病灶边界模糊
- 直接做分类时，模型可能会学到文字标记、扫描伪影等无关信息
- 分割可以作为**关键辅助环节**
  - 让模型更聚焦病灶区域
  - 增强病灶边界与形态信息表达
  - 提供一条更具可解释性的特征路线
- 但它不是唯一有效路线，因此我后面又和 ResNet18、Pyradiomics 做了系统对比

### 3.3 项目贡献边界

面试时要明确：

- 这个项目的强项不是提出全新的理论
- 而是围绕具体医学问题做了：
  - 数据处理
  - 分割改进
  - 多特征路线对比
  - 多分类头对比
  - 最终实验闭环

---

## 4. 分割支线 Baseline：Attention U-Net

### 4.1 Attention U-Net 为什么适合这个任务

Attention U-Net 本质上是在 U-Net 的跳跃连接上加入了注意力门控。

它的作用是：

- 编码器提取多尺度语义特征
- 解码器逐步恢复空间分辨率
- 注意力门控在 skip connection 上抑制无关区域，强调病灶相关区域

对于乳腺超声这种背景复杂、边界模糊的场景，它比普通 U-Net 更容易聚焦病灶。

### 4.2 当前实现的主干结构

在 `model/Models.py` 中，`AttU_Net_with_MAFM` 保留了 Attention U-Net 的经典五层结构：

```text
Encoder:
Conv1:   1  -> 64
Conv2:   64 -> 128
Conv3:   128 -> 256
Conv4:   256 -> 512
Conv5:   512 -> 1024

Decoder:
Up5: 1024 -> 512
Up4: 512  -> 256
Up3: 256  -> 128
Up2: 128  -> 64
```

### 4.3 传统 Attention U-Net 的短板

虽然 Attention U-Net 已经能抑制部分背景噪声，但对于超声病灶仍然有两个典型难点：

- 模糊边界附近的细粒度纹理容易丢失
- 单一尺度的局部卷积不够容易同时覆盖细小边缘和更大范围上下文

这就是引入 MAFM 的动机。

---

## 5. 核心改进：MAFM 模块

### 5.1 MAFM 的目标

`MAFM` 是 **Multi-scale Awareness Fusion Module**，核心目标是：

- 在解码阶段更好融合上采样特征与注意力筛过的跳连特征
- 同时增强多尺度信息建模能力
- 更关注病灶边缘、纹理和形态细节

### 5.2 MAFM 在代码里的组成

当前 `model/Models.py` 中，`MAFM` 主要由三部分组成：

1. `pre_att`
   - 先把上采样特征 `x` 与 skip 特征 `d` 拼接
   - 用 depthwise 3x3 + pointwise 1x1 做初步融合

2. `MHMC`
   - Multi-Head Multi-scale Convolution
   - 不同头使用不同感受野大小的 depthwise 卷积
   - 让同一层特征同时看到多尺度局部模式

3. `COI`
   - 由 depthwise conv、1x1 conv 和 shortcut 组成
   - 类似一个轻量卷积增强块，用于细化融合后的特征

### 5.3 可以怎么理解 MAFM

一句话理解：

> 先把解码特征和跳连特征融合，再用多头多尺度卷积去看不同范围的局部模式，最后用轻量卷积把边界和纹理细节再强化一遍。

### 5.4 `COI` 做了什么

`COI` 在代码里是一个轻量残差式卷积块：

- 一条 shortcut
- 一条 depthwise conv
- 一条 1x1 conv
- 三路相加后过 `GELU`

作用可以理解为：

- 既保留原始信息
- 又补充局部空间卷积信息
- 同时做通道混合

### 5.5 `MHMC` 做了什么

`MHMC` 的核心思路是把通道分成多个头，每个头使用不同大小的局部卷积核。

当前代码中：

- `ca_num_heads=4`
- 第 1 到第 4 个头的局部卷积核大小分别是：
  - `3x3`
  - `5x5`
  - `7x7`
  - `9x9`

因此它不是标准 Transformer 注意力，而是更接近：

- 多头
- 多尺度
- 卷积式局部建模

这样做的好处是：

- 小卷积核看细边缘
- 大卷积核看更宽的上下文
- 更适合超声图像这种尺度变化明显、边界模糊的场景

---

## 6. 分割模型：MAFM-AttU_Net

### 6.1 模块插入位置

在当前实现中，MAFM 被插入到 Attention U-Net 解码器的四个阶段：

- `MAFM5`
- `MAFM4`
- `MAFM3`
- `MAFM2`

整体流程是：

```text
Up5 + Att5 -> MAFM5
Up4 + Att4 -> MAFM4
Up3 + Att3 -> MAFM3
Up2 + Att2 -> MAFM2
最后 1x1 Conv 输出分割结果
```

### 6.2 `forward` 的一个关键设计

`model/Models.py` 中的 `AttU_Net_with_MAFM.forward` 支持：

```python
forward(x, extract_features=False)
```

两种模式：

- `extract_features=False`
  - 正常输出分割 mask
- `extract_features=True`
  - 当前代码直接返回 `d2_features`
  - 也就是 `MAFM2` 之后的解码特征图

这一点对 pCR 预测非常重要，因为它决定了 “Ours 特征” 到底从哪一层提取。后面第 10 节会专门说明这和论文最终口径之间的差异。

### 6.3 分割训练配置

当前 `train.py` 给出的主要配置是：

- 模型：`AttU_Net_with_MAFM(img_ch=1, output_ch=1)`
- 输入尺寸：`256 x 256`
- Batch Size：`16`
- Epoch：`150`
- 学习率：`1e-5`
- 验证集比例：`0.2`
- Early Stopping Patience：`20`
- 优化器：`RMSprop`

### 6.4 分割损失函数

当前实现采用的是：

```python
loss = 0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss
```

这个组合很常见，原因是：

- `BCE` 关注像素级分类稳定性
- `DiceLoss` 更直接优化前景区域重叠程度
- 两者结合更适合前景区域较小、类别不平衡的分割任务

### 6.5 分割评价指标

论文和代码中常见的指标包括：

- `Dice`
- `Precision`
- `Recall`
- `mIoU`
- `Accuracy`

其中最核心、最容易在面试中被问的是 `Dice`：

> Dice 衡量的是预测分割区域和真实 mask 的重叠程度，越高说明模型越能把病灶区域分对。

---

## 7. pCR 预测主线：三条特征路线 + 多种分类头

### 7.1 为什么预测阶段要做“多路线对比”

这是本项目很重要的一点。

我不是只做了一个模型然后报结果，而是比较了三种不同的特征来源：

- 分割模型特征
- 通用视觉迁移特征
- 影像组学特征

这样做的意义是：

- 不是预设“分割特征一定最好”
- 而是用实验去回答：**到底哪种特征更适合 pCR 预测**

### 7.2 三条特征路线

#### 路线 1：Ours

- 使用 `MAFM-AttU_Net` 的中间特征
- 代表“病灶导向”的深度特征路线

#### 路线 2：ResNet18

- 使用 ImageNet 预训练的 `ResNet18`
- 代表“通用迁移学习特征”路线

#### 路线 3：Pyradiomics

- 使用 `Pyradiomics` 提取手工影像组学特征
- 代表“传统可解释特征工程”路线

### 7.3 分类头

论文中对比了多种分类头：

- `MLP`
- `XGBoost`
- `SVM`
- `RandomForest`

其中当前 `train_pCR.py` 专门实现的是 **MLP 路线**，`train_ml.py` 则主要覆盖传统机器学习分类器。

### 7.4 为什么要比较多种分类头

因为小样本医学任务里，最终效果不只取决于“特征提取器”，也取决于后续分类器是否适配。

比如：

- 深度特征配轻量 `MLP`，可能更适合端到端表示
- 手工特征配 `SVM`、`RF` 也可能更稳

所以比较多个分类头，本质上是在做：

- 特征路线对比
- 分类器适配性对比
- 小样本泛化能力对比

### 7.5 当前 `StudentMLP` 的结构

`train_pCR.py` 中的 `StudentMLP` 结构很简单：

```text
输入特征
  -> Linear(input_dim, 128)
  -> ReLU
  -> Dropout(0.5)
  -> Linear(128, 1)
  -> BCEWithLogitsLoss
```

它本质上是一个：

- 轻量
- 小样本友好
- 易于和不同特征提取器适配

的二分类头。

### 7.6 交叉验证与评价方式

pCR 预测阶段采用：

- `5-fold StratifiedKFold`
  - 保证每折类别分布尽量一致
- 主要评价指标：
  - `AUC`
  - `Accuracy`
  - `Confusion Matrix`
  - 分类报告中的 `Precision / Recall / F1`

其中最核心的仍然是 `AUC`：

> AUC 衡量的是模型把 pCR 阳性和阴性样本区分开的能力，越接近 1 越好，0.5 相当于随机猜测。

---

## 8. 三条特征路线的代码实现

### 8.1 Ours 路线

在 `train_pCR.py` 中，如果选择：

```python
FEATURE_EXTRACTOR = 'ours'
```

则会：

1. 加载 `AttU_Net_with_MAFM`
2. 读取 `best_model.pth`
3. 在 `extract_features=True` 模式下取出特征图
4. 做全局平均池化
5. 展平后送入 `StudentMLP`

当前代码给出的基准维度是：

- `BASE_FEATURE_DIM_OURS = 64`

这说明当前仓库实现中，Ours 路线默认使用的是 **64 维 pooled 特征**，也就是来自 `MAFM2` 输出。

### 8.2 ResNet18 路线

如果选择：

```python
FEATURE_EXTRACTOR = 'resnet18'
```

则会：

1. 加载 `torchvision.models.resnet18(weights=DEFAULT)`
2. 去掉最后的全连接层，只保留到 `avgpool`
3. 因为超声图是单通道，所以先复制成 3 通道
4. 输出 `512` 维特征

维度规则：

- `pre` 或 `post`：`512`
- `mixed`：`1024`

### 8.3 Pyradiomics 路线

如果选择：

```python
FEATURE_EXTRACTOR = 'pyradiomics'
```

则会：

1. 用 `SimpleITK` 把图像转成可被 Pyradiomics 处理的格式
2. 用全图 mask 提取特征
3. 启用以下特征类：
   - `shape`
   - `firstorder`
   - `glcm`
   - `glrlm`
   - `glszm`
   - `ngtdm`
   - `gldm`
4. 对每折特征做 `StandardScaler`

这是一个更偏传统影像组学的路线，优点是：

- 可解释性更强
- 对小样本有时比较稳

缺点是：

- 高度依赖手工特征设计
- 对 end-to-end 语义表达能力有限

### 8.4 `pre / post / mixed` 在代码里怎么处理

`utils/pCR_Dataset.py` 中：

- `pre`
  - 返回一张 `preNAC/us001.png`
- `post`
  - 返回一张 `postNAC/us001.png`
- `mixed`
  - 同时返回 `(pre_image, post_image)`

随后在特征提取阶段：

- `pre/post`：直接提单时点特征
- `mixed`：分别提 pre 与 post 特征，再在特征维上拼接

---

## 9. 实验结果与结论

### 9.1 分割结果

论文在 BUSI 数据集上的主要结果如下：

| 模型 | Dice (%) | Precision (%) | Recall (%) | F1 (%) |
|------|----------|---------------|------------|--------|
| U-Net | 63.8 | 65.4 | 85.0 | 73.9 |
| U-Net++ | 67.7 | 70.6 | 84.2 | 76.8 |
| VNet | 68.9 | 73.1 | 85.1 | 78.6 |
| U2Net | 65.6 | 70.0 | 83.3 | 76.1 |
| AttU_Net | 69.5 | 75.0 | 81.9 | 78.3 |
| AttU_Net + EMA | 70.0 | 75.7 | 81.8 | 78.6 |
| **MAFM-AttU_Net** | **73.07** | **75.87** | **85.77** | **80.52** |

论文还给出了：

- `mIoU = 78.55%`
- `ACC = 96.12%`

### 9.2 分割结果怎么解释

面试中最稳的解释是：

- Dice 从 `69.5` 提升到 `73.07`
  - 说明改进后的模型在病灶区域重叠度上更好
- 尤其是在恶性病灶边界模糊、形态不规则时，MAFM 对边界细节更有帮助
- 说明把多尺度信息融合放在解码阶段是合理的

### 9.3 pCR 预测的 MLP 结果矩阵

论文第 5 章给出了 `MLP` 分类头下的 5 折交叉验证结果：

| 特征提取器 | 数据模式 | AUC (Avg ± Std) | Accuracy (Avg ± Std) | Avg Val Loss |
|------|------|------|------|------|
| Ours (MAFM-AttU_Net) | pre | 0.6270 ± 0.1711 | 0.6009 ± 0.0690 | 0.6720 ± 0.0157 |
| Ours (MAFM-AttU_Net) | post | 0.7219 ± 0.1650 | 0.5959 ± 0.0218 | 0.6555 ± 0.0283 |
| Ours (MAFM-AttU_Net) | mixed | 0.6536 ± 0.2283 | 0.6145 ± 0.0178 | 0.6626 ± 0.0166 |
| ResNet18 | pre | 0.6365 ± 0.0425 | 0.6385 ± 0.0556 | 0.6399 ± 0.0226 |
| **ResNet18** | **post** | **0.7910 ± 0.1183** | **0.7041 ± 0.1340** | **0.5233 ± 0.1034** |
| ResNet18 | mixed | 0.7155 ± 0.0919 | 0.6891 ± 0.0826 | 0.5840 ± 0.0500 |
| Pyradiomics | pre | 0.5855 ± 0.1210 | 0.6381 ± 0.1087 | 0.6540 ± 0.0470 |
| Pyradiomics | post | 0.7266 ± 0.1413 | 0.6175 ± 0.0748 | 0.5836 ± 0.0748 |
| Pyradiomics | mixed | 0.6679 ± 0.0815 | 0.6127 ± 0.1093 | 0.6228 ± 0.0753 |

### 9.4 pCR 主结论

最关键的结论只有一句：

> **在 143 例临床数据上，postNAC 模式下 ResNet18 + MLP 的 5 折交叉验证平均 AUC 达到 0.791，是整体最优方案。**

### 9.5 为什么最后不是 Ours 最优

这是最容易被追问的点。

更稳的回答逻辑是：

- 分割任务和 pCR 预测任务本质不同
  - 分割追求的是像素级 mask 重叠
  - pCR 预测追求的是病例级二分类区分能力
- Ours 特征更偏病灶区域与边界表达
- 但 pCR 预测还依赖更复杂的治疗后纹理变化、残余病灶模式、整体图像分布差异
- 在小样本条件下，ImageNet 预训练的 ResNet18 迁移特征泛化更强
- 所以最终由 ResNet18 + MLP 在 postNAC 上取得最好结果

### 9.6 这个结果说明了什么

可以从三层来解释：

1. `postNAC > preNAC`
   - 治疗后图像更接近最终病理结局

2. `ResNet18 > Ours / Pyradiomics`
   - 通用迁移特征在小样本二分类任务上更稳

3. `Ours` 仍然有价值
   - 它证明了分割改进路线是可行的
   - 在 postNAC 下也能达到 `0.7219`
   - 说明病灶导向特征并非无效，只是当前数据规模和实现方式下不如 ResNet18 稳定

---

## 10. 论文口径与当前代码的差异

这一节非常重要，既是你准备面试时的防穿点，也是你复习时必须知道的边界。

### 10.1 README 不是这篇毕设的主说明

当前仓库的 `README.md` 仍然主要在讲**裂缝分割/APCGAN 相关 artifact**，不是你这篇乳腺超声毕设的完整说明。

因此：

- 面试不要按 README 讲项目
- README 只能看作旧工程遗留背景
- 你的主口径应该以毕业论文和当前相关代码为准

### 10.2 `train.py` 中的数据路径是旧工程路径

`train.py` 默认仍然使用：

- `images/cracks_tradition`
- `PROJECT_ROOT = "/tmp/pycharm_project_734"`

这说明当前分割训练脚本保留了旧项目路径与默认配置，不等价于论文中最终使用的 BUSI 数据实验配置。

换句话说：

- **模型结构是可参考的**
- **训练脚本的默认路径与 README 说明不是论文最终实验现场**

### 10.3 论文里的 Ours 特征层位，与当前代码不完全一致

这是最大的一个口径差异。

论文英文摘要与第 5 章写的是：

- Ours 路线使用 **`MAFM-AttU_Net@MAFM5` 的 512 维特征**
- mixed 模式下拼接成 `1024` 维

但当前代码里：

- `AttU_Net_with_MAFM.forward(extract_features=True)` 返回的是 `d2_features`
- 也就是 `MAFM2` 后的特征图
- `train_pCR.py` 中还明确写了：
  - `BASE_FEATURE_DIM_OURS = 64`

也就是说：

- **论文口径：Ours = MAFM5, 512 维**
- **当前代码口径：Ours = MAFM2, pooled 后 64 维**

面试时如果老师只问项目思路和结果，优先按**论文/PPT稳定口径**回答。  
如果老师追问到代码实现，可以补一句：

> 当前仓库里的特征提取实现做过后续调整，`extract_features=True` 默认返回的是更靠后的解码特征，用于本地对比实验；而论文最终汇报口径采用的是 MAFM5 特征。

### 10.4 当前 `train_pCR.py` 默认配置不是最终最优实验

当前文件默认写的是：

```python
FEATURE_EXTRACTOR = 'pyradiomics'
DATA_MODE = 'mixed'
```

但论文最终最佳结果是：

- `ResNet18 + MLP + postNAC`

所以：

- 代码默认值只是某次实验配置
- 不是最终结论本身

### 10.5 当前代码可直接加载的样本数少于论文统计

论文稳定口径：

- 临床数据：`143` 例

当前工作区按 `pCR_Dataset.py` 的 `us001.png` 规则统计：

- pre：`108`
- post：`94`
- mixed：`52`

这说明当前仓库本地文件组织与论文最终统计口径之间存在差异，可能原因包括：

- 代码只读取固定编号图像 `us001.png`
- 部分病例图像未完整保留在当前工作区
- 论文统计的是预处理和筛选后的最终实验集，而不是当前仓库的直接文件可见数

面试时不要主动讲这些仓库细节，但自己心里要清楚。

---

## 11. 关键张量维度速查表

以下默认输入尺寸为 `256 x 256`，batch size 记为 `B`。

### 11.1 分割模型维度

```text
输入：
  x                     [B, 1, 256, 256]

Encoder：
  e1 = Conv1            [B, 64, 256, 256]
  e2 = Conv2            [B, 128, 128, 128]
  e3 = Conv3            [B, 256, 64, 64]
  e4 = Conv4            [B, 512, 32, 32]
  e5 = Conv5            [B, 1024, 16, 16]

Decoder：
  d5_up                 [B, 512, 32, 32]
  d5 = MAFM5            [B, 512, 32, 32]
  d4_up                 [B, 256, 64, 64]
  d4 = MAFM4            [B, 256, 64, 64]
  d3_up                 [B, 128, 128, 128]
  d3 = MAFM3            [B, 128, 128, 128]
  d2_up                 [B, 64, 256, 256]
  d2_features = MAFM2   [B, 64, 256, 256]

分割输出：
  out                   [B, 1, 256, 256]
```

### 11.2 Ours 特征路线维度

#### 当前代码实现

```text
extract_features=True
  -> 返回 d2_features      [B, 64, 256, 256]
  -> AdaptiveAvgPool2d     [B, 64, 1, 1]
  -> Flatten               [B, 64]

mixed 模式：
  pre 64 + post 64 -> [B, 128]
```

#### 论文稳定口径

```text
使用 MAFM5 特征
  -> pooled 后单时点约 [B, 512]
  -> mixed 模式拼接后 [B, 1024]
```

### 11.3 ResNet18 特征路线维度

```text
输入灰度图复制到 3 通道
  [B, 1, 256, 256] -> [B, 3, 256, 256]

ResNet18 avgpool 输出：
  pre / post           [B, 512]
  mixed                [B, 1024]
```

### 11.4 MLP 分类头维度

```text
输入特征              [B, D]
Linear(D, 128)        [B, 128]
ReLU                  [B, 128]
Dropout(0.5)          [B, 128]
Linear(128, 1)        [B, 1]
```

---

## 12. 项目目录结构与文件职责

```text
APCGAN-AttuNet-main/
│
├── README.md
│   └── 旧 artifact 说明，主要是裂缝分割/APCGAN 背景，不是当前毕设主说明
│
├── train.py
│   └── 分割模型训练脚本
│      当前默认模型是 AttU_Net_with_MAFM
│      当前损失是 0.5*BCE + 0.5*Dice
│
├── train_ml.py
│   └── 传统机器学习分类器实验
│      支持 Ours / ResNet18 / Pyradiomics 特征
│      支持 XGBoost / SVM / RandomForest
│
├── train_pCR.py
│   └── MLP 分类头实验主脚本
│      支持 Ours / ResNet18 / Pyradiomics
│      支持 pre / post / mixed
│      采用 5 折交叉验证
│
├── comparison.py
│   └── 可视化对比脚本，用于拼接不同分割模型结果图
│
├── model/
│   ├── Models.py
│   │   └── 主要自定义模型都在这里
│   │      包括 Attention U-Net、MAFM、AttU_Net_with_MAFM 等
│   ├── unet_model.py
│   │   └── 标准 U-Net 参考实现
│   └── unet_parts.py
│       └── U-Net 基本模块
│
├── utils/
│   ├── dataset.py
│   │   └── 分割数据集加载器
│   ├── pCR_Dataset.py
│   │   └── pCR 预测数据加载器
│   ├── loss.py
│   │   └── DiceLoss
│   └── utils_metrics.py
│       └── mIoU / Precision / Recall / Accuracy 等分割评价工具
│
├── US_filtered/
│   └── 临床超声数据与 clinical_filtered.csv
│
├── checkpoints/
│   └── 分割模型权重
│
├── pcr_checkpoints_student_5fold_mlp/
│   └── pCR 预测阶段的 MLP 模型权重
│
└── TEACHING_DOC.md
│   └── 本文档
```

### 各核心文件最该记住的一句话

- `model/Models.py`
  - 模型核心都在这里，MAFM 和 AttU_Net_with_MAFM 是重点
- `train.py`
  - 分割怎么训，loss 怎么配，早停怎么做
- `utils/pCR_Dataset.py`
  - pre/post/mixed 三种模式怎么组织数据
- `train_pCR.py`
  - 三条特征路线如何进入 MLP，5 折怎么评估
- `train_ml.py`
  - 传统机器学习分类头的对比实验

---

## 附录：项目最稳的三句总结

### 30 秒版

这个项目研究的是如何利用乳腺超声图像预测 NAC 后是否达到 pCR。考虑到超声图像噪声多、病灶边界模糊，我先做了分割改进，在 Attention U-Net 中引入 MAFM，使 BUSI 上的 Dice 达到 73.07%。随后我比较了分割特征、Pyradiomics 和 ResNet18 三条预测路线，最终在 143 例临床数据上发现，postNAC 模式下 ResNet18 + MLP 的 5 折平均 AUC 最好，达到 0.791。

### 1 分钟版

这个项目的主任务其实是 pCR 预测，分割是关键辅助环节。因为超声图像噪声多、病灶边缘模糊，如果直接做二分类，模型可能学到很多和病灶无关的噪声，所以我先对图像做预处理，并在 Attention U-Net 的解码阶段加入 MAFM 来增强边界表征。分割部分在 BUSI 数据集上的 Dice 达到 73.07%，比基线 Attention U-Net 更好。然后在预测阶段，我系统比较了 Ours 分割特征、ResNet18 迁移特征和 Pyradiomics 特征，并结合 MLP、XGBoost、SVM、RandomForest 等分类头以及 preNAC、postNAC、mixed 三种数据模式做 5 折交叉验证。最终发现 postNAC 下的 ResNet18 + MLP 表现最好，平均 AUC 为 0.791。

### 项目最容易被追问的 5 个点

- 为什么先做分割，而不是直接分类？
- 你的创新点到底在哪里，和原始 Attention U-Net 的差别是什么？
- 为什么最后不是你的 Ours 特征最好，而是 ResNet18 最好？
- 为什么 `postNAC` 比 `preNAC` 更有效？
- 论文口径和当前代码实现有哪些不完全一致的地方？

---

*文档版本：2026-03-20（基于毕业论文 + 当前代码仓整理）*
