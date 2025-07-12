import torch
import torch.nn as nn
import torch.nn.functional as F # 可能需要 F 用于其他操作，先导入

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 定义 Dice Loss 类
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Dice Loss for segmentation tasks.
        Args:
            smooth (float): A small epsilon value added to numerator and denominator
                            for numerical stability and to avoid division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Calculate Dice Loss.
        Args:
            logits (torch.Tensor): Raw output from the model (before activation).
                                   Shape: (N, C, H, W) - for binary C=1.
            targets (torch.Tensor): Ground truth label map (binary 0 or 1).
                                    Shape: (N, C, H, W) - for binary C=1.
        Returns:
            torch.Tensor: Calculated Dice Loss (scalar).
        """
        # 1. 应用 Sigmoid 激活函数获取概率 [0, 1]
        probs = torch.sigmoid(logits)

        # 2. 展平张量，方便计算 (N, C*H*W)
        #    保留 batch 维度 (dim=0)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # 3. 计算交集 (Intersection) 和各项的和
        #    乘法操作后按样本求和
        intersection = (probs * targets).sum(dim=1)
        pred_sum = probs.sum(dim=1)
        target_sum = targets.sum(dim=1)

        # 4. 计算 Dice 系数 (per sample in batch)
        #    使用 smooth 因子避免除零
        dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        # 5. 计算 Dice Loss (1 - Dice) 并求批次平均值
        #    我们希望最大化 Dice，所以最小化 1 - Dice
        dice_loss = 1 - dice.mean()

        return dice_loss