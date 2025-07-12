import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from torch.nn import Conv2d

from model.unet_model import UNet
from model.Models import AttU_Net, AttU_Net_min, AttU_Net_with_MAFM, NestedUNet
from utils.dataset import ISBI_Loader
from utils.loss import DiceLoss
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50, lraspp_mobilenet_v3_large

# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pytorch_grad_cam import GradCAM
import time
import matplotlib.pyplot as plt
import pandas as pd
# ... (other imports remain the same) ...
import os
from torch.utils.data import DataLoader, random_split

# --- Configuration ---
# 获取当前文件所在目录作为项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
print(f"Project Root: {PROJECT_ROOT}")

# 定义相对路径
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "BUSI")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


# ==============================================================================
# train_net function definition remains the same as the previous version
# It uses the globally defined absolute paths: MODEL_SAVE_DIR and RESULTS_DIR
# and receives the absolute data_path as an argument.
# ==============================================================================
def train_net(net, device, data_path, epochs=300, batch_size=16, lr=0.00001, val_percent=0.2, patience=20, loss_weights=(0.5, 0.5)):
    # 加载完整数据集
    isbi_dataset = ISBI_Loader(data_path) # data_path is absolute
    print(f'Loading dataset from: {data_path}')
    print(f'Full dataset size: {len(isbi_dataset)}')

    # 1. 划分数据集
    n_val = int(len(isbi_dataset) * val_percent)
    n_train = len(isbi_dataset) - n_val
    train_set, val_set = random_split(isbi_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    print(f'Train set size: {len(train_set)}')
    print(f'Validation set size: {len(val_set)}')

    # 2. 创建 DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 定义优化器和损失函数
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()  # 实例化 Dice Loss
    bce_weight, dice_weight = loss_weights  # 获取权重

    # 4. 记录 Loss 和指标
    train_losses = []
    val_losses = []
    # (可选) train_dices = [], val_dices = []
    # 初始化最佳验证 Loss 和耐心计数器
    best_val_loss = float('inf')
    epochs_no_improve = 0  # 用于计数的变量

    # 5. 更新模型保存逻辑 (使用绝对路径)
    best_val_loss = float('inf')
    if not os.path.exists(MODEL_SAVE_DIR): # Use absolute path
        os.makedirs(MODEL_SAVE_DIR)
        print(f"Created directory: {MODEL_SAVE_DIR}")
    best_model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth') # Absolute path

    starttime = time.time()
    actual_epochs_run = 0

    for epoch in range(epochs):
        actual_epochs_run += 1
        print(f'--- Epoch {epoch + 1}/{epochs} ---')

        # --- 训练部分 ---
        net.train()
        running_train_loss = 0.0
        # (可选)可以分别记录 bce 和 dice loss
        # running_train_bce = 0.0
        # running_train_dice = 0.0
        pbar_train = tqdm(total=len(train_loader), desc=f'Training Epoch {epoch + 1}', unit='batch')
        for image, label in train_loader:
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            image = image / 255.0
            optimizer.zero_grad()
            pred = net(image)

            # --- 修改：计算组合损失 ---
            loss_bce = criterion_bce(pred, label)
            loss_dice = criterion_dice(pred, label)
            loss = bce_weight * loss_bce + dice_weight * loss_dice  # 加权组合

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * image.size(0)
            # (可选) running_train_bce += loss_bce.item() * image.size(0)
            # (可选) running_train_dice += loss_dice.item() * image.size(0)

            pbar_train.update(1)
            pbar_train.set_postfix(
                **{'comb_loss': loss.item(), 'bce': loss_bce.item(), 'dice': loss_dice.item()})  # 显示更多loss信息
        pbar_train.close()
        epoch_train_loss = running_train_loss / len(train_set)
        train_losses.append(epoch_train_loss)
        # (可选计算平均 bce/dice)
        print(f'Epoch {epoch + 1} Training Loss (Combined): {epoch_train_loss:.4f}')

        # --- 验证循环 ---
        net.eval()
        running_val_loss = 0.0
        # (可选) running_val_bce = 0.0
        # (可选) running_val_dice = 0.0
        pbar_val = tqdm(total=len(val_loader), desc=f'Validation Epoch {epoch + 1}', unit='batch')
        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                image = image / 255.0
                pred = net(image)

                # --- 修改：计算组合损失 ---
                loss_bce = criterion_bce(pred, label)
                loss_dice = criterion_dice(pred, label)
                loss = bce_weight * loss_bce + dice_weight * loss_dice  # 使用相同的组合损失进行验证

                running_val_loss += loss.item() * image.size(0)
                # (可选记录 bce/dice)
                pbar_val.update(1)
                pbar_val.set_postfix(**{'val_comb_loss': loss.item(), 'val_bce': loss_bce.item(),
                                        'val_dice': loss_dice.item()})  # 显示更多loss信息
        pbar_val.close()
        epoch_val_loss = running_val_loss / len(val_set)
        val_losses.append(epoch_val_loss)
        print(f'Epoch {epoch + 1} Validation Loss (Combined): {epoch_val_loss:.4f}')

        # --- 早停逻辑 (基于组合验证损失 epoch_val_loss) ---
        # ... (早停逻辑不变，但现在是基于组合损失来判断) ...
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(net.state_dict(), best_model_path)
            print(
                f'>>> Validation Loss Improved! Saving Best model to {best_model_path} at epoch {epoch + 1} with Val_Loss: {best_val_loss:.4f}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'Validation Loss did not improve for {epochs_no_improve} consecutive epoch(s).')
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement.')
            break

    # ... (训练结束后的时间打印、日志保存、绘图代码不变，但绘图标签可以修改) ...
    # --- 绘图功能 ---
    plt.figure(figsize=(12, 5))
    epochs_range = range(1, actual_epochs_run + 1)
    plt.plot(epochs_range, train_losses[:actual_epochs_run], label='Training Loss (Combined)')  # 修改标签
    plt.plot(epochs_range, val_losses[:actual_epochs_run], label='Validation Loss (Combined)')  # 修改标签
    # ... (绘制最佳点代码不变) ...
    plt.xlabel('Epochs')
    plt.ylabel(f'Loss ({bce_weight}*BCE + {dice_weight}*Dice)')  # 修改Y轴标签
    plt.title('Training & Validation Combined Loss')  # 修改标题
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'loss_curves.png')
    plt.savefig(plot_path)
    print(f"Loss curves plot saved to {plot_path}")
    # plt.show()


# ==============================================================================
# Main execution block
# ==============================================================================
# --- 4. 在 __main__ 部分设置并传递 patience ---
if __name__ == "__main__":
    # ... (PROJECT_ROOT 定义和设备选择不变) ...
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # net = UNet(n_channels=1, n_classes=1)
    # net = AttU_Net1(img_ch=1, output_ch=1)
    # net = AttU_Net(img_ch=1, output_ch=1)
    # net = NestedUNet(in_ch=1, out_ch=1)
    # net = VNet2D(in_channels=1, out_channels=1, elu=True)
    # net = U2NET(in_ch=1, out_ch=1)  # 使用 U2NET
    # net = U2NETP(in_ch=1, out_ch=1) # 或者使用 U2NETP
    # net = AttUNet_with_EGA(img_ch=1, output_ch=1, ema_factor=8, deep_supervision=False)  # <--- REPLACE/ADD THIS LINE
        # Set deep_supervision=False
    # net = AttU_Net2(img_ch=1, output_ch=1, ema_factor=8)
    # net = DilateAttUNet(
    #     img_ch=1,
    #     output_ch=1,
    #     ema_factor=8,
    #     mda_num_heads=8,  # Changed from 6 to 8
    #     mda_dilation_rates=[1, 2]  # Changed from [1, 2, 3] to [1, 2]
    # )
    net = AttU_Net_with_MAFM(img_ch=1, output_ch=1)
    # net = AttU_Net0(
    #     img_ch=1,  # 输入图像通道数
    #     output_ch=1,  # 输出类别数
    #     ema_factor=8,  # EMA 模块的因子 (示例值)
    # )
    # net = AttU_Net1_with_GFM(img_ch=1, output_ch=1, gfm_expend_ratio=2)

    net.to(device=device)

    data_path = DEFAULT_DATA_PATH
    # ... (打印路径信息不变) ...
    # Project Root is printed near the top
    print(f"Using data from: {data_path}")
    print(f"Saving checkpoints to: {MODEL_SAVE_DIR}")  # Absolute path
    print(f"Saving results to: {RESULTS_DIR}")  # Absolute path

    # 设置训练参数
    EPOCHS = 150
    BATCH_SIZE = 16
    LEARNING_RATE = 0.00001
    VALIDATION_PERCENT = 0.2
    PATIENCE = 20
    # --- 设置损失函数的权重 ---
    # (BCE 权重, Dice 权重), 和为 1.0 是常见的做法，可以调整
    LOSS_WEIGHTS = (0.5, 0.5)

    # 开始训练
    train_net(net=net,
              device=device,
              data_path=data_path,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              lr=LEARNING_RATE,
              val_percent=VALIDATION_PERCENT,
              patience=PATIENCE,
              loss_weights=LOSS_WEIGHTS)  # 传递损失权重
