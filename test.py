# import os
# import time
#
# # from matplotlib import pyplot as plt # Keep if you plan to use plt later
# # from pytorch_grad_cam import GradCAM # Keep if you plan to use CAM later
# # from pytorch_grad_cam.utils.image import show_cam_on_image
# # from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
# from torch.nn import Conv2d
# # from torch.utils.tensorboard import SummaryWriter # Keep if you plan to use TensorBoard later
# from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50, lraspp_mobilenet_v3_large
# from tqdm import tqdm
#
# from model.enet import ENet
# from utils.utils_metrics import compute_mIoU, show_results
# # import glob # Not used currently
# import numpy as np
# import torch
# import os
# from model.Models import AttU_Net, AttU_Net_min, AttU_Net1
# import cv2
# from model.unet_model import UNet
#
#
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # ++ Function to calculate Dice Score for a single pair ++
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# def calculate_dice_score(pred_mask, gt_mask, threshold=127, epsilon=1e-6):
#     """
#     Calculates the Dice Similarity Coefficient (DSC) between two binary masks.
#
#     Args:
#         pred_mask (np.ndarray): Predicted mask (grayscale image, 0-255).
#         gt_mask (np.ndarray): Ground truth mask (grayscale image, 0-255).
#         threshold (int): Value to threshold the images for binarization.
#         epsilon (float): Small value to prevent division by zero.
#
#     Returns:
#         float: Dice Similarity Coefficient. Returns np.nan on error.
#     """
#     if pred_mask is None or gt_mask is None:
#         print("Warning: Invalid input mask(s) for Dice calculation.")
#         return np.nan
#     if pred_mask.shape != gt_mask.shape:
#         print(f"Warning: Shape mismatch for Dice calculation! Pred: {pred_mask.shape}, GT: {gt_mask.shape}")
#         return np.nan
#
#     # Binarize the masks (assuming input are grayscale 0-255)
#     # Pixels > threshold are considered positive (1), others negative (0)
#     pred_mask_bin = (pred_mask > threshold)
#     gt_mask_bin = (gt_mask > threshold)
#
#     # Calculate intersection (True Positives)
#     intersection = np.sum(pred_mask_bin & gt_mask_bin)
#
#     # Calculate sum of positive pixels in each mask
#     sum_pred = np.sum(pred_mask_bin)
#     sum_gt = np.sum(gt_mask_bin)
#
#     # Calculate Dice coefficient
#     # The epsilon handles the case where both sum_pred and sum_gt are 0 (empty masks) correctly -> dice = 1.0
#     dice = (2. * intersection + epsilon) / (sum_pred + sum_gt + epsilon)
#
#     return dice
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#
# def cal_miou(test_dir="/tmp/pycharm_project_734/images/cracks_tradition/Test_Images",
#              pred_dir="/tmp/pycharm_project_734/images/cracks_tradition/results", gt_dir="/tmp/pycharm_project_734/images/cracks_tradition/Test_Labels"):
#     # ... (miou_mode, num_classes, name_classes remain the same) ...
#     miou_mode = 0
#     num_classes = 2
#     name_classes = ["background", "crack"]
#
#     image_ids = None # Initialize image_ids
#
#     # --- Prediction Phase ---
#     if miou_mode == 0 or miou_mode == 1:
#         if not os.path.exists(pred_dir):
#             os.makedirs(pred_dir)
#
#         print("Load model.")
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         net = AttU_Net1(img_ch=1, output_ch=1) # todo: change the model if necessary
#         net.to(device=device)
#         # Make sure the model path is correct
#         model_path = 'best_model.pth' # todo: verify path
#         if not os.path.exists(model_path):
#              print(f"Error: Model weights file not found at {model_path}")
#              return
#         net.load_state_dict(torch.load(model_path, map_location=device))
#         net.eval()
#         print("Load model done.")
#
#         # Check if test directory exists
#         if not os.path.isdir(test_dir):
#             print(f"Error: Test directory not found: {test_dir}")
#             return
#
#         img_names = os.listdir(test_dir)
#         # Filter for common image types if necessary, e.g., .jpg, .png
#         image_ids = [os.path.splitext(image_name)[0] for image_name in img_names if image_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
#
#         if not image_ids:
#             print(f"Error: No valid image files found in {test_dir}")
#             return
#
#         print("Get predict result.")
#         times=[]
#         for image_id in tqdm(image_ids, desc="Generating Predictions"):
#
#             # Try common extensions if .jpg is not found
#             image_path = None
#             for ext in [".jpg", ".png", ".jpeg"]:
#                 potential_path = os.path.join(test_dir, image_id + ext)
#                 if os.path.exists(potential_path):
#                     image_path = potential_path
#                     break
#
#             if image_path is None:
#                 print(f"Warning: Image file for ID {image_id} not found in {test_dir}, skipping.")
#                 continue
#
#             # label_path = os.path.join(gt_dir, image_id + ".png") # Ground truth not needed for prediction generation
#
#             img = cv2.imread(image_path)
#             if img is None:
#                  print(f"Warning: Could not read image {image_path}, skipping.")
#                  continue
#
#             origin_shape = img.shape
#             img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Use BGR2GRAY for consistency
#             img_resized = cv2.resize(img_gray, (256, 256)) # Define target size, e.g., 256x256
#
#             # Normalize if necessary (model dependent) - Assuming input is 0-255 for now
#             # img_normalized = img_resized / 255.0
#
#             # Reshape and convert to tensor
#             img_batch = img_resized.reshape(1, 1, img_resized.shape[0], img_resized.shape[1])
#             img_tensor = torch.from_numpy(img_batch)
#             img_tensor = img_tensor.to(device=device, dtype=torch.float32)
#
#             # Predict
#             with torch.no_grad(): # Important for inference
#                 starttime = time.time()
#                 pred_tensor = net(img_tensor)
#                 endtime = time.time()
#             times.append(endtime-starttime)
#
#             # Process prediction output
#             # Adjust depending on model output format (e.g., logits, probabilities, segmentation map)
#             # Assuming model outputs logits/probabilities of shape [1, 1, H, W] for binary case
#             pred = pred_tensor.data.cpu().numpy()[0, 0, :, :] # Remove batch and channel dims
#
#             # Apply sigmoid if model outputs logits, then threshold
#             # pred = 1 / (1 + np.exp(-pred)) # Apply sigmoid if necessary
#             pred_binary = (pred >= 0.5).astype(np.uint8) * 255 # Thresholding
#
#             # Resize back to original shape
#             pred_resized = cv2.resize(pred_binary, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
#
#             # Save prediction
#             save_path = os.path.join(pred_dir, image_id + ".png")
#             cv2.imwrite(save_path, pred_resized)
#
#         print("\nGet predict result done.")
#         if times:
#             print(f"Average prediction time per image: {np.mean(times):.4f} seconds")
#         else:
#             print("No predictions were generated.")
#
#
#     # --- Metric Calculation Phase ---
#     if miou_mode == 0 or miou_mode == 2:
#         print("\n--- Starting Metric Calculation ---")
#
#         # Ensure image_ids is available if mode is 2
#         if image_ids is None:
#             if not os.path.exists(pred_dir):
#                  print(f"Error: Prediction directory '{pred_dir}' not found for mode 2.")
#                  return
#             # Get IDs from prediction files if not generated in this run
#             pred_files = os.listdir(pred_dir)
#             image_ids = [os.path.splitext(f)[0] for f in pred_files if f.endswith(".png")]
#             if not image_ids:
#                 print(f"Error: No prediction files (.png) found in {pred_dir} for metric calculation in mode 2.")
#                 return
#             print(f"Found {len(image_ids)} prediction files for metric calculation.")
#
#         # Check directories
#         if not os.path.isdir(pred_dir):
#              print(f"Error: Prediction directory not found: {pred_dir}")
#              return
#         if not os.path.isdir(gt_dir):
#              print(f"Error: Ground Truth directory not found: {gt_dir}")
#              return
#
#         # +++++++++++++++++++++++++++
#         # ++ Dice Score Calculation ++
#         # +++++++++++++++++++++++++++
#         print("\nCalculating Dice scores...")
#         all_dice_scores = []
#         dice_results = {} # Optional: store per-image results
#
#         for image_id in tqdm(image_ids, desc="Calculating Dice"):
#             pred_path = os.path.join(pred_dir, image_id + ".png")
#             # Assume GT is also PNG, adjust if needed
#             gt_path = os.path.join(gt_dir, image_id + ".png")
#
#             if not os.path.exists(pred_path):
#                 print(f"\nWarning: Prediction file not found {pred_path}, skipping Dice for {image_id}.")
#                 continue
#             if not os.path.exists(gt_path):
#                 print(f"\nWarning: Ground truth file not found {gt_path}, skipping Dice for {image_id}.")
#                 continue
#
#             pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
#             gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
#
#             if pred_img is None:
#                  print(f"\nWarning: Failed to load prediction image {pred_path}, skipping Dice.")
#                  continue
#             if gt_img is None:
#                  print(f"\nWarning: Failed to load ground truth image {gt_path}, skipping Dice.")
#                  continue
#
#             # Calculate Dice score using the function defined above
#             dice = calculate_dice_score(pred_img, gt_img, threshold=127) # Use threshold > 0 or > 127 based on 0/1 or 0/255 format
#
#             if not np.isnan(dice):
#                 all_dice_scores.append(dice)
#                 dice_results[image_id] = dice # Store individual score
#             # else: Errors are printed inside calculate_dice_score
#
#         if not all_dice_scores:
#             print("\nNo valid Dice scores were calculated.")
#             average_dice = np.nan
#         else:
#             average_dice = np.mean(all_dice_scores)
#             print(f"\nAverage Dice Score: {average_dice:.4f}")
#             # Optional: print individual scores if needed
#             # print("Individual Dice Scores:")
#             # for img_id, score in sorted(dice_results.items()): # Sort for consistent output
#             #    print(f"  {img_id}: {score:.4f}")
#         # +++++++++++++++++++++++++++
#
#
#         # +++++++++++++++++++++++++++
#         # ++ mIoU Calculation (existing code) ++
#         # +++++++++++++++++++++++++++
#         print("\nCalculating mIoU...")
#         # print(gt_dir) # These paths are inputs, maybe not needed to print again
#         # print(pred_dir)
#         # print(num_classes)
#         # print(name_classes)
#
#         # Ensure compute_mIoU can handle potential missing files gracefully or filter image_ids beforehand
#         valid_image_ids_for_miou = []
#         for image_id in image_ids:
#              pred_path = os.path.join(pred_dir, image_id + ".png")
#              gt_path = os.path.join(gt_dir, image_id + ".png")
#              if os.path.exists(pred_path) and os.path.exists(gt_path):
#                   valid_image_ids_for_miou.append(image_id)
#              # else: Warnings about missing files were likely printed during Dice calc
#
#         if not valid_image_ids_for_miou:
#              print("No valid image pairs found for mIoU calculation.")
#         else:
#              try:
#                  hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, valid_image_ids_for_miou, num_classes,
#                                                                  name_classes)  # Pass only valid IDs
#                  print("mIoU calculation done.")
#                  miou_out_path = "results/" # Define output path for results file
#                  # Ensure the output directory exists
#                  os.makedirs(miou_out_path, exist_ok=True)
#                  show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
#                  print(f"mIoU results saved in '{miou_out_path}'")
#              except Exception as e:
#                  print(f"Error during mIoU calculation or showing results: {e}")
#         # +++++++++++++++++++++++++++
#
#         print("\n--- Metric Calculation Finished ---")
#
#
# if __name__ == '__main__':
#     # Set your directories here if different from defaults
#     test_images_dir = "/tmp/pycharm_project_734/images/cracks_tradition/Test_Images"
#     predictions_dir = "/tmp/pycharm_project_734/images/cracks_tradition/results"
#     ground_truth_dir = "/tmp/pycharm_project_734/images/cracks_tradition/Test_Labels"
#
#     cal_miou(test_dir=test_images_dir, pred_dir=predictions_dir, gt_dir=ground_truth_dir)

# -*- coding: utf-8 -*-
# """
# -------------------------------------------------
# Project Name: unet
# File Name: test.py
# Author: chenming
# Create Date: 2022/2/7
# Description：
# -------------------------------------------------
# """
# import os
# import time
#
# from matplotlib import pyplot as plt
# # from pytorch_grad_cam import GradCAM
# # from pytorch_grad_cam.utils.image import show_cam_on_image
# # from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
# from torch.nn import Conv2d
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50, lraspp_mobilenet_v3_large
# from tqdm import tqdm
#
# from model.enet import ENet
# from utils.utils_metrics import compute_mIoU, show_results
# import glob
# import numpy as np
# import torch
# import os
# from model.Models import AttU_Net, AttU_Net_min, AttU_Net1
# import cv2
# from model.unet_model import UNet
#
#
# def cal_miou(test_dir="/tmp/pycharm_project_734/images/cracks_tradition/Test_Images",
#              pred_dir="/tmp/pycharm_project_734/images/cracks_tradition/results", gt_dir="/tmp/pycharm_project_734/images/cracks_tradition/Test_Labels"):
#     # ---------------------------------------------------------------------------#
#     #   miou_mode用于指定该文件运行时计算的内容
#     #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
#     #   miou_mode为1代表仅仅获得预测结果。
#     #   miou_mode为2代表仅仅计算miou。
#     # ---------------------------------------------------------------------------#
#     miou_mode = 0
#     # ------------------------------#
#     #   分类个数+1、如2+1
#     # ------------------------------#
#     num_classes = 2
#     # --------------------------------------------#
#     #   区分的种类，和json_to_dataset里面的一样
#     # --------------------------------------------#
#     name_classes = ["background", "crack"]
#     # name_classes    = ["_background_","cat","dog"]
#     # -------------------------------------------------------#
#     #   指向VOC数据集所在的文件夹
#     #   默认指向根目录下的VOC数据集
#     # -------------------------------------------------------#
#     # 计算结果和gt的结果进行比对
#
#     # 加载模型
#
#     if miou_mode == 0 or miou_mode == 1:
#         if not os.path.exists(pred_dir):
#             os.makedirs(pred_dir)
#
#         print("Load model.")
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         # net = UNet(n_channels=1, n_classes=1)
#         # net = AttU_Net(img_ch=1, output_ch=1)
#         net = AttU_Net1(img_ch=1, output_ch=1) # todo: change the model
#         # net = deeplabv3_resnet50(num_classes=1)
#         # net = fcn_resnet50(num_classes=1)
#         # net.backbone.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         # net = ENet(num_classes=1, in_channels=1)
#
#         # net = lraspp_mobilenet_v3_large(num_classes=1)
#         # #
#         # # # net.classifier._modules['6'] = nn.Linear(4096, 4)#for vgg16, alexnet
#         # # net.backbone.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # for vgg16, alexnet
#         #
#         # net.backbone._modules['0']._modules['0'] = Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
#         #                                                   bias=False)
#         # 将网络拷贝到deivce中
#         net.to(device=device)
#         # 加载模型参数
#         net.load_state_dict(torch.load('/tmp/pycharm_project_734/checkpoints/best_model.pth', map_location=device)) # todo
#         # 测试模式
#         net.eval()
#         print("Load model done.")
#
#         # target_layers = [net.down4.maxpool_conv]
#
#
#         img_names = os.listdir(test_dir)
#
#         image_ids = [image_name.split(".")[0] for image_name in img_names]
#
#         # kk=0
#         # writer = SummaryWriter('logs/test')
#
#
#         print("Get predict result.")
#         # with GradCAM(model=net, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
#         times=[]
#         for image_id in tqdm(image_ids):
#
#             image_path = os.path.join(test_dir, image_id + ".jpg")
#             label_path = os.path.join(gt_dir, image_id + ".png")
#
#             label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # 确保标签是单通道
#             #label = cv2.imread(label_path)
#             img = cv2.imread(image_path)
#
#             origin_shape = img.shape
#             # print(origin_shape)
#             # 转为灰度图
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             img = cv2.resize(img, (256, 256))
#             label = cv2.resize(label, (256,256))
#             label = (label/255).astype(int)
#             # print(type(label))
#
#             # 转为batch为1，通道为1，大小为512*512的数组
#             img = img.reshape(1, 1, img.shape[0], img.shape[1])
#             # 转为tensor
#             img_tensor = torch.from_numpy(img)
#             # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
#             img_tensor = img_tensor.to(device=device, dtype=torch.float32)
#             # print(img_tensor.size())
#             # 预测
#             starttime = time.time()
#             pred = net(img_tensor)
#             endtime = time.time()
#             times.append(endtime-starttime)
#
#             # print(pred.size())
#             # 提取结果
#             #------------
#             # pred=pred['out']
#             #----------
#
#
#
#             pred = np.array(pred.data.cpu()[0])[0]
#             pred[pred >= 0.5] = 255
#             pred[pred < 0.5] = 0
#             # 将预测结果调整为256x256的大小
#             #pred = cv2.resize(pred, (256, 256), interpolation=cv2.INTER_NEAREST)
#
#             pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
#             cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)
#
#             # targets = [SemanticSegmentationTarget(0, label)]
#             # grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0,:]
#             # cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
#             # plt.imshow(cam_image)
#             # plt.show()
#
#
#
#         print("Get predict result done.")
#         print(np.mean(times))
#
#     if miou_mode == 0 or miou_mode == 2:
#         print("Get miou.")
#         print(gt_dir)
#         print(pred_dir)
#         print(num_classes)
#         print(name_classes)
#         hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
#                                                         name_classes)  # 执行计算mIoU的函数
#         print("Get miou done.")
#         miou_out_path = "results/"
#         show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
#
# if __name__ == '__main__':
#     cal_miou()

# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: test.py
Author: chenming
Create Date: 2022/2/7
Description：
-------------------------------------------------
"""
import os
import time

from matplotlib import pyplot as plt
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from torch.nn import Conv2d
# from torch.utils.tensorboard import SummaryWriter # 如果不用可以注释掉
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50, lraspp_mobilenet_v3_large
from tqdm import tqdm

# from model.enet import ENet # 如果不用可以注释掉
from utils.utils_metrics import compute_mIoU, show_results
import glob
import numpy as np
import torch
import os
from model.Models import AttU_Net, AttU_Net_min, AttU_Net1, VNet2D, NestedUNet, AttU_Net2, DilateAttUNet
from model.Models import U2NET, U2NETP
from model.Models import AttU_Net_with_MAFM, AttU_Net0
import cv2
from model.unet_model import UNet

# --- 添加计算 Dice, Precision, Recall 的函数 ---
def calculate_metrics(pred_binary, label_binary, smooth=1e-6):
    """计算单个图像的 Dice, Precision, Recall"""
    pred_binary = pred_binary.astype(np.bool_)
    label_binary = label_binary.astype(np.bool_)

    tp = np.sum(pred_binary & label_binary)
    fp = np.sum(pred_binary & (~label_binary))
    fn = np.sum((~pred_binary) & label_binary)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    dice = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)

    return dice, precision, recall

# --- 修改 cal_miou 函数 ---
def cal_miou(test_dir="./images/cracks_tradition/Test_Images",
             pred_dir="./results", # 预测结果保存目录
             gt_dir="./images/cracks_tradition/Test_Labels",
             # 使用 checkpoints 目录加载最佳模型
             model_path="./checkpoints/best_model.pth"):

    miou_mode = 0
    num_classes = 2
    name_classes = ["background", "crack"]

    # --- 1. 初始化指标列表 ---
    all_dices = []
    all_precisions = []
    all_recalls = []
    times = [] # 保留原来的推理时间记录

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        # --- 确保加载与训练时相同的模型结构 ---
        # net = UNet(n_channels=1, n_classes=1)
        # net = AttU_Net1(img_ch=1, output_ch=1)
        # net = AttU_Net(img_ch=1, output_ch=1)
        # net = NestedUNet(in_ch=1, out_ch=1)
        net = VNet2D(in_channels=1, out_channels=1, elu=True)
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
        # net = AttU_Net_with_MAFM(img_ch=1, output_ch=1)
        # net = AttU_Net0(
        #     img_ch=1,  # 输入图像通道数
        #     output_ch=1,  # 输出类别数
        #     ema_factor=8,  # EMA 模块的因子 (示例值)
        # )
        net.to(device=device)

        # --- 修改加载模型参数的路径 ---
        if os.path.exists(model_path):
            net.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model weights from: {model_path}")
        else:
            print(f"Error: Model weights not found at {model_path}")
            return # 如果模型不存在则退出

        net.eval()
        print("Load model done.")

        img_names = os.listdir(test_dir)
        image_ids = [image_name.split(".")[0] for image_name in img_names]

        print("Generating predictions and calculating metrics per image...")
        for image_id in tqdm(image_ids):
            # --- 准备图像路径和标签路径 ---
            image_path = os.path.join(test_dir, image_id + ".jpg")
            label_path = os.path.join(gt_dir, image_id + ".png") # 假设标签也是 png

            if not os.path.exists(label_path):
                print(f"Warning: Label file not found, skipping {label_path}")
                continue

            # --- 加载原始图像和标签 ---
            img_orig = cv2.imread(image_path) # 加载原始图像用于获取 shape
            label_orig = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # 加载原始灰度标签

            if img_orig is None or label_orig is None:
                 print(f"Warning: Failed to load image or label, skipping {image_path} or {label_path}")
                 continue

            origin_shape = img_orig.shape[:2] # 获取 H, W

            # --- 图像预处理 (与训练时的数据加载器类似，但不做随机增强) ---
            img_proc = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) # 转灰度
            img_proc = cv2.resize(img_proc, (256, 256))           # Resize
            img_proc = img_proc.reshape(1, 1, 256, 256)           # Reshape (B, C, H, W)
            img_proc = img_proc.astype(np.float32) / 255.0         # Normalize and type conversion

            # --- 转为 Tensor 并移到设备 ---
            img_tensor = torch.from_numpy(img_proc).to(device=device, dtype=torch.float32)

            # --- 模型预测 ---
            starttime = time.time()
            with torch.no_grad():
                pred_tensor = net(img_tensor)
            endtime = time.time()
            times.append(endtime - starttime)

            # --- 预测结果后处理 ---
            # 应用 Sigmoid (因为训练时用了 BCEWithLogitsLoss) 并获取 NumPy 数组
            pred_prob = torch.sigmoid(pred_tensor).squeeze().cpu().numpy()
            # 阈值处理得到二值图 (0 或 1)
            pred_binary_resized = (pred_prob >= 0.5).astype(np.float32)
            # Resize 回原始尺寸
            pred_binary_orig = cv2.resize(pred_binary_resized, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)

            # --- 准备用于指标计算的标签 ---
            # 确保原始标签也是二值的 (0 或 1)
            label_binary_orig = (label_orig > 128).astype(np.uint8) # Thresholding in case label is not strictly 0/255

            # --- 2. 计算当前图像的指标 ---
            dice, precision, recall = calculate_metrics(pred_binary_orig, label_binary_orig)
            all_dices.append(dice)
            all_precisions.append(precision)
            all_recalls.append(recall)

            # --- 保存预测图像 (像素值为 0 或 255) ---
            pred_to_save = (pred_binary_orig * 255).astype(np.uint8)
            save_path = os.path.join(pred_dir, image_id + ".png")
            cv2.imwrite(save_path, pred_to_save)

        print("Prediction and metric calculation done.")
        print(f"Average inference time per image: {np.mean(times):.4f} seconds")

        # --- 3. 计算并打印平均指标 ---
        avg_dice = np.nanmean(all_dices) # 使用 nanmean 忽略可能的 NaN 值
        avg_precision = np.nanmean(all_precisions)
        avg_recall = np.nanmean(all_recalls)

        print("\n--- Calculated Metrics (Averaged per Image) ---")
        print(f"Average Dice Coefficient: {avg_dice:.4f}")
        print(f"Average Precision:        {avg_precision:.4f}")
        print(f"Average Recall:           {avg_recall:.4f}")
        print("--------------------------------------------------")


    # --- 保留原来的 mIoU 计算部分 ---
    if miou_mode == 0 or miou_mode == 2:
        print("\nCalculating mIoU metrics (using saved predictions)...")
        # 确保 pred_dir 和 gt_dir 正确
        print(f"Ground Truth Directory: {gt_dir}")
        print(f"Prediction Directory:   {pred_dir}")
        print(f"Number of Classes:      {num_classes}")
        print(f"Class Names:            {name_classes}")

        # 确保 image_ids 列表非空且与 gt_dir/pred_dir 中的文件对应
        if not image_ids:
             print("Warning: No images processed for mIoU calculation.")
        else:
            try:
                hist, IoUs, PA_Recall, Precision_mIoU = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
                print("mIoU calculation done.")
                # 注意：这里的 Precision_mIoU 来自 compute_mIoU 函数，可能与上面计算的 avg_precision 定义不同
                miou_out_path = os.path.join(RESULTS_DIR, "") # 保存到主 results 目录
                show_results(miou_out_path, hist, IoUs, PA_Recall, Precision_mIoU, name_classes)
                print(f"mIoU results shown/saved related to path: {miou_out_path}")
            except Exception as e:
                print(f"Error during mIoU calculation or showing results: {e}")

if __name__ == '__main__':
    # --- 定义全局变量，方便修改 ---
    PROJECT_ROOT = ""
    DATA_SUBDIR = "cracks_tradition" # 或者 "cracks", "cracks_APCGAN", etc.
    MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")
    TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "images", DATA_SUBDIR, "Test_Images")
    GROUND_TRUTH_DIR = os.path.join(PROJECT_ROOT, "images", DATA_SUBDIR, "Test_Labels")
    PREDICTION_SAVE_DIR = os.path.join(PROJECT_ROOT, "results", f"{DATA_SUBDIR}_preds") # 每个数据集的预测存到不同子目录
    # 使用全局变量 RESULTS_DIR (在 cal_miou 中也引用了)
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


    # 确保预测保存目录存在
    if not os.path.exists(PREDICTION_SAVE_DIR):
        os.makedirs(PREDICTION_SAVE_DIR)

    cal_miou(test_dir=TEST_IMAGES_DIR,
             pred_dir=PREDICTION_SAVE_DIR,
             gt_dir=GROUND_TRUTH_DIR,
             model_path=MODEL_PATH)