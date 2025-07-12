import cv2
import numpy as np
import os
import re

# --- 配置路径 ---
# 请根据你的实际情况修改 BASE_PATH
BASE_PATH = "E:/work/bishe/APCGAN-AttuNet-main/"  # <--- 修改这里为你机器上的实际路径

GROUND_TRUTH_AND_INPUT_DIR = os.path.join(BASE_PATH, "best_model/loss")

MODEL_ORDER_AND_PATHS = [
    ("MAFM", os.path.join(BASE_PATH, "best_model/loss/MAFM")),
    ("AttUnet", os.path.join(BASE_PATH, "best_model/loss/AttUnet")),
    ("AttUnet1", os.path.join(BASE_PATH, "best_model/loss/AttUnet1")),
    ("U2Net", os.path.join(BASE_PATH, "best_model/loss/U2Net")),
    ("Unet", os.path.join(BASE_PATH, "best_model/loss/Unet")),
    ("Unet++", os.path.join(BASE_PATH, "best_model/loss/Unet++")),
    ("Vnet", os.path.join(BASE_PATH, "best_model/loss/Vnet"))
]

EXISTING_MODELS = [(name, path) for name, path in MODEL_ORDER_AND_PATHS if os.path.isdir(path)]
if len(EXISTING_MODELS) != len(MODEL_ORDER_AND_PATHS):
    print("警告: 部分定义的模型文件夹未找到。")
    missing = set(name for name, path in MODEL_ORDER_AND_PATHS) - set(name for name, path in EXISTING_MODELS)
    print(f"缺失的模型文件夹对应的名称: {missing}")

OUTPUT_DIR = os.path.join(BASE_PATH, "comparison_results_fused_grid_with_legend")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 颜色定义 (BGR格式) ---
COLOR_TRUE_POSITIVE_YELLOW = (0, 255, 255)
COLOR_FALSE_POSITIVE_RED = (0, 0, 255)
COLOR_FALSE_NEGATIVE_GREEN = (0, 255, 0)
COLOR_BACKGROUND_BLACK = (0, 0, 0)
LEGEND_TEXT_COLOR = (0, 0, 0)
MODEL_NAME_TEXT_COLOR = (255, 255, 255)
MODEL_NAME_BG_COLOR = (50, 50, 50)  # 深灰色背景

# --- 字体和尺寸参数调整 ---
FONT = cv2.FONT_HERSHEY_SIMPLEX  # 保持字体一致性

# 模型名称参数
MODEL_NAME_HEIGHT = 70  # 增加模型名称区域高度
FONT_SCALE_MODEL_NAME = 1.0  # 放大模型名称字体
MODEL_NAME_FONT_THICKNESS = 2  # 加粗模型名称

# 图例参数
LEGEND_WIDTH = 280  # 增加图例区域宽度以容纳更大的文字
LEGEND_BG_COLOR = (255, 255, 255)
FONT_SCALE_LEGEND = 0.8  # 放大图例字体
LEGEND_FONT_THICKNESS = 2  # 加粗图例字体
LEGEND_COLOR_BOX_SIZE = 30  # 放大图例颜色块
LEGEND_LINE_SPACING = 60  # 增加图例行间距
LEGEND_TEXT_OFFSET_X = LEGEND_COLOR_BOX_SIZE + 20  # 颜色块与文字的间距
LEGEND_START_Y_OFFSET = 50  # 图例内容距离顶部的偏移


def load_and_binarize_mask(image_path):
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"警告: Mask 未找到或无法读取于 {image_path}")
        return None
    _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    return binary_mask


def create_fused_visualization(gt_mask_binary, pred_mask_binary, target_height, target_width):
    fused_image = np.full((target_height, target_width, 3), COLOR_BACKGROUND_BLACK, dtype=np.uint8)
    gt_mask_binary_resized = gt_mask_binary
    if gt_mask_binary.shape[0] != target_height or gt_mask_binary.shape[1] != target_width:
        gt_mask_binary_resized = cv2.resize(gt_mask_binary, (target_width, target_height),
                                            interpolation=cv2.INTER_NEAREST)
        gt_mask_binary_resized = (gt_mask_binary_resized > 0.5).astype(np.uint8)

    if pred_mask_binary is not None:
        pred_mask_binary_resized = pred_mask_binary
        if pred_mask_binary.shape[0] != target_height or pred_mask_binary.shape[1] != target_width:
            pred_mask_binary_resized = cv2.resize(pred_mask_binary, (target_width, target_height),
                                                  interpolation=cv2.INTER_NEAREST)
            pred_mask_binary_resized = (pred_mask_binary_resized > 0.5).astype(np.uint8)

        tp_pixels = np.logical_and(pred_mask_binary_resized == 1, gt_mask_binary_resized == 1)
        fp_pixels = np.logical_and(pred_mask_binary_resized == 1, gt_mask_binary_resized == 0)
        fn_pixels = np.logical_and(pred_mask_binary_resized == 0, gt_mask_binary_resized == 1)

        fused_image[tp_pixels] = COLOR_TRUE_POSITIVE_YELLOW
        fused_image[fp_pixels] = COLOR_FALSE_POSITIVE_RED
        fused_image[fn_pixels] = COLOR_FALSE_NEGATIVE_GREEN
    else:
        fn_pixels_gt_only = (gt_mask_binary_resized == 1)
        fused_image[fn_pixels_gt_only] = COLOR_FALSE_NEGATIVE_GREEN
        # "Pred N/A" 提示文字也相应调整
        text_size_na, _ = cv2.getTextSize("Pred N/A", FONT, FONT_SCALE_MODEL_NAME * 0.7, MODEL_NAME_FONT_THICKNESS)
        text_x_na = (target_width - text_size_na[0]) // 2
        text_y_na = (target_height + text_size_na[1]) // 2
        cv2.putText(fused_image, "Pred N/A", (text_x_na, text_y_na),
                    FONT, FONT_SCALE_MODEL_NAME * 0.7, MODEL_NAME_TEXT_COLOR,
                    MODEL_NAME_FONT_THICKNESS - 1 if MODEL_NAME_FONT_THICKNESS > 1 else 1)
    return fused_image


def create_legend_image(height):
    legend_img = np.full((height, LEGEND_WIDTH, 3), LEGEND_BG_COLOR, dtype=np.uint8)

    start_y = LEGEND_START_Y_OFFSET

    # 1. False Negative (Green) - Label
    cv2.rectangle(legend_img, (20, start_y), (20 + LEGEND_COLOR_BOX_SIZE, start_y + LEGEND_COLOR_BOX_SIZE),
                  COLOR_FALSE_NEGATIVE_GREEN, -1)
    cv2.putText(legend_img, "Label (FN)",
                (20 + LEGEND_TEXT_OFFSET_X, start_y + LEGEND_COLOR_BOX_SIZE - (LEGEND_COLOR_BOX_SIZE // 4)), FONT,
                FONT_SCALE_LEGEND, LEGEND_TEXT_COLOR, LEGEND_FONT_THICKNESS)

    # 2. True Positive (Yellow) - Correct Prediction
    start_y += LEGEND_LINE_SPACING
    cv2.rectangle(legend_img, (20, start_y), (20 + LEGEND_COLOR_BOX_SIZE, start_y + LEGEND_COLOR_BOX_SIZE),
                  COLOR_TRUE_POSITIVE_YELLOW, -1)
    cv2.putText(legend_img, "Correct Prediction (TP)",
                (20 + LEGEND_TEXT_OFFSET_X, start_y + LEGEND_COLOR_BOX_SIZE - (LEGEND_COLOR_BOX_SIZE // 4)), FONT,
                FONT_SCALE_LEGEND, LEGEND_TEXT_COLOR, LEGEND_FONT_THICKNESS)

    # 3. False Positive (Red) - Incorrect Prediction
    start_y += LEGEND_LINE_SPACING
    cv2.rectangle(legend_img, (20, start_y), (20 + LEGEND_COLOR_BOX_SIZE, start_y + LEGEND_COLOR_BOX_SIZE),
                  COLOR_FALSE_POSITIVE_RED, -1)
    cv2.putText(legend_img, "Incorrect Prediction (FP)",
                (20 + LEGEND_TEXT_OFFSET_X, start_y + LEGEND_COLOR_BOX_SIZE - (LEGEND_COLOR_BOX_SIZE // 4)), FONT,
                FONT_SCALE_LEGEND, LEGEND_TEXT_COLOR, LEGEND_FONT_THICKNESS)

    return legend_img


# --- 主处理逻辑 ---
all_gt_mask_files = sorted([f for f in os.listdir(GROUND_TRUTH_AND_INPUT_DIR) if f.endswith(".png")])
files_to_process_gt_masks = [
    "benign (10).png",
    "malignant93.png",
    "malignant150.png",
    "normal (60).png"
]
files_to_process_gt_masks = [f for f in files_to_process_gt_masks if f in all_gt_mask_files]

if not files_to_process_gt_masks:
    print(f"错误: 指定的GT Mask文件未找到。请检查 'files_to_process_gt_masks' 和 {GROUND_TRUTH_AND_INPUT_DIR} 目录。")
    exit()
print(f"将处理以下GT Mask文件: {files_to_process_gt_masks}")

all_rows_of_images = []
target_thumb_height, target_thumb_width = 0, 0

for i, gt_mask_filename in enumerate(files_to_process_gt_masks):
    input_image_filename = gt_mask_filename.replace(".png", ".jpg")
    input_image_path = os.path.join(GROUND_TRUTH_AND_INPUT_DIR, input_image_filename)
    gt_mask_path = os.path.join(GROUND_TRUTH_AND_INPUT_DIR, gt_mask_filename)

    input_image_bgr = cv2.imread(input_image_path)
    if input_image_bgr is None:
        print(f"警告: 原始输入图像未找到: {input_image_path}. 跳过.")
        continue

    if i == 0:
        # target_thumb_height, target_thumb_width = 256, 256 # 可选固定尺寸
        target_thumb_height, target_thumb_width = input_image_bgr.shape[0], input_image_bgr.shape[1]
        print(f"缩略图基准尺寸: H={target_thumb_height}, W={target_thumb_width}")

    gt_mask_binary = load_and_binarize_mask(gt_mask_path)
    if gt_mask_binary is None:
        print(f"警告: GT Mask加载失败: {gt_mask_path}. 跳过.")
        continue

    current_row_images = []
    display_input_resized = cv2.resize(input_image_bgr, (target_thumb_width, target_thumb_height))
    current_row_images.append(display_input_resized)

    for model_name, model_folder_path in EXISTING_MODELS:
        predicted_mask_path = os.path.join(model_folder_path, gt_mask_filename)
        pred_mask_binary = None
        if os.path.exists(predicted_mask_path):
            pred_mask_binary = load_and_binarize_mask(predicted_mask_path)
        else:
            print(f"信息: 模型 '{model_name}' 的预测mask未找到: {predicted_mask_path}")

        fused_viz = create_fused_visualization(gt_mask_binary, pred_mask_binary, target_thumb_height,
                                               target_thumb_width)
        current_row_images.append(fused_viz)

    if current_row_images:
        try:
            row_concatenated = np.hstack(tuple(current_row_images))
            all_rows_of_images.append(row_concatenated)
        except Exception as e:
            print(f"错误: 水平拼接图像失败 for {gt_mask_filename}: {e}")

if not all_rows_of_images:
    print("错误: 没有成功处理任何图像行。")
    exit()

# 创建模型名称行
num_cols_main_grid = 1 + len(EXISTING_MODELS)
main_grid_width = num_cols_main_grid * target_thumb_width
model_names_row_img = np.full((MODEL_NAME_HEIGHT, main_grid_width, 3), MODEL_NAME_BG_COLOR, dtype=np.uint8)
column_names = ["Input"] + [name for name, _ in EXISTING_MODELS]

for col_idx, col_name in enumerate(column_names):
    if col_idx < num_cols_main_grid:
        text_size, _ = cv2.getTextSize(col_name, FONT, FONT_SCALE_MODEL_NAME, MODEL_NAME_FONT_THICKNESS)
        text_x = (col_idx * target_thumb_width) + (target_thumb_width - text_size[0]) // 2
        text_y = (MODEL_NAME_HEIGHT + text_size[1]) // 2
        cv2.putText(model_names_row_img, col_name, (text_x, text_y), FONT, FONT_SCALE_MODEL_NAME, MODEL_NAME_TEXT_COLOR,
                    MODEL_NAME_FONT_THICKNESS)

try:
    main_grid_body = np.vstack(tuple(all_rows_of_images))
    main_grid_with_names = np.vstack((main_grid_body, model_names_row_img))
except Exception as e:
    print(f"错误: 垂直拼接主网格失败: {e}")
    exit()

legend_height = main_grid_with_names.shape[0]
legend_image = create_legend_image(legend_height)

try:
    if main_grid_with_names.shape[0] != legend_image.shape[0]:
        # 如果图例高度不匹配（不太可能发生，因为是基于主网格高度创建的），重新创建
        legend_image = create_legend_image(main_grid_with_names.shape[0])

    final_image_with_legend = np.hstack((main_grid_with_names, legend_image))
    output_filepath = os.path.join(OUTPUT_DIR, "final_comparison_grid_with_legend_larger_text.png")
    cv2.imwrite(output_filepath, final_image_with_legend)
    print(f"最终拼接图已保存 (含图例和大号文字): {output_filepath}")
except Exception as e:
    print(f"错误: 拼接主网格与图例失败: {e}")

print("所有处理完成。")