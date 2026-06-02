import os
import cv2
import numpy as np
import torch
import glob
import csv  # 新增：用于保存指标到CSV
from tqdm import tqdm
import argparse

# 导入 SAM 核心组件
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

def get_bbox_from_mask(mask, bbox_shift=5):
    """
    从 GT Mask 中提取边界框，并加入随机扰动以模拟真实的人类提示
    """
    y_indices, x_indices = np.where(mask > 0)
    H, W = mask.shape
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return np.array([0, 0, W, H]) # 如果遇到全黑mask的兜底策略
        
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # 模拟医生的框选误差，向外随机扩张 0~bbox_shift 个像素
    x_min = max(0, x_min - np.random.randint(0, bbox_shift + 1))
    x_max = min(W, x_max + np.random.randint(0, bbox_shift + 1))
    y_min = max(0, y_min - np.random.randint(0, bbox_shift + 1))
    y_max = min(H, y_max + np.random.randint(0, bbox_shift + 1))
    
    return np.array([x_min, y_min, x_max, y_max])

def compute_dice(mask_gt, mask_pred):
    """计算 Dice 系数"""
    mask_gt = (mask_gt > 0).astype(np.float32)
    mask_pred = (mask_pred > 0).astype(np.float32)
    intersection = (mask_gt * mask_pred).sum()
    return (2. * intersection) / (mask_gt.sum() + mask_pred.sum() + 1e-8)

def compute_iou(mask_gt, mask_pred):
    """新增：计算 IoU (Intersection over Union)"""
    mask_gt = (mask_gt > 0).astype(np.float32)
    mask_pred = (mask_pred > 0).astype(np.float32)
    intersection = (mask_gt * mask_pred).sum()
    union = mask_gt.sum() + mask_pred.sum() - intersection
    # 避免分母为0的情况
    if union == 0:
        return 1.0 if mask_gt.sum() == 0 and mask_pred.sum() == 0 else 0.0
    return intersection / union

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-image_dir', type=str, default='/data/ReaSeg/MedSAM/data/Skin/test/images', help='测试集图片路径')
    parser.add_argument('-mask_dir', type=str, default='/data/ReaSeg/MedSAM/data/Skin/test/masks', help='测试集标签路径')
    parser.add_argument('-save_dir', type=str, default='./test_results', help='预测结果保存路径')
    parser.add_argument('-checkpoint', type=str, default='./work_dir/MedSAM/medsam_vit_b.pth', help='MedSAM 权重路径')
    parser.add_argument('-device', type=str, default='cuda:0', help='运行设备')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 加载 MedSAM 模型
    print(f"正在加载 MedSAM 模型: {args.checkpoint} ...")
    medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    medsam_model = medsam_model.to(args.device)
    medsam_model.eval()

    # 初始化图像尺寸转换器 (目标 1024)
    sam_transform = ResizeLongestSide(medsam_model.image_encoder.img_size)

    # 2. 获取数据集文件列表 (支持 png, jpg)
    image_paths = glob.glob(os.path.join(args.image_dir, '*.*'))
    image_paths = [p for p in image_paths if p.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 初始化指标列表和CSV数据结构
    dice_scores = []
    iou_scores = []
    csv_data = [] # 用于保存要写入CSV的行

    print(f"共找到 {len(image_paths)} 张测试图片，开始推理评估...")
    
    for img_path in tqdm(image_paths):
        img_name = os.path.basename(img_path)
        img_stem = os.path.splitext(img_name)[0]
        
        # 寻找对应的 Mask (假设 Mask 统一为 png 格式，或者与图片同后缀)
        mask_path = os.path.join(args.mask_dir, img_stem + '.png')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(args.mask_dir, img_name) # 尝试原后缀
        if not os.path.exists(mask_path):
            print(f"⚠️ 找不到 {img_name} 对应的 mask，跳过。")
            continue

        # 3. 读取图像和 Mask (保持原尺寸)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转为 RGB
        original_size = image.shape[:2] # (H, W)

        mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 4. 生成 Prompt Bounding Box (基于原图尺寸)
        bbox_original = get_bbox_from_mask(mask_gt)

        # 5. 在线预处理 (On-the-fly Preprocessing)
        # 5.1 图像：等比例缩放
        input_image = sam_transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=args.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :] # (1, 3, H_resized, W_resized)
        
        # 5.2 Prompt：同样按照图片的缩放比例缩放 Bounding Box
        box_torch = sam_transform.apply_boxes(bbox_original, original_size)
        box_torch = torch.as_tensor(box_torch, dtype=torch.float, device=args.device)

        with torch.no_grad():
            # 5.3 调用 SAM 自带的 preprocess 进行 1024 填充 (Padding) 和 归一化 (Normalization)
            input_image_padded = medsam_model.preprocess(input_image_torch.squeeze(0)).unsqueeze(0)

            # 6. 网络前向传播 (Forward)
            # 编码图像
            image_embedding = medsam_model.image_encoder(input_image_padded)
            # 编码 Prompt
            sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            # 解码 Mask
            low_res_masks, iou_predictions = medsam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=medsam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # 7. 后处理：自动去黑边 (Un-pad) 并反向缩放 (Un-resize) 回原图尺寸
            input_size = (input_image_torch.shape[2], input_image_torch.shape[3]) # 缩放后、填充前的大小
            pred_mask_original_size = medsam_model.postprocess_masks(
                low_res_masks,
                input_size=input_size,
                original_size=original_size, 
            )

            # 8. 获取最终二值化预测掩码
            pred_mask = torch.sigmoid(pred_mask_original_size)
            pred_mask = (pred_mask > 0.5).squeeze().cpu().numpy().astype(np.uint8)

        # 9. 计算指标：Dice 和 IoU
        dice = compute_dice(mask_gt, pred_mask)
        iou = compute_iou(mask_gt, pred_mask)
        
        # 将当前图片指标加入列表
        dice_scores.append(dice)
        iou_scores.append(iou)
        csv_data.append([img_name, round(dice, 4), round(iou, 4)])

        # 保存预测结果用于可视化对照
        pred_save_path = os.path.join(args.save_dir, f"{img_stem}_pred.png")
        cv2.imwrite(pred_save_path, pred_mask * 255)

    # 10. 输出最终评估结果并保存为 CSV
    if len(dice_scores) > 0:
        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(iou_scores) # 计算 mIoU
        
        # 将总体均值添加到 CSV 最后一行
        csv_data.append(['Mean', round(mean_dice, 4), round(mean_iou, 4)])
        
        # 写入 CSV 文件
        csv_file_path = os.path.join(args.save_dir, 'evaluation_metrics.csv')
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Image_Name', 'Dice', 'IoU']) # 写入表头
            writer.writerows(csv_data)                     # 写入数据
            
        print(f"\n✅ 推理评估完成！")
        print(f"📊 数据集总体平均 Dice: {mean_dice:.4f}")
        print(f"📊 数据集总体平均 mIoU: {mean_iou:.4f}")
        print(f"📄 指标已保存至 CSV: {csv_file_path}")
        print(f"📁 预测结果图片已保存在: {args.save_dir}")
    else:
        print("未找到有效的图像和掩码对。")

if __name__ == '__main__':
    main()