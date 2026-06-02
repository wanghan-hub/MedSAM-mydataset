import os
import cv2
import numpy as np
import torch
import glob
import csv
from tqdm import tqdm
import argparse
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

def get_bbox_from_mask(mask, bbox_shift=5):
    
    y_indices, x_indices = np.where(mask > 0)
    H, W = mask.shape
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return np.array([0, 0, W, H])
        
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    x_min = max(0, x_min - np.random.randint(0, bbox_shift + 1))
    x_max = min(W, x_max + np.random.randint(0, bbox_shift + 1))
    y_min = max(0, y_min - np.random.randint(0, bbox_shift + 1))
    y_max = min(H, y_max + np.random.randint(0, bbox_shift + 1))
    
    return np.array([x_min, y_min, x_max, y_max])

def compute_dice(mask_gt, mask_pred):
    mask_gt = (mask_gt > 0).astype(np.float32)
    mask_pred = (mask_pred > 0).astype(np.float32)
    intersection = (mask_gt * mask_pred).sum()
    return (2. * intersection) / (mask_gt.sum() + mask_pred.sum() + 1e-8)

def compute_iou(mask_gt, mask_pred):
    mask_gt = (mask_gt > 0).astype(np.float32)
    mask_pred = (mask_pred > 0).astype(np.float32)
    intersection = (mask_gt * mask_pred).sum()
    union = mask_gt.sum() + mask_pred.sum() - intersection
    if union == 0:
        return 1.0 if mask_gt.sum() == 0 and mask_pred.sum() == 0 else 0.0
    return intersection / union

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-image_dir', type=str, default='/data/ReaSeg/MedSAM/data/Skin/test/images', help='train set image path')
    parser.add_argument('-mask_dir', type=str, default='/data/ReaSeg/MedSAM/data/Skin/test/masks', help='test set image path')
    parser.add_argument('-save_dir', type=str, default='./test_results', help='Prediction result storage path')
    parser.add_argument('-checkpoint', type=str, default='./work_dir/MedSAM/medsam_vit_b.pth', help='MedSAM checkpoint')
    parser.add_argument('-device', type=str, default='cuda:0', help='device')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading the MedSAM model: {args.checkpoint} ...")
    medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    medsam_model = medsam_model.to(args.device)
    medsam_model.eval()

    sam_transform = ResizeLongestSide(medsam_model.image_encoder.img_size)

    image_paths = glob.glob(os.path.join(args.image_dir, '*.*'))
    image_paths = [p for p in image_paths if p.endswith(('.png', '.jpg', '.jpeg'))]
    
    dice_scores = []
    iou_scores = []
    csv_data = []

    print(f"A total of {len(image_paths)} test images were found. Now, the inference and evaluation process begins...")
    
    for img_path in tqdm(image_paths):
        img_name = os.path.basename(img_path)
        img_stem = os.path.splitext(img_name)[0]
        
        mask_path = os.path.join(args.mask_dir, img_stem + '.png')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(args.mask_dir, img_name) # 尝试原后缀
        if not os.path.exists(mask_path):
            print(f"⚠️ Could not find the corresponding mask for {img_name}, skipping.")
            continue

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2] # (H, W)

        mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        bbox_original = get_bbox_from_mask(mask_gt)

        input_image = sam_transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=args.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :] # (1, 3, H_resized, W_resized)
        
        box_torch = sam_transform.apply_boxes(bbox_original, original_size)
        box_torch = torch.as_tensor(box_torch, dtype=torch.float, device=args.device)

        with torch.no_grad():
            input_image_padded = medsam_model.preprocess(input_image_torch.squeeze(0)).unsqueeze(0)

            image_embedding = medsam_model.image_encoder(input_image_padded)
            sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            low_res_masks, iou_predictions = medsam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=medsam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            input_size = (input_image_torch.shape[2], input_image_torch.shape[3])
            pred_mask_original_size = medsam_model.postprocess_masks(
                low_res_masks,
                input_size=input_size,
                original_size=original_size, 
            )

            pred_mask = torch.sigmoid(pred_mask_original_size)
            pred_mask = (pred_mask > 0.5).squeeze().cpu().numpy().astype(np.uint8)

        dice = compute_dice(mask_gt, pred_mask)
        iou = compute_iou(mask_gt, pred_mask)
        
        dice_scores.append(dice)
        iou_scores.append(iou)
        csv_data.append([img_name, round(dice, 4), round(iou, 4)])

        pred_save_path = os.path.join(args.save_dir, f"{img_stem}_pred.png")
        cv2.imwrite(pred_save_path, pred_mask * 255)

    if len(dice_scores) > 0:
        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(iou_scores) 
        
        csv_data.append(['Mean', round(mean_dice, 4), round(mean_iou, 4)])
        
        csv_file_path = os.path.join(args.save_dir, 'evaluation_metrics.csv')
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Image_Name', 'Dice', 'IoU'])
            writer.writerows(csv_data)
            
        print(f"\n✅ Reasoning assessment completed！")
        print(f"📊 Overall average Dice of the dataset: {mean_dice:.4f}")
        print(f"📊 Overall average mIoU of the dataset: {mean_iou:.4f}")
        print(f"📄 The indicator has been saved as CSV.: {csv_file_path}")
        print(f"📁 The predicted result image has been saved in: {args.save_dir}")
    else:
        print("No valid image and mask pairs were found.")

if __name__ == '__main__':
    main()
