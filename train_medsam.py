import os
import cv2
import glob
import time
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

import monai
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# ==========================================
# 1. 智能数据集定义 (支持单/多数据集自动发现与在线预处理)
# ==========================================
class MedSAMOnlineDataset(Dataset):
    def __init__(self, data_dir, split="train", img_size=1024, bbox_shift=5):
        """
        data_dir: 基础数据路径
        split: "train" 或 "val"
        """
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.bbox_shift = bbox_shift
        self.sam_transform = ResizeLongestSide(img_size)
        self.image_paths = []

        # 智能路径解析逻辑
        # 1. 判断是否为单数据集 (即 data_dir 下直接存在 train/val 文件夹)
        if os.path.exists(os.path.join(data_dir, split, 'images')):
            img_dir = os.path.join(data_dir, split, 'images')
            self.image_paths.extend(glob.glob(os.path.join(img_dir, '*.*')))
            dataset_type = "单数据集"
        # 2. 判断是否为多数据集集合 (即 data_dir 下包含 Skin, BraTS 等子文件夹)
        else:
            for subdir in os.listdir(data_dir):
                dataset_path = os.path.join(data_dir, subdir)
                if os.path.isdir(dataset_path):
                    img_dir = os.path.join(dataset_path, split, 'images')
                    if os.path.exists(img_dir):
                        self.image_paths.extend(glob.glob(os.path.join(img_dir, '*.*')))
            dataset_type = "多数据集"

        # 过滤有效的图片后缀
        self.image_paths = [p for p in self.image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"[Dataset - {split.upper()}] 模式: {dataset_type} | 共找到 {len(self.image_paths)} 张图像。")

    def __len__(self):
        return len(self.image_paths)

    def get_bbox_from_mask(self, mask):
        y_indices, x_indices = np.where(mask > 0)
        H, W = mask.shape
        if len(y_indices) == 0 or len(x_indices) == 0:
            return np.array([0, 0, W, H])
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min = max(0, x_min - np.random.randint(0, self.bbox_shift + 1))
        x_max = min(W, x_max + np.random.randint(0, self.bbox_shift + 1))
        y_min = max(0, y_min - np.random.randint(0, self.bbox_shift + 1))
        y_max = min(H, y_max + np.random.randint(0, self.bbox_shift + 1))
        return np.array([x_min, y_min, x_max, y_max])

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 动态推导 mask 路径 (将路径中的 '/images' 替换为 '/masks')
        parent_dir = os.path.dirname(img_path)
        mask_dir = parent_dir.replace(f'{os.sep}images', f'{os.sep}masks')
        
        img_name = os.path.basename(img_path)
        img_stem = os.path.splitext(img_name)[0]
        
        mask_path = os.path.join(mask_dir, img_stem + '.png')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, img_name)
            
        # 读取图像和标签
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_size = image.shape[:2]
        
        # 提取 BBox 并缩放
        bbox_original = self.get_bbox_from_mask(mask_gt)
        box_scaled = self.sam_transform.apply_boxes(bbox_original, original_size)
        
        # 图像缩放与转Tensor
        input_image = self.sam_transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1).contiguous()
        
        # Mask 缩放与转Tensor (最近邻插值)
        input_mask = self.sam_transform.apply_image(mask_gt)
        input_mask_torch = torch.as_tensor(input_mask).contiguous()

        # 统一填充到 1024x1024
        h, w = input_image_torch.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        image_padded = F.pad(input_image_torch, (0, padw, 0, padh))
        mask_padded = F.pad(input_mask_torch, (0, padw, 0, padh))
        
        # Mask 归一化为 0 和 1
        mask_padded = (mask_padded > 0).float().unsqueeze(0) 

        return {
            "image": image_padded.float(),
            "box": torch.as_tensor(box_scaled, dtype=torch.float),
            "mask": mask_padded
        }

# ==========================================
# 2. 参数统计与打印输出
# ==========================================
def print_parameter_statistics(model, local_rank):
    if local_rank != 0:
        return
        
    def get_stats(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        ratio = (trainable / total * 100) if total > 0 else 0.0
        return total, trainable, ratio

    total_all, train_all, ratio_all = get_stats(model)
    total_img, train_img, ratio_img = get_stats(model.image_encoder)
    total_prm, train_prm, ratio_prm = get_stats(model.prompt_encoder)
    total_msk, train_msk, ratio_msk = get_stats(model.mask_decoder)

    print("\n" + "="*85)
    print(f"{'组件 (Component)':<25} | {'参数总量 (Total)':<18} | {'训练参数量 (Trainable)':<18} | {'训练占比 (%)':<10}")
    print("-" * 85)
    print(f"{'MedSAM (整体模型)':<25} | {total_all:<18,} | {train_all:<18,} | {ratio_all:.2f}%")
    print(f"{'  ├─ Image Encoder':<24} | {total_img:<18,} | {train_img:<18,} | {ratio_img:.2f}%")
    print(f"{'  ├─ Prompt Encoder':<24} | {total_prm:<18,} | {train_prm:<18,} | {ratio_prm:.2f}%")
    print(f"{'  └─ Mask Decoder':<24} | {total_msk:<18,} | {train_msk:<18,} | {ratio_msk:.2f}%")
    print("="*85 + "\n")


# ==========================================
# 3. 训练与验证主循环
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='数据集根路径 (单数据集或多数据集均可)')
    parser.add_argument('--save_dir', type=str, default='./work_dir/MedSAM_Finetuned', help='模型保存路径')
    parser.add_argument('--checkpoint', type=str, default='./work_dir/MedSAM/medsam_vit_b.pth', help='预训练权重')
    
    # 超参数
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    # 冻结控制参数
    parser.add_argument('--freeze_image_encoder', action='store_true', help='是否冻结 Image Encoder (推荐开启)')
    parser.add_argument('--freeze_prompt_encoder', action='store_true', help='是否冻结 Prompt Encoder')
    parser.add_argument('--freeze_mask_decoder', action='store_true', help='是否冻结 Mask Decoder')
    args = parser.parse_args()

    # DDP 分布式环境初始化
    is_distributed = "WORLD_SIZE" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if local_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"✅ 启动任务: 分布式={is_distributed}, GPU数量={os.environ.get('WORLD_SIZE', 1)}")

    # 加载模型
    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    
    if args.freeze_image_encoder:
        for param in model.image_encoder.parameters(): param.requires_grad = False
    if args.freeze_prompt_encoder:
        for param in model.prompt_encoder.parameters(): param.requires_grad = False
    if args.freeze_mask_decoder:
        for param in model.mask_decoder.parameters(): param.requires_grad = False

    print_parameter_statistics(model, local_rank)

    model = model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    sam_model = model.module if is_distributed else model

    # ================= 数据加载 (训练集与验证集) =================
    train_dataset = MedSAMOnlineDataset(args.data_dir, split="train")
    val_dataset = MedSAMOnlineDataset(args.data_dir, split="val")
    
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle=False, num_workers=4, pin_memory=True)

    # 优化器与损失函数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("所有组件都被冻结了！没有可训练的参数！")
        
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    seg_loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # ================= 主循环 =================
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        if is_distributed: train_sampler.set_epoch(epoch)
        
        # ----------------- 训练阶段 -----------------
        model.train()
        train_loss = 0.0
        pbar_train = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", disable=(local_rank != 0))
        
        for batch_data in pbar_train:
            images = batch_data["image"].to(device)
            boxes = batch_data["box"].to(device)
            gt_masks = batch_data["mask"].to(device)

            images = sam_model.preprocess(images) 

            with torch.set_grad_enabled(not args.freeze_image_encoder):
                image_embeddings = sam_model.image_encoder(images)
            with torch.set_grad_enabled(not args.freeze_prompt_encoder):
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None, boxes=boxes.unsqueeze(1), masks=None,
                )
            with torch.set_grad_enabled(not args.freeze_mask_decoder):
                low_res_masks, _ = sam_model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
            
            pred_masks = F.interpolate(low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False)
            loss = seg_loss_fn(pred_masks, gt_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if local_rank == 0:
                pbar_train.set_postfix({"Loss": f"{loss.item():.4f}"})

        # ----------------- 验证阶段 -----------------
        model.eval()
        val_loss = 0.0
        pbar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", disable=(local_rank != 0))
        
        with torch.no_grad():
            for batch_data in pbar_val:
                images = batch_data["image"].to(device)
                boxes = batch_data["box"].to(device)
                gt_masks = batch_data["mask"].to(device)

                images = sam_model.preprocess(images)
                image_embeddings = sam_model.image_encoder(images)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None, boxes=boxes.unsqueeze(1), masks=None,
                )
                low_res_masks, _ = sam_model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                pred_masks = F.interpolate(low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False)
                loss = seg_loss_fn(pred_masks, gt_masks)
                val_loss += loss.item()
                if local_rank == 0:
                    pbar_val.set_postfix({"Loss": f"{loss.item():.4f}"})

        # --- 跨多卡汇总验证Loss (确保精确度) ---
        avg_val_loss = val_loss / len(val_dataloader)
        if is_distributed:
            loss_tensor = torch.tensor([avg_val_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = loss_tensor.item() / int(os.environ["WORLD_SIZE"])

        # ================= Epoch 结束，模型保存 =================
        if local_rank == 0:
            avg_train_loss = train_loss / len(train_dataloader)
            print(f"🎯 Epoch {epoch+1} 结束 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            # 保存 Latest
            torch.save(sam_model.state_dict(), os.path.join(args.save_dir, "medsam_latest.pth"))
            # 根据 Val Loss 保存 Best Model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(sam_model.state_dict(), os.path.join(args.save_dir, "medsam_best.pth"))
                print(f"🌟 发现最佳模型 (Val Loss: {best_val_loss:.4f})，已保存！")

    if is_distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()