#!/bin/bash
# 遇到错误即停止运行
set -e

# ==========================================
# ⚙️ 核心参数配置区 (在此处修改你的设置)
# ==========================================

# 1. 硬件配置
NUM_GPUS=8  # 你想要使用的 GPU 数量 (例如设为 1 表示单卡，2 表示双卡)

# 2. 路径配置 (⚠️ 核心修改点)
# 玩法A：多数据集训练 -> 指定到 /data 即可
# DATA_DIR="/data/ReaSeg/MedSAM/data"
# 玩法B：单数据集训练 -> 指定到具体文件夹即可
DATA_DIR="/data/ReaSeg/MedSAM/data"

CHECKPOINT="/data/ReaSeg/MedSAM/pretrained/medsam_vit_b.pth"
SAVE_DIR="./MedSAM_Finetuned"

# 3. 训练超参数
BATCH_SIZE=8
EPOCHS=50
LEARNING_RATE=1e-4

# 4. 冻结策略控制 (1=冻结，0=参与训练)
FREEZE_IMAGE_ENCODER=1   
FREEZE_PROMPT_ENCODER=1
FREEZE_MASK_DECODER=0

# ==========================================
# 🚀 组装参数并启动训练 (无需修改下方代码)
# ==========================================

# 组装冻结参数
FREEZE_ARGS=""
if [ "$FREEZE_IMAGE_ENCODER" -eq 1 ]; then
    FREEZE_ARGS="$FREEZE_ARGS --freeze_image_encoder"
fi
if [ "$FREEZE_PROMPT_ENCODER" -eq 1 ]; then
    FREEZE_ARGS="$FREEZE_ARGS --freeze_prompt_encoder"
fi
if [ "$FREEZE_MASK_DECODER" -eq 1 ]; then
    FREEZE_ARGS="$FREEZE_ARGS --freeze_mask_decoder"
fi

echo "🚀 开始执行训练脚本..."
echo "卡数: $NUM_GPUS, 数据路径: $DATA_DIR, 冻结参数: $FREEZE_ARGS"

# 使用 PyTorch 官方支持的 torchrun 启动脚本
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    train_medsam.py \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --checkpoint "$CHECKPOINT" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    $FREEZE_ARGS