#!/bin/bash
set -e

NUM_GPUS=8

DATA_DIR="/data/ReaSeg/MedSAM/data"

CHECKPOINT="/data/ReaSeg/MedSAM/pretrained/medsam_vit_b.pth"
SAVE_DIR="./MedSAM_Finetuned"

BATCH_SIZE=8
EPOCHS=50
LEARNING_RATE=1e-4

FREEZE_IMAGE_ENCODER=1   
FREEZE_PROMPT_ENCODER=1
FREEZE_MASK_DECODER=0

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

echo "🚀 Start executing the training script..."
echo "GPU_num: $NUM_GPUS, Data_path: $DATA_DIR, Freeze_args: $FREEZE_ARGS"

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
