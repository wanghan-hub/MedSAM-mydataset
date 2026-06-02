# MedSAM 微调与推理评估代码库 (MedSAM Fine-tuning & Evaluation)

本项目提供了一套完整、易用且高效的代码，用于微调（Fine-tune）和评估 **MedSAM**（Medical Segment Anything Model）模型。代码支持单卡/多卡分布式训练（DDP）、灵活的参数冻结策略、智能数据集结构识别以及基于真实场景模拟的在线预处理。

## ✨ 核心特性

- **🚀 分布式多卡训练 (DDP)**: 采用 `torchrun` 启动，原生支持多 GPU 加速，跨卡同步计算 Validation Loss 确保精度。
- **🧠 灵活的冻结策略**: 训练脚本中可以灵活配置是否冻结 `Image Encoder`、`Prompt Encoder` 或 `Mask Decoder`，有效控制显存占用并适配不同的微调需求。
- **🔄 在线预处理与 BBox 模拟**: 
  - 动态对图像进行 1024x1024 的 Padding 与缩放。
  - **真实的提示模拟**: 从 Ground Truth Mask 提取 Bounding Box 时，自动加入 `0~5` 像素的随机扰动（`bbox_shift`），模拟真实临床医生的框选误差。

---

## 🛠️ 环境依赖

推荐使用 Python 3.10+ 和 PyTorch 2.0+ 环境。

```bash
pip install torch torchvision monai opencv-python tqdm scipy
```
*(注：项目已内置定制版 segment_anything 核心库，无需额外通过 pip 安装原版 SAM)*

---

## 📁 数据集准备

代码支持以下两种灵活的数据集目录结构（程序会自动识别，无需手动修改代码）：

### 选项 A：单数据集模式
```text
data_dir/
 ├── train/
 │    ├── images/ (存放 .png, .jpg)
 │    └── masks/  (存放对应的 .png 标签，必须为灰度图)
 └── val/
      ├── images/
      └── masks/
```

### 选项 B：多数据集混合模式
适合将多个器官或不同来源的数据混合训练。
```text
data_dir/
 ├── Skin/
 │    ├── train/
 │    │    ├── images/
 │    │    └── masks/
 │    └── val/...
 ├── Poly/
 │    ├── train/...
 │    └── val/...
```
> **⚠️ 注意**：Mask 图像必须是灰度图（背景为 0，前景为 >0 的值）。图像和标签的名称需保持一致（或同名不同后缀）。

---

## 🚀 模型训练 (Training)

训练过程由 `train_medsam.sh` 和 `train_medsam.py` 驱动。

### 1. 修改启动脚本配置
打开 `train_medsam.sh`，根据您的环境修改核心参数：

```bash
# 1. 硬件配置
NUM_GPUS=8  # 参与训练的 GPU 数量

# 2. 路径配置
DATA_DIR="/path/to/your/dataset"                  # 数据集根路径
CHECKPOINT="/path/to/pretrained/medsam_vit_b.pth" # 预训练权重路径
SAVE_DIR="./MedSAM_Finetuned"                     # 模型保存路径

# 3. 训练超参数
BATCH_SIZE=8
EPOCHS=50
LEARNING_RATE=1e-4

# 4. 冻结策略控制 (1=冻结，0=参与训练)
FREEZE_IMAGE_ENCODER=1   # 推荐冻结 Image Encoder 以节省显存
FREEZE_PROMPT_ENCODER=1
FREEZE_MASK_DECODER=0    # 仅微调 Mask Decoder
```

### 2. 启动训练
运行 shell 脚本即可启动 DDP 分布式训练：

```bash
bash train_medsam.sh
```
*(训练时，终端会输出参数总量及可训练参数占比。每个 Epoch 结束后保存 medsam_latest.pth，验证集 Loss 达到最低时保存 medsam_best.pth)*

---

## 🧪 推理与评估 (Evaluation)

评估代码位于 `eval_medsam.py`，通过 `eval_medsam.sh` 启动。将在测试集上模拟 BBox 提示，并输出 Dice 和 IoU 评分。

### 1. 修改评估脚本配置
打开 `eval_medsam.sh`，更新测试集路径和测试权重路径：

```bash
python eval_medsam.py \
    -image_dir /path/to/test/images \
    -mask_dir /path/to/test/masks \
    -checkpoint ./MedSAM_Finetuned/medsam_best.pth \
    -save_dir ./output/Test_Results
```

### 2. 启动评估
```bash
bash eval_medsam.sh
```

### 3. 评估输出结果
运行结束后，指定的 `-save_dir` 目录下会生成：
1. **预测掩码图片**：`*_pred.png`（与原图对照，方便可视化检查）。
2. **评估报告**：`evaluation_metrics.csv`（包含每张图片的 Dice、IoU 评分，以及最后一行的整体平均指标）。

---

## 📂 核心代码结构说明

```text
MedSAM/
 ├── train_medsam.sh / py       # 分布式训练入口与主逻辑，包含智能 Dataset 类设计
 ├── eval_medsam.sh / py        # 推理评估脚本，包含 Prompt 生成与预处理/后处理逻辑
 ├── segment_anything/          # SAM 核心模型代码（已根据 MedSAM 进行适配）
 │    ├── modeling/             # 包含 ViT (Image Encoder)、Mask Decoder 等核心网络结构
 │    ├── build_sam.py          # 模型构建工厂函数
 │    └── utils/transforms.py   # 包含 1024 尺寸对齐与 Longest Side Resize 工具
 └── utils/
      └── SurfaceDice.py        # 提供边缘距离、Surface Dice、Hausdorff 距离计算的备用扩展库
```
