# GFIN

**基于全局上下文与成对融合特征的人-物交互检测**

GFIN 检测并分类图像中的人-物交互 (如 "人骑自行车"). 它将冻结的 DETR 目标检测器与自定义交互头结合, 交互头包含三个核心模块:

- **GCE** (全局上下文编码器): 通过 Transformer 编码器编码图像级上下文
- **HOF** (人-物融合): 通过多头成对注意力融合外观与空间特征
- **PID** (成对交互解码器): 交叉注意力解码器, 用于 HOI 分类

**v2 改进** (可选功能, 通过命令行参数启用):
- **CLIP 文本先验**: 使用 CLIP 语义特征初始化分类器权重 (`--clip-init`)
- **CLIP 视觉融合**: 将 CLIP 全局视觉特征融合到 GCE (`--clip-visual`)
- **匈牙利匹配**: 最优一对一标签分配 (`--Hungarian`)
- **去偏损失**: 缓解同一配对多标签竞争 (`--debiasing`)

**v3 原创改进** (可选功能):
- **有向 HOF**: 三分支非对称注意力 + 学习门控 (`--directional-hof`)
- **逐类先验校准**: 每个类别学习独立的检测分数指数 (`--learnable-prior`)
- **动作原型记忆**: EMA 原型辅助稀有类分类 (`--prototype-memory`)
- **空间精炼**: 逐层动态调制空间特征 (`--spatial-refine`)
- **动词层级分类**: 粗粒度语义组辅助损失 (`--hierarchical-verb`)
- **注意力蒸馏**: IoU 先验引导注意力学习 (`--attn-distill`)

```
图像 -> ResNet -> DETR (冻结) -> 目标候选
                -> GCE -> 全局记忆
     候选框     -> HOF -> 成对 Token
     全局记忆   -> PID -> HOI 预测
```

## 基准结果

| 数据集 | 骨干网络 | Full (mAP) | Rare | Non-Rare |
|-------|---------|-----------|------|----------|
| HICO-DET | ResNet-101 | 见论文 | 见论文 | 见论文 |
| V-COCO | ResNet-50 | 见论文 | - | - |

---

## 快速开始

### 环境要求

- Python 3.7+
- PyTorch >= 1.7.1 + CUDA 10.1+
- NVIDIA GPU(s)

### 安装

```bash
git clone <repo-url> GFIN && cd GFIN

conda create -n gfin python=3.7 -y
conda activate gfin
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch

pip install -r requirements.txt
cp .env.example .env  # 编辑填写数据集路径
```

### 数据集准备

**HICO-DET**: 将图像放置或软链接到 `hicodet/hico_20160224_det/`

**V-COCO**: 将 COCO 图像放置或软链接到 `vcoco/mscoco2014/`, V-COCO 标注到 `vcoco/v-coco/`

**DETR 预训练模型**: 从 [OneDrive](https://1drv.ms/f/s!AmZ_JQSzOHiqla8FNyxEJIiSqqz62A?e=txzh8q) 下载并放置在 `checkpoints/detr/`

预训练流程请参考 [UPT](https://github.com/fredzzhang/upt).

### 训练

```bash
# HICO-DET + ResNet-101 (基线)
python main.py \
    --world-size 4 --dataset hicodet --data-root hicodet/ \
    --backbone resnet101 --batch-size 8 \
    --pretrained checkpoints/detr/hico_det/detr-r101-hicodet.pth \
    --output-dir logs/hicodet/gfin_r101/ --lr-head 2e-4

# V-COCO + ResNet-50 (基线)
python main.py \
    --world-size 4 --dataset vcoco --data-root vcoco/ \
    --partitions trainval test --backbone resnet50 --batch-size 8 \
    --pretrained checkpoints/detr/vcoco/detr-r50-vcoco.pth \
    --output-dir logs/vcoco/gfin_r50/ --lr-head 2e-4

# HICO-DET + v2 改进
python main.py \
    --world-size 4 --dataset hicodet --data-root hicodet/ \
    --backbone resnet101 --batch-size 8 \
    --pretrained checkpoints/detr/hico_det/detr-r101-hicodet.pth \
    --output-dir logs/hicodet/gfin_r101_v2/ --lr-head 2e-4 \
    --clip-init --clip-visual --clip-model ViT-B-32 \
    --Hungarian --debiasing --debias-weight 0.5

# HICO-DET + v2 + v3 全部改进
python main.py \
    --world-size 4 --dataset hicodet --data-root hicodet/ \
    --backbone resnet101 --batch-size 8 \
    --pretrained checkpoints/detr/hico_det/detr-r101-hicodet.pth \
    --output-dir logs/hicodet/gfin_r101_v3/ --lr-head 2e-4 \
    --clip-init --clip-visual --Hungarian --debiasing \
    --directional-hof --learnable-prior --prototype-memory \
    --spatial-refine --hierarchical-verb --attn-distill
```

### 改进参数说明

| 参数 | 类型 | 默认值 | 描述 |
|-----|------|--------|------|
| `--clip-init` | flag | False | 使用 CLIP 文本先验初始化分类器权重 |
| `--clip-visual` | flag | False | 融合 CLIP 视觉特征到 GCE |
| `--clip-model` | str | ViT-B-32 | CLIP 模型变体 (ViT-B-32, ViT-B-16, ViT-L-14) |
| `--Hungarian` | flag | False | 使用匈牙利匹配标签分配 |
| `--set-cost-object` | float | 1.0 | 匈牙利匹配物体分类代价权重 |
| `--set-cost-verb` | float | 1.0 | 匈牙利匹配动词分类代价权重 |
| `--debiasing` | flag | False | 启用 Toxic Siblings 去偏损失 |
| `--debias-weight` | float | 0.5 | 去偏损失权重 |
| `--directional-hof` | flag | False | 启用有向非对称 HOF 注意力 |
| `--learnable-prior` | flag | False | 启用逐类可学习先验校准 |
| `--prototype-memory` | flag | False | 启用 EMA 动作原型记忆库 |
| `--proto-weight` | float | 0.1 | 原型辅助损失权重 |
| `--proto-momentum` | float | 0.999 | 原型 EMA 动量系数 |
| `--spatial-refine` | flag | False | 启用逐层空间特征门控 |
| `--hierarchical-verb` | flag | False | 启用动词层级分类辅助损失 |
| `--hier-weight` | float | 0.3 | 层级分类损失权重 |
| `--attn-distill` | flag | False | 启用 IoU 注意力蒸馏 |
| `--attn-distill-weight` | float | 0.1 | 注意力蒸馏损失权重 |

### 评估

```bash
python main.py \
    --world-size 1 --dataset hicodet --data-root hicodet/ \
    --backbone resnet101 --resume logs/hicodet/gfin_r101/ckpt_20.pt --eval
```

### 推理

```bash
python inference.py \
    --dataset hicodet --data-root hicodet/ --backbone resnet101 \
    --resume logs/hicodet/gfin_r101/ckpt_20.pt \
    --image-path /path/to/image.jpg --draw_pic
```

---

## Docker 部署

```bash
# 构建镜像
docker build -t gfin:latest .

# 训练 (HICO-DET)
bash scripts/docker_train_hico.sh

# 训练 (V-COCO)
bash scripts/docker_train_vcoco.sh

# 评估
RESUME=logs/hicodet/gfin_r101/ckpt_20.pt bash scripts/docker_eval_hico.sh

# 交互式终端
docker run --rm -it --gpus all --shm-size=16g \
    -v $(pwd)/hicodet/hico_20160224_det:/workspace/gfin/hicodet/hico_20160224_det:ro \
    -v $(pwd)/logs:/workspace/gfin/logs \
    --entrypoint /bin/bash gfin:latest
```

详见 [Docker 部署指南](docs/2026-02-23-v1-docker-guide.md).

---

## 项目结构

```
GFIN/
+-- main.py                  # 训练/评估入口
+-- inference.py             # 推理/可视化入口
+-- gfin.py                  # GFIN 模型 (检测器 + 交互头)
+-- interaction_head.py      # InteractionHead: GCE + HOF + PID
+-- ops.py                   # 损失函数, 空间编码
+-- utils.py                 # 数据加载, 训练引擎, 评估
+-- gfin_module/             # 核心模块 (GCE, HOF, PID, MBF, CLIP)
+-- detr/                    # DETR 目标检测器
+-- pocket/                  # 轻量级深度学习库
+-- add_on/                  # 推理与可视化辅助工具
+-- hicodet/                 # HICO-DET 数据集工具
+-- vcoco/                   # V-COCO 数据集工具
+-- experiments/             # 实验训练脚本
+-- scripts/                 # Docker 辅助脚本
+-- docs/                    # 文档
```

---

## 文档

- [系统架构文档](docs/2026-02-23-v1-architecture.md) -- 含 Mermaid 架构图和 v2 改进说明
- [安装与使用指南](docs/2026-02-23-v1-setup-guide.md) -- 详细安装和训练说明
- [Docker 部署指南](docs/2026-02-23-v1-docker-guide.md) -- 容器化部署
- [改进方向](docs/2026-02-23-v1-improvement-ideas.md) -- 未来改进路线图

---

## 实验脚本

预配置的训练脚本在 `experiments/`:

```bash
cd experiments/hico/train && bash run.sh    # HICO-DET 训练
cd experiments/vcoco/train && bash run.sh   # V-COCO 训练
```

---

## TensorBoard 监控

```bash
tensorboard --logdir=logs/ --port=6006
# 或使用 Docker Compose:
docker compose up tensorboard -d
```

---

## 已训练模型

训练好的模型权重: [OneDrive](https://1drv.ms/f/s!AmZ_JQSzOHiqla8FNyxEJIiSqqz62A?e=txzh8q)

修正的标注标签 (无 HOI 图像): [OneDrive](https://1drv.ms/f/s!AmZ_JQSzOHiqlZ8H5FaDuUSZ5Th7bg?e=PMMtQI)

---

## 联系方式

如有问题请联系: ddwhzh@163.com
