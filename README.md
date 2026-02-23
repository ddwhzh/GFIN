# GFIN

**基于全局上下文与成对融合特征的人-物交互检测**

GFIN 检测并分类图像中的人-物交互 (如 "人骑自行车"). 它将冻结的 DETR 目标检测器与自定义交互头结合, 交互头包含三个核心模块:

- **GCE** (全局上下文编码器): 通过 Transformer 编码器编码图像级上下文
- **HOF** (人-物融合): 通过多头成对注意力融合外观与空间特征
- **PID** (成对交互解码器): 交叉注意力解码器, 用于 HOI 分类

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
# HICO-DET + ResNet-101
python main.py \
    --world-size 4 --dataset hicodet --data-root hicodet/ \
    --backbone resnet101 --batch-size 8 \
    --pretrained checkpoints/detr/hico_det/detr-r101-hicodet.pth \
    --output-dir logs/hicodet/gfin_r101/ --lr-head 2e-4

# V-COCO + ResNet-50
python main.py \
    --world-size 4 --dataset vcoco --data-root vcoco/ \
    --partitions trainval test --backbone resnet50 --batch-size 8 \
    --pretrained checkpoints/detr/vcoco/detr-r50-vcoco.pth \
    --output-dir logs/vcoco/gfin_r50/ --lr-head 2e-4
```

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
+-- gfin_module/             # 核心模块 (GCE, HOF, PID, MBF)
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

- [系统架构文档](docs/2026-02-23-v1-architecture.md) -- 含 Mermaid 架构图
- [安装与使用指南](docs/2026-02-23-v1-setup-guide.md) -- 详细安装和训练说明
- [Docker 部署指南](docs/2026-02-23-v1-docker-guide.md) -- 容器化部署

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
