# GFIN 安装与使用指南

> 版本: v1 | 日期: 2026-02-23

---

## 1. 环境要求

- Python 3.7+
- PyTorch >= 1.7.1, CUDA 10.1+ 支持
- NVIDIA GPU(s), >= 8GB 显存 (推荐: 4x V100/A100)
- Docker (可选, 用于容器化部署)

---

## 2. 本地安装

### 2.1 克隆仓库

```bash
git clone <repo-url> GFIN
cd GFIN
```

### 2.2 创建虚拟环境

```bash
conda create -n gfin python=3.7 -y
conda activate gfin

conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```

### 2.3 环境配置

```bash
cp .env.example .env
# 编辑 .env 填写你的本地数据集路径
```

---

## 3. 数据集准备

### 3.1 HICO-DET

```bash
cd hicodet
# 方式 A: 下载
bash download.sh

# 方式 B: 软链接已有数据集
ln -s /path/to/hico_20160224_det ./hico_20160224_det
```

期望的目录结构:
```
hicodet/
+-- hico_20160224_det/
|   +-- images/
|       +-- train2015/
|       +-- test2015/
+-- instances_train2015.json
+-- instances_test2015.json
+-- file_name_to_obj_cat.json
+-- hoi_list_new.json
```

### 3.2 V-COCO

```bash
cd vcoco
# 方式 A: 下载
bash download.sh

# 方式 B: 软链接已有数据集
ln -s /path/to/mscoco2014 ./mscoco2014
ln -s /path/to/v-coco ./v-coco
```

期望的目录结构:
```
vcoco/
+-- mscoco2014/
|   +-- train2014/
|   +-- val2014/
+-- v-coco/
|   +-- data/
|       +-- vcoco/
|       +-- splits/
|       +-- instances_vcoco_all_2014.json
+-- instances_vcoco_trainval.json
+-- instances_vcoco_test.json
```

### 3.3 DETR 预训练模型

下载 DETR 预训练权重并放置在 `checkpoints/`:

```
checkpoints/
+-- detr/
    +-- hico_det/
    |   +-- detr-r50-hicodet.pth
    |   +-- detr-r101-hicodet.pth
    +-- vcoco/
        +-- detr-r50-vcoco.pth
```

预训练模型下载: [OneDrive 链接](https://1drv.ms/f/s!AmZ_JQSzOHiqla8FNyxEJIiSqqz62A?e=txzh8q)

预训练流程请参考 [UPT](https://github.com/fredzzhang/upt).

---

## 4. 训练

### 4.1 HICO-DET 训练

```bash
export OMP_NUM_THREADS=1
python main.py \
    --world-size 4 \
    --master_port 1123 \
    --dataset hicodet \
    --data-root hicodet/ \
    --backbone resnet101 \
    --batch-size 8 \
    --output-dir logs/hicodet/gfin_r101/ \
    --pretrained checkpoints/detr/hico_det/detr-r101-hicodet.pth \
    --lr-head 2e-4
```

### 4.2 V-COCO 训练

```bash
export OMP_NUM_THREADS=1
python main.py \
    --world-size 4 \
    --master_port 1199 \
    --dataset vcoco \
    --data-root vcoco/ \
    --partitions trainval test \
    --backbone resnet50 \
    --batch-size 8 \
    --output-dir logs/vcoco/gfin_r50/ \
    --pretrained checkpoints/detr/vcoco/detr-r50-vcoco.pth \
    --lr-head 2e-4
```

### 4.3 恢复训练

添加 `--resume <checkpoint路径>`:

```bash
python main.py \
    ... \
    --resume logs/hicodet/gfin_r101/ckpt_10.pt
```

---

## 5. 评估

### 5.1 HICO-DET 评估

```bash
python main.py \
    --world-size 1 \
    --dataset hicodet \
    --data-root hicodet/ \
    --backbone resnet101 \
    --resume logs/hicodet/gfin_r101/ckpt_20.pt \
    --eval
```

Known Object (KO) 评估模式:
```bash
python main.py \
    ... \
    --eval --ko_mode
```

### 5.2 V-COCO 评估

```bash
python main.py \
    --world-size 1 \
    --dataset vcoco \
    --data-root vcoco/ \
    --partitions trainval test \
    --backbone resnet50 \
    --resume logs/vcoco/gfin_r50/ckpt_20.pt \
    --eval \
    --vcoco-vsrl-annot vcoco/v-coco/data/vcoco/vcoco_test.json \
    --vcoco-coco-annot vcoco/v-coco/data/instances_vcoco_all_2014.json \
    --vcoco-split-file vcoco/v-coco/data/splits/vcoco_test.ids
```

---

## 6. 推理

### 6.1 单张图片

```bash
python inference.py \
    --dataset hicodet \
    --data-root hicodet/ \
    --backbone resnet101 \
    --resume logs/hicodet/gfin_r101/ckpt_20.pt \
    --image-path /path/to/image.jpg \
    --action-score-thresh 0.2 \
    --draw_pic
```

### 6.2 视频

```bash
python inference.py \
    --dataset hicodet \
    --data-root hicodet/ \
    --backbone resnet101 \
    --resume logs/hicodet/gfin_r101/ckpt_20.pt \
    --video \
    --video_dir /path/to/video.mp4 \
    --save_video /path/to/output.mp4
```

---

## 7. Docker 部署

详见 [Docker 部署指南](./2026-02-23-v1-docker-guide.md).

---

## 8. TensorBoard 监控

```bash
tensorboard --logdir=logs/ --port=6006
```

打开 http://localhost:6006 监控训练指标 (mAP, 损失, 稀有/非稀有类 AP).

---

## 9. 常见问题

| 问题 | 解决方案 |
|-----|---------|
| CUDA 显存不足 | 减小 `--batch-size` 或 `--max-instances` |
| NaN 损失 | 检查预训练模型路径; 确认数据集分区正确 |
| mAP 偏低 | 确认 DETR 预训练权重与骨干网络配置匹配 |
| 分布式训练错误 | 检查 `--master_port` 是否被占用; 尝试不同端口 |
| V-COCO 评估错误 | 正确设置 `--vcoco-vsrl-annot` / `--vcoco-coco-annot` / `--vcoco-split-file` |
