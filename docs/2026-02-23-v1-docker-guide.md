# GFIN Docker 部署指南

> 版本: v1 | 日期: 2026-02-23

---

## 1. 环境要求

- Docker >= 20.10
- NVIDIA Container Toolkit (`nvidia-docker2`)
- NVIDIA GPU, 驱动版本 >= 450.80.02
- Docker Compose >= 1.29 (可选, 用于 `docker-compose.yml`)

### 验证 GPU 支持

```bash
docker run --rm --gpus all nvidia/cuda:10.2-base-ubuntu18.04 nvidia-smi
```

---

## 2. 构建镜像

### 方式 A: 直接构建

```bash
cd GFIN
docker build -t gfin:latest .
```

### 方式 B: 使用辅助脚本

```bash
bash scripts/docker_build.sh
```

---

## 3. 数据目录准备

运行前, 请准备好宿主机上的数据目录:

```
GFIN/
+-- hicodet/
|   +-- hico_20160224_det/    # HICO-DET 图像 (挂载为数据卷)
|   +-- instances_*.json      # 标注文件 (已内置到镜像)
+-- vcoco/
|   +-- mscoco2014/           # COCO 图像 (挂载为数据卷)
|   +-- v-coco/               # V-COCO 标注 (挂载为数据卷)
|   +-- instances_*.json      # 标注文件 (已内置到镜像)
+-- checkpoints/              # 预训练模型 (挂载为数据卷)
+-- logs/                     # 输出目录 (挂载为数据卷)
```

---

## 4. 训练

### 4.1 HICO-DET 训练

```bash
bash scripts/docker_train_hico.sh
```

或手动运行:

```bash
docker run --rm -it \
    --gpus all \
    --shm-size=16g \
    -v $(pwd)/hicodet/hico_20160224_det:/workspace/gfin/hicodet/hico_20160224_det:ro \
    -v $(pwd)/checkpoints:/workspace/gfin/checkpoints:ro \
    -v $(pwd)/logs:/workspace/gfin/logs \
    gfin:latest \
    main.py \
    --world-size 4 \
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
bash scripts/docker_train_vcoco.sh
```

或手动运行:

```bash
docker run --rm -it \
    --gpus all \
    --shm-size=16g \
    -v $(pwd)/vcoco/mscoco2014:/workspace/gfin/vcoco/mscoco2014:ro \
    -v $(pwd)/vcoco/v-coco:/workspace/gfin/vcoco/v-coco:ro \
    -v $(pwd)/checkpoints:/workspace/gfin/checkpoints:ro \
    -v $(pwd)/logs:/workspace/gfin/logs \
    -e VCOCO_VSRL_ANNOT=vcoco/v-coco/data/vcoco/vcoco_test.json \
    -e VCOCO_COCO_ANNOT=vcoco/v-coco/data/instances_vcoco_all_2014.json \
    -e VCOCO_SPLIT_FILE=vcoco/v-coco/data/splits/vcoco_test.ids \
    gfin:latest \
    main.py \
    --world-size 4 \
    --dataset vcoco \
    --data-root vcoco/ \
    --partitions trainval test \
    --backbone resnet50 \
    --batch-size 8 \
    --output-dir logs/vcoco/gfin_r50/ \
    --pretrained checkpoints/detr/vcoco/detr-r50-vcoco.pth \
    --lr-head 2e-4
```

### 4.3 自定义训练参数

通过环境变量覆盖辅助脚本中的默认值:

```bash
NUM_GPU=2 BATCH_SIZE=4 BACKBONE=resnet50 bash scripts/docker_train_hico.sh
```

---

## 5. 评估

### 5.1 HICO-DET

```bash
RESUME=logs/hicodet/gfin_r101/ckpt_20.pt bash scripts/docker_eval_hico.sh
```

### 5.2 V-COCO

```bash
docker run --rm -it \
    --gpus all \
    --shm-size=16g \
    -v $(pwd)/vcoco/mscoco2014:/workspace/gfin/vcoco/mscoco2014:ro \
    -v $(pwd)/vcoco/v-coco:/workspace/gfin/vcoco/v-coco:ro \
    -v $(pwd)/checkpoints:/workspace/gfin/checkpoints:ro \
    -v $(pwd)/logs:/workspace/gfin/logs \
    -e VCOCO_VSRL_ANNOT=vcoco/v-coco/data/vcoco/vcoco_test.json \
    -e VCOCO_COCO_ANNOT=vcoco/v-coco/data/instances_vcoco_all_2014.json \
    -e VCOCO_SPLIT_FILE=vcoco/v-coco/data/splits/vcoco_test.ids \
    gfin:latest \
    main.py \
    --world-size 1 \
    --dataset vcoco \
    --data-root vcoco/ \
    --partitions trainval test \
    --backbone resnet50 \
    --resume logs/vcoco/gfin_r50/ckpt_20.pt \
    --eval
```

---

## 6. Docker Compose

### 6.1 启动训练 + TensorBoard

```bash
# 编辑 docker-compose.yml 中的 command 为训练参数, 然后:
docker compose up -d
```

### 6.2 仅启动 TensorBoard

```bash
docker compose up tensorboard -d
# 访问 http://localhost:6006
```

### 6.3 停止所有服务

```bash
docker compose down
```

---

## 7. 交互式终端

```bash
docker run --rm -it \
    --gpus all \
    --shm-size=16g \
    -v $(pwd)/hicodet/hico_20160224_det:/workspace/gfin/hicodet/hico_20160224_det:ro \
    -v $(pwd)/checkpoints:/workspace/gfin/checkpoints:ro \
    -v $(pwd)/logs:/workspace/gfin/logs \
    --entrypoint /bin/bash \
    gfin:latest
```

---

## 8. 数据卷挂载参考

| 宿主机路径 | 容器路径 | 模式 | 用途 |
|-----------|---------|------|------|
| `hicodet/hico_20160224_det/` | `/workspace/gfin/hicodet/hico_20160224_det/` | ro | HICO-DET 图像 |
| `vcoco/mscoco2014/` | `/workspace/gfin/vcoco/mscoco2014/` | ro | COCO 图像 |
| `vcoco/v-coco/` | `/workspace/gfin/vcoco/v-coco/` | ro | V-COCO 标注 |
| `checkpoints/` | `/workspace/gfin/checkpoints/` | ro | 预训练模型 |
| `logs/` | `/workspace/gfin/logs/` | rw | 训练输出 |
| `outputs_img/` | `/workspace/gfin/outputs_img/` | rw | 推理输出 |

---

## 9. 常见问题排查

| 问题 | 解决方案 |
|-----|---------|
| `RuntimeError: DataLoader worker ... killed` | 增大 `--shm-size` (如 `32g`) |
| GPU 不可见 | 确认已安装 `nvidia-docker2`; 运行 `docker run --gpus all nvidia/cuda:10.2-base nvidia-smi` 验证 |
| logs/ 权限被拒绝 | 确保宿主机 `logs/` 目录已创建且可写 |
| NCCL 超时 | 减少 `--world-size` 或设置 `NCCL_SOCKET_IFNAME` 环境变量 |
| 显存不足 | 减小 `--batch-size` 或 `--max-instances` |
