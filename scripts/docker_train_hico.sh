#!/bin/bash
set -e

NUM_GPU=${NUM_GPU:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
BACKBONE=${BACKBONE:-resnet101}
EXP_ID=${EXP_ID:-gfin_r101}
PRETRAINED=${PRETRAINED:-checkpoints/detr/hico_det/detr-r101-hicodet.pth}
LR_HEAD=${LR_HEAD:-2e-4}
MASTER_PORT=${MASTER_PORT:-1123}

docker run --rm -it \
    --gpus all \
    --shm-size=16g \
    -v "$(pwd)/hicodet/hico_20160224_det:/workspace/gfin/hicodet/hico_20160224_det:ro" \
    -v "$(pwd)/checkpoints:/workspace/gfin/checkpoints:ro" \
    -v "$(pwd)/logs:/workspace/gfin/logs" \
    gfin:latest \
    main.py \
    --world-size "$NUM_GPU" \
    --master_port "$MASTER_PORT" \
    --dataset hicodet \
    --data-root hicodet/ \
    --backbone "$BACKBONE" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "logs/hicodet/$EXP_ID/" \
    --pretrained "$PRETRAINED" \
    --lr-head "$LR_HEAD"
