#!/bin/bash
set -e

NUM_GPU=${NUM_GPU:-1}
BACKBONE=${BACKBONE:-resnet101}
RESUME=${RESUME:?'Please set RESUME to the checkpoint path, e.g. RESUME=logs/hicodet/gfin_r101/ckpt_20.pt'}
MASTER_PORT=${MASTER_PORT:-1124}

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
    --resume "$RESUME" \
    --eval
