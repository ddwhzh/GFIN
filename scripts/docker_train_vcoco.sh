#!/bin/bash
set -e

NUM_GPU=${NUM_GPU:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
BACKBONE=${BACKBONE:-resnet50}
EXP_ID=${EXP_ID:-gfin_r50}
PRETRAINED=${PRETRAINED:-checkpoints/detr/vcoco/detr-r50-vcoco.pth}
LR_HEAD=${LR_HEAD:-2e-4}
MASTER_PORT=${MASTER_PORT:-1199}

docker run --rm -it \
    --gpus all \
    --shm-size=16g \
    -v "$(pwd)/vcoco/mscoco2014:/workspace/gfin/vcoco/mscoco2014:ro" \
    -v "$(pwd)/vcoco/v-coco:/workspace/gfin/vcoco/v-coco:ro" \
    -v "$(pwd)/checkpoints:/workspace/gfin/checkpoints:ro" \
    -v "$(pwd)/logs:/workspace/gfin/logs" \
    -e VCOCO_VSRL_ANNOT=vcoco/v-coco/data/vcoco/vcoco_test.json \
    -e VCOCO_COCO_ANNOT=vcoco/v-coco/data/instances_vcoco_all_2014.json \
    -e VCOCO_SPLIT_FILE=vcoco/v-coco/data/splits/vcoco_test.ids \
    gfin:latest \
    main.py \
    --world-size "$NUM_GPU" \
    --master_port "$MASTER_PORT" \
    --dataset vcoco \
    --data-root vcoco/ \
    --partitions trainval test \
    --backbone "$BACKBONE" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "logs/vcoco/$EXP_ID/" \
    --pretrained "$PRETRAINED" \
    --lr-head "$LR_HEAD"
