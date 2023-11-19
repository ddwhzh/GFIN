cd ../../../
exp_id="baseline"

addr="127.0.0.1"
port=1199
num_gpu=4
batch_size=4

backbone='resnet50'
dataset='vcoco'
dataset_path='vcoco/'

# resume='checkpoints/detr/vcoco/detr-r50-vcoco.pth'
pretrained='checkpoints/detr/detr-r50-e632da11.pth'
output_dir=logs/$dataset/pre_$exp_id/

export OMP_NUM_THREADS=1
python hicodet/detections/main_detr.py \
--backbone $backbone \
--world-size $num_gpu \
--master_addr $addr \
--master_port $port \
--batch_size $batch_size \
--output_dir $output_dir \
--epochs 300 \
--lr_drop 200
# --pretrained $pretrained
# --resume $resume \
# --eval