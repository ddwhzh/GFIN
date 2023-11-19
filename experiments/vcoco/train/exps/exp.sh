cd ../../../
exp_id="gfin_r50_new"

addr="127.0.0.1"
port=1199
num_gpu=4
batch_size=8

backbone='resnet50'
dataset='vcoco'
dataset_path='vcoco/'


pretrained='checkpoints/detr/vcoco/detr-r50-vcoco.pth'
resume='logs/vcoco/baseline/ckpt_20.pt'
output_dir=logs/$dataset/$exp_id/

export OMP_NUM_THREADS=1
python main.py \
--world-size $num_gpu \
--master_addr $addr \
--master_port $port \
--dataset $dataset \
--data-root $dataset_path \
--partitions trainval test \
--backbone $backbone \
--batch-size $batch_size \
--output-dir $output_dir \
--pretrained $pretrained \
--lr-head 2e-4 
# --resume $resume \
# --eval 