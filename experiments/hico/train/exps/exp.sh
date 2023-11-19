cd ../../../
exp_id="gfin_r101_new"

addr="127.0.0.1"
port=1123
num_gpu=4
batch_size=8

backbone='resnet101'
dataset='hicodet'
dataset_path='hicodet/'

pretrained='checkpoints/detr/hico_det/detr-r101-hicodet.pth'
# resume='logs/hicodet/ha4/ckpt_20.pt'
output_dir=logs/$dataset/$exp_id/

export OMP_NUM_THREADS=1
python main.py \
--world-size $num_gpu \
--master_addr $addr \
--master_port $port \
--dataset $dataset \
--data-root $dataset_path \
--backbone $backbone \
--batch-size $batch_size \
--output-dir $output_dir \
--pretrained $pretrained \
--lr-head 2e-4 