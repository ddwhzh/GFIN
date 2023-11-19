cd ../../../hicodet/detections
exp_id="pretrain"

addr="127.0.0.1"
port=2500
num_gpu=4

backbone='resnet50'
dataset='vcoco'

#pretrained=../../checkpoints/detr/$dataset/detr-r50-e632da11.pth
coco_path=../../vcoco/v-coco
output_dir=../../logs/$dataset/$exp_id/


export OMP_NUM_THREADS=1
python main_detr.py \
--world-size $num_gpu \
--master_addr $addr \
--master_port $port \
--coco_path $coco_path \
--output_dir $output_dir \
--pretrained $pretrained 
