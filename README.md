# GFIN
Human-Object Interaction Detection via Global Context and Pairwise-level Fusion Features Integration

Training inference (vscode debugging scripts are provided and can be used directly)

. /experiments/ contains scripts for foreground & background running, and scripts for waiting for the graphics card to idle.

## pre-training

For pre-training models, please refer to the UPT (https://github.com/fredzzhang/upt)

## dataset

hicodet\hico_20160224_det Here is the HICODET dataset (refer to UPT).

pocket repository I modified a little to support multi-threaded training (https://github.com/fredzzhang/pocket)

vcoco\v-coco put vcoco repository (refer to UPT)

Fix previous issue of labels not counting images without HOI (VCOCO/HICO-DET onedrive)
https://1drv.ms/f/s!AmZ_JQSzOHiqlZ8H5FaDuUSZ5Th7bg?e=PMMtQI

## train && output

logs to put the output model

Location of trained model (VCOCO/HICO-DET onedrive)
https://1drv.ms/f/s!AmZ_JQSzOHiqla8FNyxEJIiSqqz62A?e=txzh8q


Pytorch using 1.7.1 + cuda10.1.


If there are any problems you can contact: ddwhzh@163.com