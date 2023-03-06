#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/pascal_r50_labeled_only.yaml
labeled_id_path=partitions/pascal/366/labeled.txt
unlabeled_id_path=partitions/pascal/366/unlabeled.txt
save_path=exp/pascal/supervised/r50_321x_2x16/366

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    supervised.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt