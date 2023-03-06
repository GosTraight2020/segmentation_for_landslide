#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/landslide_128.yaml
labeled_id_path=partitions/landslide/100/labeled.txt
unlabeled_id_path=partitions/landslide/100/unlabeled.txt
save_path=exp/landslide128/fixmatch/100/

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    fixmatch.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt