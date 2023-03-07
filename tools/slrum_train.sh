#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job=slurm_train

config=configs/landslide_128.yaml
labeled_id_path=partitions/landslide/100/labeled.txt
unlabeled_id_path=partitions/landslide/100/unlabeled.txt
save_path=exp/landslide128/fixmatch/100/

mkdir -p $save_path
srun  -p $2 -N 1 -n $1 --gres=gpu:$1 --ntasks-per-node=$1 --job-name=$job --mem-per-cpu=16GB --time 01-00:00:00 -A cs601_gpu \
python -u fixmatch.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $3 2>&1 | tee $save_path/$now.txt