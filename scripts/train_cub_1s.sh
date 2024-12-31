#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu_4090
#SBATCH --output=/HOME/scw6c09/run/BLAN_0508/out/cub/%j.out

module load anaconda/2022.10
source activate py10_0308

date
mkdir /dev/shm/datasets
tar -xf /HOME/scw6c09/run/datasets/cub.tar -C /dev/shm/datasets
date

python run.py \
  --data_root "/dev/shm/datasets" \
  --dataset "cub" \
  --fine_tuning \
  --pretrain_dir "checkpoint/cub/pretrain_max_accuracy_100_64_0.01_0.1_0509.pth" \
  --crop_mode "grid" \
  --max_epoch 60 \
  --milestones 20 40 \
  --patch_ratio 2.0 \
  --learning_rate 0.0005 \
  --gamma 0.2 \
  --metric "cos" \
  --tau 0.1 \
  --shot 1 \
  --query 5 \
  --random 4
