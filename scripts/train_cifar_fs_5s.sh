#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu_4090
#SBATCH --output=/HOME/scw6c09/run/BLAN_0508/out/cifar_fs/%j.out

module load anaconda/2022.10
source activate py10_0308

date
mkdir /dev/shm/datasets
tar -xf /HOME/scw6c09/run/datasets/cifar_fs.tar -C /dev/shm/datasets
date

python run.py \
  --data_root "/dev/shm/datasets" \
  --dataset "cifar_fs" \
  --fine_tuning \
  --pretrain_dir "checkpoint/cifar_fs/pretrain_max_accuracy_100_128_0.01_0.1_0509.pth" \
  --crop_mode "grid" \
  --max_epoch 40 \
  --milestones 20 30 \
  --patch_ratio 2.0 \
  --learning_rate 0.0005 \
  --gamma 0.2 \
  --metric "cos" \
  --tau 0.1 \
  --shot 5 \
  --query 5 \
  --val_episode 100 \
  --test_episode 1000 \
  --random 4
