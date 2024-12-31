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
  --is_pretrained \
  --crop_mode "none" \
  --max_epoch 100 \
  --milestones 60 80 \
  --batch_size 128 \
  --learning_rate 0.01 \
  --gamma 0.1 \
  --val_episode 100 \
  --test_episode 1000 \
  --shot 1 \
  --query 15
