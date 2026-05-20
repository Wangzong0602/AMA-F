#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00

for trial in 1 2 3 4 5 6 7 8 9 10
do
	python train_ext.py \
	--dataset regdb  \
	--lr 0.1  \
	--method adp \
	--augc 1 \
	--rande 0.5 \
	--gpu 0 \
	--trial $trial
done
echo 'Done!'