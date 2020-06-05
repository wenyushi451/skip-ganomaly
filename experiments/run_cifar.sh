#!/bin/bash

# Run CIFAR10 experiment on ganomaly

declare -a arr=("airplane" "automobile" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck" )
/bigdisk/lax/weshiz/anaconda3/envs/ganomaly/bin/python3.7 train.py --dataset cifar10 --isize 32 --niter 25 --abnormal_class "airplane" --model ganomaly
exit 0
