#!/bin/bash

# Run MNIST experiment for each individual dataset.
# For each anomalous digit
/bigdisk/lax/weshiz/anaconda3/envs/ganomaly/bin/python3.7 train.py --dataset mnist --isize 32 --nc 1 --niter 15 --abnormal_class 1 --model ganomaly --save_test_images
exit 0
