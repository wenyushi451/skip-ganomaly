#!/bin/bash
/bigdisk/lax/weshiz/anaconda3/envs/ganomaly/bin/python3.7 train.py                     \
    --dataset csot    \
    --model ganomaly        \
	--batchsize 4        \
	--isize 64            \
    --niter 15      \
	--save_test_image    
