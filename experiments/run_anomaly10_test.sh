#!/bin/bash
/bigdisk/lax/weshiz/anaconda3/envs/ganomaly/bin/python3.7 train.py                     \
    --dataset anomaly10    \
    --batchsize 16   \
	--isize 256           \
    --niter 20      \
	--nc 3          \
	--ngf 32        \
	--ndf 32        \
	--nz 100         \
	--lr_policy step    \
	--lr 0.0002      \
	--model ganomaly        \
	--save_test_image     \
    --verbose      
