#!/bin/bash
/bigdisk/lax/weshiz/anaconda3/envs/ganomaly/bin/python3.7 train.py                     \
    --dataset anomaly1    \
    --batchsize 16   \
	--isize 128            \
    --niter 40      \
	--nc 1          \
	--ngf 64        \
	--ndf 64        \
	--nz 64         \
	--lr_policy step    \
	--lr 0.001      \
	--model ganomaly        \
	--save_test_image     \
    --verbose      
