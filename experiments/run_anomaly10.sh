#!/bin/bash
/bigdisk/lax/weshiz/anaconda3/envs/ganomaly/bin/python3.7 train.py                     \
    --dataset anomaly10    \
    --batchsize 32   \
	--print_freq 16    \
	--isize 256            \
    --niter 20      \
	--nc 1          \
	--ngf 32        \
	--ndf 32        \
	--nz 100         \
	--lr 0.0002      \
	--model ganomaly        \
	--save_test_image     \
    --verbose      
