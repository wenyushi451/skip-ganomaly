#!/bin/bash
/bigdisk/lax/weshiz/anaconda3/envs/ganomaly/bin/python3.7 train.py                     \
    --dataset yunfangbu    \
    --batchsize 4   \
	--print_freq 4    \
	--isize 256            \
    --niter 15      \
	--ngf 64        \
	--ndf 64        \
	--lr 0.0002      \
	--model ganomaly        \
	--save_test_image     \
    --verbose
