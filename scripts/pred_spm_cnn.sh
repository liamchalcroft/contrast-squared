#! /bin/bash

# 100% of training data
python ../src/pred_seg_healthy.py \
    --weights ../spm-cnn-simclr-mprage-pc100/checkpoint.pt \
    --net cnn \
    --amp \

python ../src/pred_seg_healthy.py \
    --weights ../spm-cnn-simclr-bloch-pc100/checkpoint.pt \
    --net cnn \
    --amp \

python ../src/pred_seg_healthy.py \
    --weights ../spm-cnn-simclr-bloch-paired-pc100/checkpoint.pt \
    --net cnn \
    --amp \

# 10% of training data
python ../src/pred_seg_healthy.py \
    --weights ../spm-cnn-simclr-mprage-pc10/checkpoint.pt \
    --net cnn \
    --amp \

python ../src/pred_seg_healthy.py \
    --weights ../spm-cnn-simclr-bloch-pc10/checkpoint.pt \
    --net cnn \
    --amp \

python ../src/pred_seg_healthy.py \
    --weights ../spm-cnn-simclr-bloch-paired-pc10/checkpoint.pt \
    --net cnn \
    --amp \

# 1% of training data
python ../src/pred_seg_healthy.py \
    --weights ../spm-cnn-simclr-mprage-pc1/checkpoint.pt \
    --net cnn \
    --amp \

python ../src/pred_seg_healthy.py \
    --weights ../spm-cnn-simclr-bloch-pc1/checkpoint.pt \
    --net cnn \
    --amp \

python ../src/pred_seg_healthy.py \
    --weights ../spm-cnn-simclr-bloch-paired-pc1/checkpoint.pt \
    --net cnn \
    --amp \
