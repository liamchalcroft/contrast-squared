#! /bin/bash

# 100% of training data
python ../test_seg_stroke.py \
    --weights ../stroke-cnn-simclr-mprage-pc100/checkpoint.pt \
    --net cnn \
    --amp \

python ../test_seg_stroke.py \
    --weights ../stroke-cnn-simclr-bloch-pc100/checkpoint.pt \
    --net cnn \
    --amp \

python ../test_seg_stroke.py \
    --weights ../stroke-cnn-simclr-bloch-paired-pc100/checkpoint.pt \
    --net cnn \
    --amp \

# 10% of training data
python ../test_seg_stroke.py \
    --weights ../stroke-cnn-simclr-mprage-pc10/checkpoint.pt \
    --net cnn \
    --amp \

python ../test_seg_stroke.py \
    --weights ../stroke-cnn-simclr-bloch-pc10/checkpoint.pt \
    --net cnn \
    --amp \

python ../test_seg_stroke.py \
    --weights ../stroke-cnn-simclr-bloch-paired-pc10/checkpoint.pt \
    --net cnn \
    --amp \

# 1% of training data
python ../test_seg_stroke.py \
    --weights ../stroke-cnn-simclr-mprage-pc1/checkpoint.pt \
    --net cnn \
    --amp \

python ../test_seg_stroke.py \
    --weights ../stroke-cnn-simclr-bloch-pc1/checkpoint.pt \
    --net cnn \
    --amp \

python ../test_seg_stroke.py \
    --weights ../stroke-cnn-simclr-bloch-paired-pc1/checkpoint.pt \
    --net cnn \
    --amp \
