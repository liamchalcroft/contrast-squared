#! /bin/bash

python ../train_seg_healthy.py \
    --name spm-cnn-simclr-mprage-pc10 \
    --net cnn \
    --amp \
    --logdir ../logs \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 10 \
    --debug \
    --resume

python ../train_seg_healthy.py \
    --name spm-cnn-simclr-bloch-pc10 \
    --net cnn \
    --amp \
    --logdir ../logs \
    --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
    --pc_data 10 \
    --debug \
    --resume

python ../train_seg_healthy.py \
    --name spm-cnn-simclr-bloch-paired-pc10 \
    --net cnn \
    --amp \
    --logdir ../logs \
    --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
    --pc_data 10 \
    --debug \
    --resume