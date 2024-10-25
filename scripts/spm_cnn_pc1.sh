#! /bin/bash

python ../train_seg_healthy.py \
    --name spm-cnn-simclr-mprage-pc1 \
    --net cnn \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 1 \
    --debug \
    --resume

python ../train_seg_healthy.py \
    --name spm-cnn-simclr-bloch-pc1 \
    --net cnn \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
    --pc_data 1 \
    --debug \
    --resume

python ../train_seg_healthy.py \
    --name spm-cnn-simclr-bloch-paired-pc1 \
    --net cnn \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
    --pc_data 1 \
    --debug \
    --resume