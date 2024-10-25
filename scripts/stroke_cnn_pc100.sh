#! /bin/bash

python ../train_seg_stroke.py \
    --name stroke-cnn-simclr-mprage-pc100 \
    --net cnn \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 100 \
    --debug \
    --resume

python ../train_seg_stroke.py \
    --name stroke-cnn-simclr-bloch-pc100 \
    --net cnn \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
    --pc_data 100 \
    --debug \
    --resume

python ../train_seg_stroke.py \
    --name stroke-cnn-simclr-bloch-paired-pc100 \
    --net cnn \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
    --pc_data 100 \
    --debug \
    --resume