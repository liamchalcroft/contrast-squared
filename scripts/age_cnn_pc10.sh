#! /bin/bash

python ../train_regressor.py \
    --name age-cnn-simclr-mprage-pc10 \
    --net cnn \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 10 \
    --debug \
    --resume

python ../train_regressor.py \
    --name age-cnn-simclr-bloch-pc10 \
    --net cnn \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
    --pc_data 10 \
    --debug \
    --resume

python ../train_regressor.py \
    --name age-cnn-simclr-bloch-paired-pc10 \
    --net cnn \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
    --pc_data 10 \
    --debug \
    --resume