#! /bin/bash

python ../train_classifier.py \
    --name age-cnn-simclr-mprage-pc1 \
    --net cnn \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 1 \
    --debug \
    --resume

python ../train_classifier.py \
    --name age-cnn-simclr-bloch-pc1 \
    --net cnn \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
    --pc_data 1 \
    --debug \
    --resume

python ../train_classifier.py \
    --name age-cnn-simclr-bloch-paired-pc1 \
    --net cnn \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
    --pc_data 1 \
    --debug \
    --resume