#! /bin/bash

python ../src/plot_tsne.py \
    --weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --net cnn \
    --amp \

python ../src/plot_tsne.py \
    --weights ../3d-cnn-simclr-bloch/checkpoint.pt \
    --net cnn \
    --amp \

python ../src/plot_tsne.py \
    --weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
    --net cnn \
    --amp \