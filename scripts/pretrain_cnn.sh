#! /bin/bash

python pretrain_3d.py \
    --name 3d-cnn-simclr-mprage \
    --batch_size 8 \
    --epochs 200 \
    --epoch_length 200 \
    --lr 1e-3 \
    --loss simclr \
    --data mprage \
    --net cnn \
    --amp \
    --logdir ../logs \
    --debug \
    --resume

python pretrain_3d.py \
    --name 3d-cnn-simclr-mprage \
    --batch_size 8 \
    --epochs 200 \
    --epoch_length 200 \
    --lr 1e-3 \
    --loss simclr \
    --data mprage \
    --net cnn \
    --amp \
    --logdir ../logs \
    --debug \
    --resume

python pretrain_3d.py \
    --name 3d-cnn-simclr-bloch-paired \
    --batch_size 8 \
    --epochs 200 \
    --epoch_length 200 \
    --lr 1e-3 \
    --loss simclr \
    --data bloch-paired \
    --net cnn \
    --amp \
    --logdir ../logs \
    --debug \
    --resume