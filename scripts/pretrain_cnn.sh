#! /bin/bash

python ../src/pretrain_3d.py \
    --name 3d-cnn-simclr-mprage \
    --batch_size 8 \
    --epochs 300 \
    --epoch_length 200 \
    --lr 1e-3 \
    --loss simclr \
    --data mprage \
    --net cnn \
    --amp \
    --logdir ../ \
    --debug \
    --resume

python ../src/pretrain_3d.py \
    --name 3d-cnn-simclr-bloch \
    --batch_size 8 \
    --epochs 300 \
    --epoch_length 200 \
    --lr 1e-3 \
    --loss simclr \
    --data bloch \
    --net cnn \
    --amp \
    --logdir ../ \
    --debug \
    --resume

python ../src/pretrain_3d.py \
    --name 3d-cnn-simclr-bloch-paired \
    --batch_size 8 \
    --epochs 300 \
    --epoch_length 200 \
    --lr 1e-3 \
    --loss simclr \
    --data bloch-paired \
    --net cnn \
    --amp \
    --logdir ../ \
    --debug \
    --resume