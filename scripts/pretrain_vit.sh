#! /bin/bash

python ../pretrain_3d.py \
    --name 3d-vit-simclr-mprage \
    --batch_size 8 \
    --epochs 200 \
    --epoch_length 200 \
    --lr 1e-3 \
    --loss simclr \
    --data mprage \
    --net vit \
    --amp \
    --logdir ../ \
    --debug \
    --resume

python ../pretrain_3d.py \
    --name 3d-vit-simclr-mprage \
    --batch_size 8 \
    --epochs 200 \
    --epoch_length 200 \
    --lr 1e-3 \
    --loss simclr \
    --data mprage \
    --net vit \
    --amp \
    --logdir ../ \
    --debug \
    --resume

python ../pretrain_3d.py \
    --name 3d-vit-simclr-mprage-paired \
    --batch_size 8 \
    --epochs 200 \
    --epoch_length 200 \
    --lr 1e-3 \
    --loss simclr \
    --data mprage-paired \
    --net vit \
    --amp \
    --logdir ../ \
    --debug \
    --resume