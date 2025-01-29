#! /bin/bash

python ../src/pretrain_3d.py \
    --name 3d-cnn-simclr-mprage-norecon \
    --batch_size 8 \
    --epochs 300 \
    --epoch_length 200 \
    --lr 1e-3 \
    --loss simclr \
    --data mprage \
    --no_recon \
    --net cnn \
    --amp \
    --logdir ../ \
    --debug \
    --resume

python ../src/pretrain_3d.py \
    --name 3d-cnn-simclr-bloch-norecon \
    --batch_size 8 \
    --epochs 300 \
    --epoch_length 200 \
    --lr 1e-3 \
    --loss simclr \
    --data bloch \
    --no_recon \
    --net cnn \
    --amp \
    --logdir ../ \
    --debug \
    --resume

python ../src/pretrain_3d.py \
    --name 3d-cnn-simclr-bloch-paired-norecon \
    --batch_size 8 \
    --epochs 300 \
    --epoch_length 200 \
    --lr 1e-3 \
    --loss simclr \
    --data bloch-paired \
    --no_recon \
    --net cnn \
    --amp \
    --logdir ../ \
    --debug \
    --resume