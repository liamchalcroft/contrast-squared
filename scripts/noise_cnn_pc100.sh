#! /bin/bash

# MPRAGE
python ../src/train_denoise.py \
    --name noise-t1-cnn-simclr-mprage-pc100 \
    --net cnn \
    --modality t1 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 100 \
    --debug \
    --resume

python ../src/train_denoise.py \
    --name noise-t2-cnn-simclr-mprage-pc100 \
    --net cnn \
    --modality t2 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 100 \
    --debug \
    --resume

python ../src/train_denoise.py \
    --name noise-pd-cnn-simclr-mprage-pc100 \
    --net cnn \
    --modality pd \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 100 \
    --debug \
    --resume

# BLOCH
python ../src/train_denoise.py \
    --name noise-t1-cnn-simclr-bloch-pc100 \
    --net cnn \
    --modality t1 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
    --pc_data 100 \
    --debug \
    --resume

python ../src/train_denoise.py \
    --name noise-t2-cnn-simclr-bloch-pc100 \
    --net cnn \
    --modality t2 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
    --pc_data 100 \
    --debug \
    --resume

python ../src/train_denoise.py \
    --name noise-pd-cnn-simclr-bloch-pc100 \
    --net cnn \
    --modality pd \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
    --pc_data 100 \
    --debug \
    --resume

# BLOCH PAIRED
python ../src/train_denoise.py \
    --name noise-t1-cnn-simclr-bloch-paired-pc100 \
    --net cnn \
    --modality t1 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
    --pc_data 100 \
    --debug \
    --resume

python ../src/train_denoise.py \
    --name noise-t2-cnn-simclr-bloch-paired-pc100 \
    --net cnn \
    --modality t2 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
    --pc_data 100 \
    --debug \
    --resume

python ../src/train_denoise.py \
    --name noise-pd-cnn-simclr-bloch-paired-pc100 \
    --net cnn \
    --modality pd \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
    --pc_data 100 \
    --debug \
    --resume
