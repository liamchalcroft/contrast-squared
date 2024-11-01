#! /bin/bash


# MPRAGE
python ../train_seg_healthy.py \
    --name spm-t1-cnn-simclr-mprage-pc10 \
    --net cnn \
    --modality t1 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 10 \
    --debug \
    --resume

python ../train_seg_healthy.py \
    --name spm-t2-cnn-simclr-mprage-pc10 \
    --net cnn \
    --modality t2 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 10 \
    --debug \
    --resume

python ../train_seg_healthy.py \
    --name spm-pd-cnn-simclr-mprage-pc10 \
    --net cnn \
    --modality pd \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 10 \
    --debug \
    --resume


# BLOCH
python ../train_seg_healthy.py \
    --name spm-t1-cnn-simclr-bloch-pc10 \
    --net cnn \
    --modality t1 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
    --pc_data 10 \
    --debug \
    --resume

python ../train_seg_healthy.py \
    --name spm-t2-cnn-simclr-bloch-pc10 \
    --net cnn \
    --modality t2 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
    --pc_data 10 \
    --debug \
    --resume

python ../train_seg_healthy.py \
    --name spm-pd-cnn-simclr-bloch-pc10 \
    --net cnn \
    --modality pd \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
    --pc_data 10 \
    --debug \
    --resume


# BLOCH PAIRED
python ../train_seg_healthy.py \
    --name spm-t1-cnn-simclr-bloch-paired-pc10 \
    --net cnn \
    --modality t1 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
    --pc_data 10 \
    --debug \
    --resume

python ../train_seg_healthy.py \
    --name spm-t2-cnn-simclr-bloch-paired-pc10 \
    --net cnn \
    --modality t2 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
    --pc_data 10 \
    --debug \
    --resume

python ../train_seg_healthy.py \
    --name spm-pd-cnn-simclr-bloch-paired-pc10 \
    --net cnn \
    --modality pd \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
    --pc_data 10 \
    --debug \
    --resume
