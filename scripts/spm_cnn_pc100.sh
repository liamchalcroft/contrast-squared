#! /bin/bash


# MPRAGE
python ../train_seg_healthy.py \
    --name spm-t1-cnn-simclr-mprage-pc100 \
    --net cnn \
    --modality t1 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 100 \
    --debug \
    --resume

python ../train_seg_healthy.py \
    --name spm-t2-cnn-simclr-mprage-pc100 \
    --net cnn \
    --modality t2 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 100 \
    --debug \
    --resume

python ../train_seg_healthy.py \
    --name spm-flair-cnn-simclr-mprage-pc100 \
    --net cnn \
    --modality flair \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 100 \
    --debug \
    --resume


# # BLOCH
# python ../train_seg_healthy.py \
#     --name spm-t1-cnn-simclr-bloch-pc100 \
#     --net cnn \
#     --modality t1 \
#     --amp \
#     --logdir ../ \
#     --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
#     --pc_data 100 \
#     --debug \
#     --resume

# python ../train_seg_healthy.py \
#     --name spm-t2-cnn-simclr-bloch-pc100 \
#     --net cnn \
#     --modality t2 \
#     --amp \
#     --logdir ../ \
#     --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
#     --pc_data 100 \
#     --debug \
#     --resume

# python ../train_seg_healthy.py \
#     --name spm-flair-cnn-simclr-bloch-pc100 \
#     --net cnn \
#     --modality flair \
#     --amp \
#     --logdir ../ \
#     --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
#     --pc_data 100 \
#     --debug \
#     --resume


# # BLOCH PAIRED
# python ../train_seg_healthy.py \
#     --name spm-t1-cnn-simclr-bloch-paired-pc100 \
#     --net cnn \
#     --modality t1 \
#     --amp \
#     --logdir ../ \
#     --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
#     --pc_data 100 \
#     --debug \
#     --resume

# python ../train_seg_healthy.py \
#     --name spm-t2-cnn-simclr-bloch-paired-pc100 \
#     --net cnn \
#     --modality t2 \
#     --amp \
#     --logdir ../ \
#     --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
#     --pc_data 100 \
#     --debug \
#     --resume

# python ../train_seg_healthy.py \
#     --name spm-flair-cnn-simclr-bloch-paired-pc100 \
#     --net cnn \
#     --modality flair \
#     --amp \
#     --logdir ../ \
#     --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
#     --pc_data 100 \
#     --debug \
#     --resume
