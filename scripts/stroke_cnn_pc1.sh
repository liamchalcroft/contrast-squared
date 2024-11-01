#! /bin/bash


# MPRAGE
python ../train_seg_stroke.py \
    --name stroke-t1-cnn-simclr-mprage-pc1 \
    --net cnn \
    --modality t1 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 1 \
    --debug \
    --resume

python ../train_seg_stroke.py \
    --name stroke-t2-cnn-simclr-mprage-pc1 \
    --net cnn \
    --modality t2 \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 1 \
    --debug \
    --resume

python ../train_seg_stroke.py \
    --name stroke-flair-cnn-simclr-mprage-pc1 \
    --net cnn \
    --modality flair \
    --amp \
    --logdir ../ \
    --backbone_weights ../3d-cnn-simclr-mprage/checkpoint.pt \
    --pc_data 1 \
    --debug \
    --resume


# # BLOCH
# python ../train_seg_stroke.py \
#     --name stroke-t1-cnn-simclr-bloch-pc1 \
#     --net cnn \
#     --modality t1 \
#     --amp \
#     --logdir ../ \
#     --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
#     --pc_data 1 \
#     --debug \
#     --resume

# python ../train_seg_stroke.py \
#     --name stroke-t2-cnn-simclr-bloch-pc1 \
#     --net cnn \
#     --modality t2 \
#     --amp \
#     --logdir ../ \
#     --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
#     --pc_data 1 \
#     --debug \
#     --resume

# python ../train_seg_stroke.py \
#     --name stroke-flair-cnn-simclr-bloch-pc1 \
#     --net cnn \
#     --modality flair \
#     --amp \
#     --logdir ../ \
#     --backbone_weights ../3d-cnn-simclr-bloch/checkpoint.pt \
#     --pc_data 1 \
#     --debug \
#     --resume


# # BLOCH PAIRED
# python ../train_seg_stroke.py \
#     --name stroke-t1-cnn-simclr-bloch-paired-pc1 \
#     --net cnn \
#     --modality t1 \
#     --amp \
#     --logdir ../ \
#     --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
#     --pc_data 1 \
#     --debug \
#     --resume

# python ../train_seg_stroke.py \
#     --name stroke-t2-cnn-simclr-bloch-paired-pc1 \
#     --net cnn \
#     --modality t2 \
#     --amp \
#     --logdir ../ \
#     --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
#     --pc_data 1 \
#     --debug \
#     --resume

# python ../train_seg_stroke.py \
#     --name stroke-flair-cnn-simclr-bloch-paired-pc1 \
#     --net cnn \
#     --modality flair \
#     --amp \
#     --logdir ../ \
#     --backbone_weights ../3d-cnn-simclr-bloch-paired/checkpoint.pt \
#     --pc_data 1 \
#     --debug \
#     --resume
