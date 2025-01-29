#! /bin/bash

## 100% of training data
# t1
python ../src/test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-mprage-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../src/test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-bloch-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../src/test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-bloch-paired-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
# t2
python ../src/test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-mprage-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../src/test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-bloch-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../src/test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-bloch-paired-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
# flair
python ../src/test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-mprage-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality flair
python ../src/test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-bloch-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality flair
python ../src/test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-bloch-paired-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality flair


## 10% of training data
# t1
python ../src/test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-mprage-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../src/test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-bloch-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../src/test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-bloch-paired-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
# t2
python ../src/test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-mprage-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../src/test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-bloch-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../src/test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-bloch-paired-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
# flair
python ../src/test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-mprage-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality flair
python ../src/test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-bloch-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality flair
python ../src/test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-bloch-paired-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality flair


## 1% of training data
# t1
python ../src/test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-mprage-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../src/test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-bloch-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../src/test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-bloch-paired-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
# t2
python ../src/test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-mprage-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../src/test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-bloch-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../src/test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-bloch-paired-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
# flair
python ../src/test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-mprage-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality flair
python ../src/test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-bloch-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality flair
python ../src/test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-bloch-paired-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality flair
