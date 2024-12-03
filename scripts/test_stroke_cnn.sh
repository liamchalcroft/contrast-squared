#! /bin/bash

## 100% of training data
# t1
python ../test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-mprage-pc100/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t1
python ../test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-bloch-pc100/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t1
python ../test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-bloch-paired-pc100/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t1
# t2
python ../test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-mprage-pc100/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t2
python ../test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-bloch-pc100/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t2
python ../test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-bloch-paired-pc100/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t2
# flair
python ../test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-mprage-pc100/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality flair
python ../test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-bloch-pc100/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality flair
python ../test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-bloch-paired-pc100/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality flair


## 10% of training data
# t1
python ../test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-mprage-pc10/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t1
python ../test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-bloch-pc10/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t1
python ../test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-bloch-paired-pc10/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t1
# t2
python ../test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-mprage-pc10/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t2
python ../test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-bloch-pc10/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t2
python ../test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-bloch-paired-pc10/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t2
# flair
python ../test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-mprage-pc10/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality flair
python ../test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-bloch-pc10/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality flair
python ../test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-bloch-paired-pc10/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality flair


## 1% of training data
# t1
python ../test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-mprage-pc1/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t1
python ../test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-bloch-pc1/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t1
python ../test_seg_stroke.py \
    --weights ../stroke-t1-cnn-simclr-bloch-paired-pc1/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t1
# t2
python ../test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-mprage-pc1/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t2
python ../test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-bloch-pc1/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t2
python ../test_seg_stroke.py \
    --weights ../stroke-t2-cnn-simclr-bloch-paired-pc1/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality t2
# flair
python ../test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-mprage-pc1/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality flair
python ../test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-bloch-pc1/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality flair
python ../test_seg_stroke.py \
    --weights ../stroke-flair-cnn-simclr-bloch-paired-pc1/checkpoint_best.pt \
    --net cnn \
    --amp \
    --modality flair
