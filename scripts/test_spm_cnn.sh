#! /bin/bash

## 100% of training data
# t1
python ../src/test_seg_healthy.py \
    --weights ../spm-t1-cnn-simclr-mprage-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../src/test_seg_healthy.py \
    --weights ../spm-t1-cnn-simclr-bloch-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../src/test_seg_healthy.py \
    --weights ../spm-t1-cnn-simclr-bloch-paired-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
# t2
python ../src/test_seg_healthy.py \
    --weights ../spm-t2-cnn-simclr-mprage-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../src/test_seg_healthy.py \
    --weights ../spm-t2-cnn-simclr-bloch-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../src/test_seg_healthy.py \
    --weights ../spm-t2-cnn-simclr-bloch-paired-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
# pd
python ../src/test_seg_healthy.py \
    --weights ../spm-pd-cnn-simclr-mprage-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd
python ../src/test_seg_healthy.py \
    --weights ../spm-pd-cnn-simclr-bloch-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd
python ../src/test_seg_healthy.py \
    --weights ../spm-pd-cnn-simclr-bloch-paired-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd


## 10% of training data
# t1
python ../src/test_seg_healthy.py \
    --weights ../spm-t1-cnn-simclr-mprage-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../src/test_seg_healthy.py \
    --weights ../spm-t1-cnn-simclr-bloch-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../src/test_seg_healthy.py \
    --weights ../spm-t1-cnn-simclr-bloch-paired-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
# t2
python ../src/test_seg_healthy.py \
    --weights ../spm-t2-cnn-simclr-mprage-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../src/test_seg_healthy.py \
    --weights ../spm-t2-cnn-simclr-bloch-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../src/test_seg_healthy.py \
    --weights ../spm-t2-cnn-simclr-bloch-paired-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
# pd
python ../src/test_seg_healthy.py \
    --weights ../spm-pd-cnn-simclr-mprage-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd
python ../src/test_seg_healthy.py \
    --weights ../spm-pd-cnn-simclr-bloch-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd
python ../src/test_seg_healthy.py \
    --weights ../spm-pd-cnn-simclr-bloch-paired-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd


## 1% of training data
# t1
python ../src/test_seg_healthy.py \
    --weights ../spm-t1-cnn-simclr-mprage-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../src/test_seg_healthy.py \
    --weights ../spm-t1-cnn-simclr-bloch-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../src/test_seg_healthy.py \
    --weights ../spm-t1-cnn-simclr-bloch-paired-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
# t2
python ../src/test_seg_healthy.py \
    --weights ../spm-t2-cnn-simclr-mprage-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../src/test_seg_healthy.py \
    --weights ../spm-t2-cnn-simclr-bloch-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../src/test_seg_healthy.py \
    --weights ../spm-t2-cnn-simclr-bloch-paired-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
# pd
python ../src/test_seg_healthy.py \
    --weights ../spm-pd-cnn-simclr-mprage-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd
python ../src/test_seg_healthy.py \
    --weights ../spm-pd-cnn-simclr-bloch-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd
python ../src/test_seg_healthy.py \
    --weights ../spm-pd-cnn-simclr-bloch-paired-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd