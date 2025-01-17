#! /bin/bash

## 100% of training data
# t1
python ../test_sex_classification.py \
    --weights ../sex-t1-cnn-simclr-mprage-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../test_sex_classification.py \
    --weights ../sex-t1-cnn-simclr-bloch-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../test_sex_classification.py \
    --weights ../sex-t1-cnn-simclr-bloch-paired-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
# t2
python ../test_sex_classification.py \
    --weights ../sex-t2-cnn-simclr-mprage-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../test_sex_classification.py \
    --weights ../sex-t2-cnn-simclr-bloch-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../test_sex_classification.py \
    --weights ../sex-t2-cnn-simclr-bloch-paired-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
# pd
python ../test_sex_classification.py \
    --weights ../sex-pd-cnn-simclr-mprage-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd
python ../test_sex_classification.py \
    --weights ../sex-pd-cnn-simclr-bloch-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd
python ../test_sex_classification.py \
    --weights ../sex-pd-cnn-simclr-bloch-paired-pc100/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd

## 10% of training data
# t1
python ../test_sex_classification.py \
    --weights ../sex-t1-cnn-simclr-mprage-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../test_sex_classification.py \
    --weights ../sex-t1-cnn-simclr-bloch-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../test_sex_classification.py \
    --weights ../sex-t1-cnn-simclr-bloch-paired-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
# t2
python ../test_sex_classification.py \
    --weights ../sex-t2-cnn-simclr-mprage-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../test_sex_classification.py \
    --weights ../sex-t2-cnn-simclr-bloch-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../test_sex_classification.py \
    --weights ../sex-t2-cnn-simclr-bloch-paired-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
# pd
python ../test_sex_classification.py \
    --weights ../sex-pd-cnn-simclr-mprage-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd
python ../test_sex_classification.py \
    --weights ../sex-pd-cnn-simclr-bloch-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd
python ../test_sex_classification.py \
    --weights ../sex-pd-cnn-simclr-bloch-paired-pc10/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd

## 1% of training data
# t1
python ../test_sex_classification.py \
    --weights ../sex-t1-cnn-simclr-mprage-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../test_sex_classification.py \
    --weights ../sex-t1-cnn-simclr-bloch-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
python ../test_sex_classification.py \
    --weights ../sex-t1-cnn-simclr-bloch-paired-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t1
# t2
python ../test_sex_classification.py \
    --weights ../sex-t2-cnn-simclr-mprage-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../test_sex_classification.py \
    --weights ../sex-t2-cnn-simclr-bloch-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
python ../test_sex_classification.py \
    --weights ../sex-t2-cnn-simclr-bloch-paired-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality t2
# pd
python ../test_sex_classification.py \
    --weights ../sex-pd-cnn-simclr-mprage-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd
python ../test_sex_classification.py \
    --weights ../sex-pd-cnn-simclr-bloch-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd
python ../test_sex_classification.py \
    --weights ../sex-pd-cnn-simclr-bloch-paired-pc1/checkpoint.pt \
    --net cnn \
    --amp \
    --modality pd 