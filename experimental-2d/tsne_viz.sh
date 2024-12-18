#!/bin/bash

# Define paths and parameters
H5_PATH="task_data/ixi.h5"
OUTPUT_DIR="output/tsne_plots"
PERPLEXITY=30
N_ITER=1000

mkdir -p $OUTPUT_DIR

echo "Generating t-SNE plots for the IXI dataset using ResNet50"

echo "Using random init."
MODEL_OUTPUT_DIR="$OUTPUT_DIR/random_init"
mkdir -p $MODEL_OUTPUT_DIR
python visualize_ixi_tsne.py \
    --model_name timm/resnet50.a1_in1k \
    --output_dir $MODEL_OUTPUT_DIR \
    --perplexity $PERPLEXITY \
    --n_iter $N_ITER \
    --name random_init

echo "Using imagenet weights."
MODEL_OUTPUT_DIR="$OUTPUT_DIR/imagenet"
mkdir -p $MODEL_OUTPUT_DIR
python visualize_ixi_tsne.py \
    --model_name timm/resnet50.a1_in1k \
    --pretrained \
    --output_dir $MODEL_OUTPUT_DIR \
    --perplexity $PERPLEXITY \
    --n_iter $N_ITER \
    --name imagenet

# Iterate over all folders in checkpoints/
for MODEL_DIR in checkpoints/*; do
    if [ -d "$MODEL_DIR" ]; then
        MODEL_NAME=$(basename "$MODEL_DIR")
        WEIGHTS_PATH="$MODEL_DIR/latest_model.pt"
        MODEL_OUTPUT_DIR="$OUTPUT_DIR/$MODEL_NAME"
        
        mkdir -p $MODEL_OUTPUT_DIR
        
        echo "Generating t-SNE for model: $MODEL_NAME"
        python visualize_ixi_tsne.py \
            --h5_path $H5_PATH \
            --model_name timm/resnet50.a1_in1k \
            --weights_path $WEIGHTS_PATH \
            --output_dir $MODEL_OUTPUT_DIR \
            --perplexity $PERPLEXITY \
            --n_iter $N_ITER \
            --name $MODEL_NAME
    fi
done

echo "t-SNE generation completed for all models."