#!/bin/bash

# Base directory for all checkpoints
BASE_DIR="task_checkpoints"

# Model configurations (name, weights path)
declare -A MODELS=(
    ["random-resnet50"]="timm/resnet50.a1_in1k"
    ["imagenet-resnet50"]="timm/resnet50.a1_in1k --pretrained"
    ["mprage-resnet50-view2"]="timm/resnet50.a1_in1k --weights_path checkpoints/mprage-resnet50-view2/best_model.pt"
    ["mprage-resnet50-view5"]="timm/resnet50.a1_in1k --weights_path checkpoints/mprage-resnet50-view5/best_model.pt"
    ["mprage-resnet50-barlow"]="timm/resnet50.a1_in1k --weights_path checkpoints/mprage-resnet50-barlow/best_model.pt"
    ["mprage-resnet50-vicreg"]="timm/resnet50.a1_in1k --weights_path checkpoints/mprage-resnet50-vicreg/best_model.pt"
    ["bloch-resnet50-view2"]="timm/resnet50.a1_in1k --weights_path checkpoints/bloch-resnet50-view2/best_model.pt"
    ["bloch-resnet50-view5"]="timm/resnet50.a1_in1k --weights_path checkpoints/bloch-resnet50-view5/best_model.pt"
    ["bloch-resnet50-barlow"]="timm/resnet50.a1_in1k --weights_path checkpoints/bloch-resnet50-barlow/best_model.pt"
    ["bloch-resnet50-vicreg"]="timm/resnet50.a1_in1k --weights_path checkpoints/bloch-resnet50-vicreg/best_model.pt"
)

# Modalities to process
MODALITIES=("t1" "t2" "pd")

# Sites to process - only GST for training, the rest are for testing
SITES=("GST")

# Common training parameters
EPOCHS=50
BATCH_SIZE=32
LR=1e-3

# Create base directory if it doesn't exist
mkdir -p "$BASE_DIR"

# Loop through all combinations
for MODEL_NAME in "${!MODELS[@]}"; do
    MODEL_ARGS="${MODELS[$MODEL_NAME]}"
    
    for MODALITY in "${MODALITIES[@]}"; do
        for SITE in "${SITES[@]}"; do
            echo "Training regression model: $MODEL_NAME ($MODEL_ARGS) on $MODALITY data from $SITE"
            
            # Create checkpoint directory
            CHECKPOINT_DIR="$BASE_DIR/${MODEL_NAME}/"
            mkdir -p "$CHECKPOINT_DIR"
            
            # Run training with specified parameters
            python train_regression.py \
                --output_dir "$CHECKPOINT_DIR" \
                --model_name $MODEL_ARGS \
                --modality "$MODALITY" \
                --site "$SITE" \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --learning_rate $LR \
                --amp \
                2>&1 | tee "$CHECKPOINT_DIR/regression_${MODALITY}_${SITE}.log"
            
            # Check if training completed successfully
            if [ $? -eq 0 ]; then
                echo "Successfully completed regression training for $MODEL_NAME on $MODALITY data from $SITE"
            else
                echo "Error training regression model for $MODEL_NAME on $MODALITY data from $SITE"
            fi
        done
    done
done 