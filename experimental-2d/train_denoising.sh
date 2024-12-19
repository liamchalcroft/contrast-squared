#!/bin/bash

# Base directory for all checkpoints
BASE_DIR="task_checkpoints"

# Model configurations (name, weights path)
declare -A MODELS=(
    ["random-resnet50"]="timm/resnet50.a1_in1k"
    ["imagenet-resnet50"]="timm/resnet50.a1_in1k --pretrained"
    ["mprage-resnet50-view2"]="timm/resnet50.a1_in1k --weights_path checkpoints/mprage-resnet50-view2/model_best.pth"
    ["mprage-resnet50-view5"]="timm/resnet50.a1_in1k --weights_path checkpoints/mprage-resnet50-view5/model_best.pth"
    ["mprage-resnet50-barlow"]="timm/resnet50.a1_in1k --weights_path checkpoints/mprage-resnet50-barlow/model_best.pth"
    ["mprage-resnet50-vicreg"]="timm/resnet50.a1_in1k --weights_path checkpoints/mprage-resnet50-vicreg/model_best.pth"
    ["bloch-resnet50-view2"]="timm/resnet50.a1_in1k --weights_path checkpoints/bloch-resnet50-view2/model_best.pth"
    ["bloch-resnet50-view5"]="timm/resnet50.a1_in1k --weights_path checkpoints/bloch-resnet50-view5/model_best.pth"
    ["bloch-resnet50-barlow"]="timm/resnet50.a1_in1k --weights_path checkpoints/bloch-resnet50-barlow/model_best.pth"
    ["bloch-resnet50-vicreg"]="timm/resnet50.a1_in1k --weights_path checkpoints/bloch-resnet50-vicreg/model_best.pth"
)

# Modalities to process
MODALITIES=("t1" "t2" "pd")

# Sites to process - only GST for training, the rest are for testing
SITES=("GST")

# Common training parameters - try 20 epochs first
EPOCHS=20
BATCH_SIZE=32
LR=1e-3

# Create base directory if it doesn't exist
mkdir -p "$BASE_DIR"

# Loop through all combinations
for MODEL_NAME in "${!MODELS[@]}"; do
    MODEL_ARGS="${MODELS[$MODEL_NAME]}"
    
    for MODALITY in "${MODALITIES[@]}"; do
        for SITE in "${SITES[@]}"; do
            echo "Training model: $MODEL_NAME ($MODEL_ARGS) on $MODALITY data from $SITE"
            
            # Create checkpoint directory
            CHECKPOINT_DIR="$BASE_DIR/${MODEL_NAME}/"
            mkdir -p "$CHECKPOINT_DIR"
            
            # Run training with specified parameters
            python train_denoising.py \
                --output_dir "$CHECKPOINT_DIR" \
                --model_name $MODEL_ARGS \
                --modality "$MODALITY" \
                --site "$SITE" \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --learning_rate $LR \
                --amp \
                --resume \
                2>&1 | tee "$CHECKPOINT_DIR/training_${MODALITY}_${SITE}.log"
            
            # Check if training completed successfully
            if [ $? -eq 0 ]; then
                echo "Successfully completed training for $MODEL_NAME on $MODALITY data from $SITE"
            else
                echo "Error training $MODEL_NAME on $MODALITY data from $SITE"
                # Optionally, you could add error handling here
            fi
        done
    done
done 