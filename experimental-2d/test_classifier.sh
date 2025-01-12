#!/bin/bash

# Base directory for all checkpoints
BASE_DIR="task_checkpoints"

# Model configurations (same as in train_classifier.sh)
declare -A MODELS=(
    ["random-resnet50"]="Random ResNet-50"
    ["imagenet-resnet50"]="ImageNet ResNet-50"
    ["mprage-resnet50-view2"]="MPRAGE View2 ResNet-50"
    ["mprage-resnet50-view5"]="MPRAGE View5 ResNet-50"
    ["mprage-resnet50-barlow"]="MPRAGE Barlow ResNet-50"
    ["mprage-resnet50-vicreg"]="MPRAGE VICReg ResNet-50"
    ["bloch-resnet50-view2"]="Bloch View2 ResNet-50"
    ["bloch-resnet50-view5"]="Bloch View5 ResNet-50"
    ["bloch-resnet50-barlow"]="Bloch Barlow ResNet-50"
    ["bloch-resnet50-vicreg"]="Bloch VICReg ResNet-50"
)

# Create results directory
RESULTS_DIR="task_results/classification"
mkdir -p "$RESULTS_DIR"

# Test each model
for MODEL_DIR in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$MODEL_DIR]}"
    echo "Testing model: $MODEL_NAME"
    
    python test_classifier.py \
        --model_dir "$BASE_DIR/$MODEL_DIR" \
        --model_name "$MODEL_NAME" \
        --output_file "$RESULTS_DIR/${MODEL_DIR}_results.csv"
done

# Combine all results
python -c "
import pandas as pd
import glob
files = glob.glob('$RESULTS_DIR/*_results.csv')
combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
combined.to_csv('$RESULTS_DIR/combined_results.csv', index=False)
"

echo "Testing complete. Results saved in $RESULTS_DIR/combined_results.csv"