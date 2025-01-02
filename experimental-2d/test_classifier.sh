#!/bin/bash

# Base directory for all checkpoints
BASE_DIR="task_checkpoints"

# Model configurations (name, weights path)
declare -A MODELS=(
    ["random-resnet50"]="timm/resnet50.a1_in1k"
    ["imagenet-resnet50"]="timm/resnet50.a1_in1k"
    ["mprage-resnet50-view2"]="timm/resnet50.a1_in1k"
    ["mprage-resnet50-view5"]="timm/resnet50.a1_in1k"
    ["mprage-resnet50-barlow"]="timm/resnet50.a1_in1k"
    ["mprage-resnet50-vicreg"]="timm/resnet50.a1_in1k"
    ["bloch-resnet50-view2"]="timm/resnet50.a1_in1k"
    ["bloch-resnet50-view5"]="timm/resnet50.a1_in1k"
    ["bloch-resnet50-barlow"]="timm/resnet50.a1_in1k"
    ["bloch-resnet50-vicreg"]="timm/resnet50.a1_in1k"
)

# Modalities to process
MODALITIES=("t1" "t2" "pd")

# Sites to process - test on all sites
SITES=("GST" "HH" "IOP")

# Common parameters
BATCH_SIZE=32

# Loop through all combinations
for MODEL_NAME in "${!MODELS[@]}"; do
    MODEL_ARGS="${MODELS[$MODEL_NAME]}"
    
    # Get checkpoint directory
    CHECKPOINT_DIR="$BASE_DIR/${MODEL_NAME}/"
    
    # Skip if checkpoint directory doesn't exist
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "Skipping $MODEL_NAME - checkpoint directory not found"
        continue
    fi
    
    for MODALITY in "${MODALITIES[@]}"; do
        for SITE in "${SITES[@]}"; do
            echo "Testing classifier: $MODEL_NAME on $MODALITY data from $SITE"
            
            # Check if model checkpoint exists
            if [ ! -f "$CHECKPOINT_DIR/classifier_${MODALITY}_GST_best.pth" ]; then
                echo "Skipping - no checkpoint found for $MODEL_NAME on $MODALITY"
                continue
            fi
            
            # Run testing with specified parameters
            python test_classifier.py \
                --checkpoint_dir "$CHECKPOINT_DIR" \
                --model_name $MODEL_ARGS \
                --modality "$MODALITY" \
                --site "$SITE" \
                --batch_size $BATCH_SIZE \
                2>&1 | tee "$CHECKPOINT_DIR/test_${MODALITY}_${SITE}.log"
            
            # Check if testing completed successfully
            if [ $? -eq 0 ]; then
                echo "Successfully completed testing for $MODEL_NAME on $MODALITY data from $SITE"
            else
                echo "Error testing $MODEL_NAME on $MODALITY data from $SITE"
            fi
        done
    done
done

# Combine all results into a single CSV
echo "Combining results..."
COMBINED_RESULTS="$BASE_DIR/combined_test_results.csv"

# Find the first CSV file to get headers
FIRST_CSV=$(find "$BASE_DIR" -name "test_results_*.csv" | head -n 1)

if [ -n "$FIRST_CSV" ]; then
    # Copy headers from first file
    head -n 1 "$FIRST_CSV" > "$COMBINED_RESULTS"
    
    # Append all results (excluding headers)
    find "$BASE_DIR" -name "test_results_*.csv" -exec tail -n +2 {} \; >> "$COMBINED_RESULTS"
    
    echo "Combined results saved to $COMBINED_RESULTS"
else
    echo "No results files found"
fi 