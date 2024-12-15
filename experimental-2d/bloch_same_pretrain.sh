#!/bin/bash

# Set strict error handling
set -euo pipefail
trap 'echo "Error on line $LINENO"' ERR

# Configuration
WANDB_ENTITY="atlas-ploras"
CHECKPOINT_BASE="checkpoints"
DATASET="bloch"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper function for logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log "Running experiments with the Bloch dataset using the same contrast for both/all views"

# ResNet50 Experiments
log "Running ResNet50 experiments with different views"
# 2 views
log "Running ResNet50 SimCLR with 2 views"
python pretrain.py \
    --dataset $DATASET \
    --model_name timm/resnet50.a1_in1k \
    --pretrained \
    --same_contrast \
    --wandb_entity $WANDB_ENTITY \
    --checkpoint_dir $CHECKPOINT_BASE/$DATASET-resnet50-view2-same-contrast \
    --wandb_name $DATASET-resnet50-view2-same-contrast
# 3 views
log "Running ResNet50 SimCLR with 3 views"
python pretrain.py \
    --dataset $DATASET \
    --model_name timm/resnet50.a1_in1k \
    --pretrained \
    --same_contrast \
    --wandb_entity $WANDB_ENTITY \
    --checkpoint_dir $CHECKPOINT_BASE/$DATASET-resnet50-view3-same-contrast \
    --wandb_name $DATASET-resnet50-view3-same-contrast
# 4 views
log "Running ResNet50 SimCLR with 4 views"
python pretrain.py \
    --dataset $DATASET \
    --model_name timm/resnet50.a1_in1k \
    --pretrained \
    --same_contrast \
    --wandb_entity $WANDB_ENTITY \
    --checkpoint_dir $CHECKPOINT_BASE/$DATASET-resnet50-view4-same-contrast \
    --wandb_name $DATASET-resnet50-view4-same-contrast
# 5 views
log "Running ResNet50 SimCLR with 5 views"
python pretrain.py \
    --dataset $DATASET \
    --model_name timm/resnet50.a1_in1k \
    --pretrained \
    --same_contrast \
    --wandb_entity $WANDB_ENTITY \
    --checkpoint_dir $CHECKPOINT_BASE/$DATASET-resnet50-view5-same-contrast \
    --wandb_name $DATASET-resnet50-view5-same-contrast
# Barlow Twins
log "Running ResNet50 Barlow Twins"
python pretrain.py \
    --dataset $DATASET \
    --model_name timm/resnet50.a1_in1k \
    --pretrained \
    --same_contrast \
    --wandb_entity $WANDB_ENTITY \
    --checkpoint_dir $CHECKPOINT_BASE/$DATASET-resnet50-barlow-same-contrast \
    --wandb_name $DATASET-resnet50-barlow-same-contrast \
    --loss_type barlow
# VICReg
log "Running ResNet50 VICReg"
python pretrain.py \
    --dataset $DATASET \
    --model_name timm/resnet50.a1_in1k \
    --pretrained \
    --same_contrast \
    --wandb_entity $WANDB_ENTITY \
    --checkpoint_dir $CHECKPOINT_BASE/$DATASET-resnet50-vicreg-same-contrast \
    --wandb_name $DATASET-resnet50-vicreg-same-contrast \
    --loss_type vicreg

# CLIP ViT-B/16
# 2 views
log "Running CLIP ViT-B/16 with 2 views"
python pretrain.py \
    --dataset $DATASET \
    --model_name timm/vit_base_patch16_clip_224.openai \
    --pretrained \
    --same_contrast \
    --wandb_entity $WANDB_ENTITY \
    --checkpoint_dir $CHECKPOINT_BASE/$DATASET-vitclip-view2-same-contrast \
    --wandb_name $DATASET-vitclip-view2-same-contrast

# DINO ViT-B/16
# 2 views
log "Running DINO ViT-B/16 with 2 views"
python pretrain.py \
    --dataset $DATASET \
    --model_name timm/vit_base_patch16_224.dino \
    --pretrained \
    --same_contrast \
    --wandb_entity $WANDB_ENTITY \
    --checkpoint_dir $CHECKPOINT_BASE/$DATASET-vitdino-view2-same-contrast \
    --wandb_name $DATASET-vitdino-view2-same-contrast