# ResNet50
# 2 views
python pretrain.py --dataset bloch --same_contrast --model_name timm/resnet50.a1_in1k --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/bloch-same-resnet50-view2 --wandb_name bloch-same-resnet50-view2
# 3 views
python pretrain.py --dataset bloch --same_contrast --model_name timm/resnet50.a1_in1k --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/bloch-same-resnet50-view3 --wandb_name bloch-same-resnet50-view3
# 4 views
python pretrain.py --dataset bloch --same_contrast --model_name timm/resnet50.a1_in1k --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/bloch-same-resnet50-view4 --wandb_name bloch-same-resnet50-view4
# 5 views
python pretrain.py --dataset bloch --same_contrast --model_name timm/resnet50.a1_in1k --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/bloch-same-resnet50-view5 --wandb_name bloch-same-resnet50-view5

# CLIP ViT-B/16
# 2 views
python pretrain.py --dataset bloch --same_contrast --model_name timm/vit_base_patch16_clip_224.openai --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/bloch-same-vitclip-view2 --wandb_name bloch-same-vitclip-view2
# # 3 views
# python pretrain.py --dataset bloch --same_contrast --model_name timm/vit_base_patch16_clip_224.openai --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/bloch-same-vitclip-view3 --wandb_name bloch-same-vitclip-view3
# # 4 views
# python pretrain.py --dataset bloch --same_contrast --model_name timm/vit_base_patch16_clip_224.openai --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/bloch-same-vitclip-view4 --wandb_name bloch-same-vitclip-view4
# # 5 views
# python pretrain.py --dataset bloch --same_contrast --model_name timm/vit_base_patch16_clip_224.openai --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/bloch-same-vitclip-view5 --wandb_name bloch-same-vitclip-view5

# DINO ViT-B/16
# 2 views
python pretrain.py --dataset bloch --same_contrast --model_name timm/vit_base_patch16_224.dino --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/bloch-same-vitdino-view2 --wandb_name bloch-same-vitdino-view2
# # 3 views
# python pretrain.py --dataset bloch --same_contrast --model_name timm/vit_base_patch16_224.dino --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/bloch-same-vitdino-view3 --wandb_name bloch-same-vitdino-view3
# # 4 views
# python pretrain.py --dataset bloch --same_contrast --model_name timm/vit_base_patch16_224.dino --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/bloch-same-vitdino-view4 --wandb_name bloch-same-vitdino-view4
# # 5 views
# python pretrain.py --dataset bloch --same_contrast --model_name timm/vit_base_patch16_224.dino --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/bloch-same-vitdino-view5 --wandb_name bloch-same-vitdino-view5

