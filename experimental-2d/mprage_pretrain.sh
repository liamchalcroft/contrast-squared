# ResNet50
# 2 views
python pretrain.py --dataset mprage --model_name timm/resnet50.a1_in1k --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/mprage-resnet50-view2 --wandb_name mprage-resnet50-view2
# 3 views
python pretrain.py --dataset mprage --model_name timm/resnet50.a1_in1k --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/mprage-resnet50-view3 --wandb_name mprage-resnet50-view3
# 4 views
python pretrain.py --dataset mprage --model_name timm/resnet50.a1_in1k --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/mprage-resnet50-view4 --wandb_name mprage-resnet50-view4
# 5 views
python pretrain.py --dataset mprage --model_name timm/resnet50.a1_in1k --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/mprage-resnet50-view5 --wandb_name mprage-resnet50-view5
# Barlow Twins
python pretrain.py --dataset mprage --model_name timm/resnet50.a1_in1k --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/mprage-resnet50-barlow --wandb_name mprage-resnet50-barlow --loss_type barlow
# VICReg
python pretrain.py --dataset mprage --model_name timm/resnet50.a1_in1k --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/mprage-resnet50-vicreg --wandb_name mprage-resnet50-vicreg --loss_type vicreg

# CLIP ViT-B/16
# 2 views
python pretrain.py --dataset mprage --model_name timm/vit_base_patch16_clip_224.openai --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/mprage-vitclip-view2 --wandb_name mprage-vitclip-view2
# # 3 views
# python pretrain.py --dataset mprage --model_name timm/vit_base_patch16_clip_224.openai --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/mprage-vitclip-view3 --wandb_name mprage-vitclip-view3
# # 4 views
# python pretrain.py --dataset mprage --model_name timm/vit_base_patch16_clip_224.openai --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/mprage-vitclip-view4 --wandb_name mprage-vitclip-view4
# # 5 views
# python pretrain.py --dataset mprage --model_name timm/vit_base_patch16_clip_224.openai --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/mprage-vitclip-view5 --wandb_name mprage-vitclip-view5

# DINO ViT-B/16
# 2 views
python pretrain.py --dataset mprage --model_name timm/vit_base_patch16_224.dino --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/mprage-vitdino-view2 --wandb_name mprage-vitdino-view2
# # 3 views
# python pretrain.py --dataset mprage --model_name timm/vit_base_patch16_224.dino --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/mprage-vitdino-view3 --wandb_name mprage-vitdino-view3
# # 4 views
# python pretrain.py --dataset mprage --model_name timm/vit_base_patch16_224.dino --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/mprage-vitdino-view4 --wandb_name mprage-vitdino-view4
# # 5 views
# python pretrain.py --dataset mprage --model_name timm/vit_base_patch16_224.dino --pretrained --wandb_entity atlas-ploras --checkpoint_dir checkpoints/mprage-vitdino-view5 --wandb_name mprage-vitdino-view5

