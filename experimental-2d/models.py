import torch
import torchseg as ts
import timm
import torch.nn as nn
import logging

def create_unet_model(model_name: str, weights_path: str = None, pretrained: bool = True) -> ts.Unet:
    """Create a U-Net model with a specific backbone encoder.
    
    This function creates a U-Net model using either a ResNet or ViT backbone,
    and optionally loads pretrained weights from either a local path or the timm model hub.

    Args:
        model_name: Name of the model from timm model hub (e.g. 'timm/resnet50.a1_in1k')
        weights_path: Optional path to local weights file. If provided, these weights
            are loaded instead of pretrained weights from timm.
        pretrained: Whether to load pretrained weights from timm if weights_path is None.
            Default is True.

    Returns:
        ts.Unet: A U-Net model with the specified backbone and loaded weights.

    Raises:
        ValueError: If the model_name is not a supported ResNet or ViT variant.
    
    Examples:
        >>> model = create_model('timm/resnet50.a1_in1k')  # ResNet with pretrained weights
        >>> model = create_model('timm/vit_base_patch16_224.dino')  # ViT with pretrained weights
        >>> model = create_model('timm/resnet50.a1_in1k', 'weights.pt')  # Load custom weights
    """
    # Extract the model type
    if 'resnet' in model_name.lower():
        model_type = 'resnet50'
    elif 'vit' in model_name.lower():
        model_type = 'vit_base_patch16_224'
    else:
        raise ValueError(f"Model type not supported: {model_name}")
    
    # Create model with appropriate config
    if 'vit' in model_type.lower():
        model = ts.Unet(
            model_type,
            in_channels=1,
            classes=1,
            encoder_depth=5,
            encoder_indices=(2, 4, 6, 8, 10),
            encoder_weights=None,
            decoder_channels=(256, 128, 64, 32, 16),
            encoder_params={
                "scale_factors": (16, 8, 4, 2, 1),
                "img_size": 224,
            },
        )
    else:
        model = ts.Unet(
            model_type,
            in_channels=1, 
            classes=1,
            encoder_depth=5,
            encoder_weights=None,
            decoder_channels=(256, 128, 64, 32, 16),
        )

    # Load weights
    if weights_path:
        model = torch.compile(model)
        # Weights are for compiled model
        state_dict = torch.load(weights_path)['model_state_dict']
        # Filter out the _orig_mod. torch compile prefix from all keys
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    elif pretrained:
        state_dict = timm.create_model(model_name, pretrained=True).state_dict()
        model.load_state_dict(state_dict, strict=False)
        model = torch.compile(model)
    else:
        model = torch.compile(model)

    return model

def create_classification_model(model_name: str, num_classes: int, weights_path: str = None, pretrained: bool = True) -> nn.Module:
    """Create a classification model with a specific backbone encoder."""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=1)

    # Get encoder output dimension
    with torch.no_grad():
        dummy_input = torch.zeros(1, 1, 224, 224)
        output = model(dummy_input)
        encoder_dim = output.shape[1]

    # Load weights if provided
    if weights_path:
        state_dict = torch.load(weights_path)['model_state_dict']
        # Filter out the _orig_mod. torch compile prefix from all keys
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    
    # Add a linear classification head
    classifier = nn.Sequential(
        model,
        nn.Flatten(),
        nn.Linear(encoder_dim, num_classes)
    )

    classifier = torch.compile(classifier)
    
    return classifier