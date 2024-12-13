import monai as mn
import glob
import os
from torch.utils.data import DataLoader
import torch
from random import shuffle, seed, sample
from PIL import Image
import numpy as np

seed(786)

def load_png_as_tensor(path):
    img = Image.open(path)
    return torch.from_numpy(np.array(img)).float() / 255.0

def get_augmentations(keys, size):
    return [
        mn.transforms.RandAffineD(
            keys=keys,
            rotate_range=(-15, 15),
            scale_range=(0.85, 1.15),
            translate_range=(-0.1, 0.1),
            padding_mode='zeros',
            prob=0.8
        ),
        mn.transforms.RandAxisFlipd(keys=keys, prob=0.5),
        mn.transforms.RandRotate90d(keys=keys, prob=0.5),
        mn.transforms.ResizeWithPadOrCropD(
            keys=keys, spatial_size=(size, size)
        ),
        # Lighter intensity augmentations
        mn.transforms.RandGaussianNoiseD(keys=keys, prob=0.5, mean=0.0, std=0.02),
        mn.transforms.RandAdjustContrastD(keys=keys, prob=0.5, gamma=(0.8, 1.2)),
        mn.transforms.RandGaussianSmoothD(keys=keys, prob=0.5, sigma_x=(0.25, 1.5)),
    ]

def get_prepare_data(keys, size):
    prepare_data = [
        mn.transforms.NormalizeIntensityD(keys=keys),
        mn.transforms.ResizeD(
            keys=keys, spatial_size=(size, size)
        ),
        mn.transforms.ToTensorD(
            dtype=torch.float32, keys=keys
        ),
    ]
    return prepare_data

def get_bloch_loader(
    batch_size=1,
    device="cpu",
    lowres=False,
    same_contrast=False,
    num_views=2,
):
    if num_views < 2:
        raise ValueError("num_views must be at least 2")

    # Get all slice files
    slice_files = glob.glob(os.path.join("output/qmri_slices/*.png"))
    
    # Group files by slice (removing contrast number)
    slice_groups = {}
    for f in slice_files:
        base_name = "_".join(os.path.basename(f).split("_")[:2])  # subject_id + axis
        if base_name not in slice_groups:
            slice_groups[base_name] = []
        slice_groups[base_name].append(f)

    # Create training dictionary
    train_dict = []
    for base_name, contrasts in slice_groups.items():
        if same_contrast:
            # Use the same contrast for all views
            contrast = sample(contrasts, 1)[0]
            train_dict.append({f"image{i+1}": contrast for i in range(num_views)})
        else:
            # Use different contrasts for each view
            selected_contrasts = sample(contrasts, num_views)
            train_dict.append({f"image{i+1}": contrast for i, contrast in enumerate(selected_contrasts)})

    shuffle(train_dict)

    size = 48 if lowres else 96
    image_keys = [f"image{i+1}" for i in range(num_views)]

    # Prepare transforms
    prepare_images = [
        mn.transforms.LoadImageD(keys=image_keys, reader="PILReader"),
        mn.transforms.EnsureChannelFirstD(keys=image_keys),
    ]

    # Apply augmentations to each view
    all_augmentations = []
    for i in range(num_views):
        aug = get_augmentations([f"image{i+1}"], size)
        all_augmentations.extend(aug)

    # Final preparation
    prepare_all = get_prepare_data(image_keys, size)

    train_transform = mn.transforms.Compose(
        transforms=[
            *prepare_images,
            *all_augmentations,
            *prepare_all,
        ]
    )

    train_data = mn.data.Dataset(train_dict, transform=train_transform)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    return train_loader, train_transform

def get_mprage_loader(
    batch_size=1,
    device="cpu",
    lowres=False,
    num_views=2,
):
    if num_views < 2:
        raise ValueError("num_views must be at least 2")

    # Get all slice files
    slice_files = glob.glob(os.path.join("output/mprage_slices/*.png"))
    
    # Create training dictionary - use the same slice for all views
    train_dict = [{f"image{i+1}": f for i in range(num_views)} for f in slice_files]
    shuffle(train_dict)

    size = 48 if lowres else 96
    image_keys = [f"image{i+1}" for i in range(num_views)]

    # Prepare transforms
    prepare_images = [
        mn.transforms.LoadImageD(keys=image_keys, reader="PILReader"),
        mn.transforms.EnsureChannelFirstD(keys=image_keys),
    ]

    # Apply augmentations to each view
    all_augmentations = []
    for i in range(num_views):
        aug = get_augmentations([f"image{i+1}"], size)
        all_augmentations.extend(aug)

    # Final preparation
    prepare_all = get_prepare_data(image_keys, size)

    train_transform = mn.transforms.Compose(
        transforms=[
            *prepare_images,
            *all_augmentations,
            *prepare_all,
        ]
    )

    train_data = mn.data.Dataset(train_dict, transform=train_transform)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    return train_loader, train_transform
