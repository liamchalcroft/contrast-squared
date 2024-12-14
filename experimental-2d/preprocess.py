import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F, Transform
import h5py
from random import shuffle, seed, sample
import numpy as np
from typing import Any, Dict, List

seed(786)

class H5SliceDataset(Dataset):
    def __init__(self, h5_path, transform=None, same_contrast=False, num_views=2):
        self.h5_path = h5_path
        self.transform = transform
        self.same_contrast = same_contrast
        self.num_views = num_views
        
        # Create index mapping
        self.index_map = []
        with h5py.File(h5_path, 'r') as f:
            for subject in f.keys():
                if 'contrasts' in f[subject].keys():  # qMRI data
                    num_slices = f[subject]['contrasts'].shape[1]  # [num_contrasts, num_slices, H, W]
                    for slice_idx in range(num_slices):
                        self.index_map.append((subject, slice_idx))
                else:  # MPRAGE data
                    num_slices = f[subject]['slices'].shape[0]  # [num_slices, H, W]
                    for slice_idx in range(num_slices):
                        self.index_map.append((subject, slice_idx))
        
        shuffle(self.index_map)
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        subject, slice_idx = self.index_map[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            if 'contrasts' in f[subject].keys():  # qMRI data
                all_contrasts = f[subject]['contrasts'][:, slice_idx]  # [num_contrasts, H, W]
                
                if self.same_contrast:
                    contrast_idx = sample(range(len(all_contrasts)), 1)[0]
                    images = {f"image{i+1}": all_contrasts[contrast_idx] for i in range(self.num_views)}
                else:
                    contrast_indices = sample(range(len(all_contrasts)), self.num_views)
                    images = {f"image{i+1}": all_contrasts[contrast_idx] for i, contrast_idx in enumerate(contrast_indices)}
            else:  # MPRAGE data
                slice_data = f[subject]['slices'][slice_idx]  # [H, W]
                images = {f"image{i+1}": slice_data for i in range(self.num_views)}
        
        # Convert to tensor and add channel dimension, keeping data in uint8
        images = {k: torch.from_numpy(v).unsqueeze(0) for k, v in images.items()}
        
        # Apply transforms
        if self.transform:
            images = {k: self.transform(v) for k, v in images.items()}
        
        return images

def add_gaussian_noise(x, mean=0.0, std_range=(0.001, 0.2)):
    std = torch.empty(1).uniform_(std_range[0], std_range[1]).item()
    noise = torch.randn_like(x) * std + mean
    return x + noise

def get_transforms():
    """
    Returns a composition of augmentations for MRI slices using torchvision v2 API:
    1. RandomResizedCrop: Maintains local structure while providing different views
    2. RandomHorizontalFlip & RandomVerticalFlip: Valid due to bilateral symmetry
    3. RandomRotation: Full rotation for axial slices
    4. RandomAffine: Small geometric variations (rotation, translation, scale)
    5. ToDtype: Convert to float32 for intensity transforms
    6. GaussianBlur: Simulates resolution variations
    7. RandomAdjustSharpness: Simulates focus variations
    8. First Normalize: Standardizes to mean 0, std 1
    9. Gaussian Noise: Simulates scanner noise
    10. Final Normalize: Scales to [-1, 1] range
    """
    return v2.Compose([
        # Geometric transformations
        v2.RandomResizedCrop(
            size=224,
            scale=(0.8, 1.0),  # Less aggressive scale variation for medical images
            ratio=(0.9, 1.1),  # Keep aspect ratio close to original
            antialias=True
        ),
        # Rotations and flips
        v2.RandomHorizontalFlip(p=0.5),  # Valid due to bilateral symmetry
        v2.RandomVerticalFlip(p=0.5),    # Valid for axial slices
        v2.RandomRotation(
            degrees=180,  # Full rotation for axial slices
        ),
        # Small affine transforms
        v2.RandomAffine(
            degrees=15,  # Moderate rotation
            translate=(0.1, 0.1),  # Small translations
            scale=(0.9, 1.1),  # Moderate scaling
            fill=0,
            interpolation=v2.InterpolationMode.BILINEAR
        ),
        
        # Convert to float32 before intensity transforms
        v2.ToDtype(torch.float32, scale=True),
        
        # Intensity transformations
        v2.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 1.0)  # Moderate blur range
        ),
        v2.RandomAdjustSharpness(
            sharpness_factor=1.5,
            p=0.5
        ),
        # Initial normalization to standard normal
        v2.Normalize(
            mean=[0.0],
            std=[1.0]
        ),
        # Add random Gaussian noise
        v2.Lambda(lambda x: add_gaussian_noise(x, mean=0.0, std_range=(0.001, 0.2))),
        
        # Final normalization to range centered around 0.5
        v2.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])

def get_bloch_loader(
    batch_size=1,
    same_contrast=False,
    num_views=2,
    num_workers=4,
    pin_memory=True,
    pin_memory_device=None,
    persistent_workers=True,
    prefetch_factor=2,
):
    if num_views < 2:
        raise ValueError("num_views must be at least 2")

    transform = get_transforms()
    dataset = H5SliceDataset(
        "output/qmri_data.h5",
        transform=transform,
        same_contrast=same_contrast,
        num_views=num_views
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

def get_mprage_loader(
    batch_size=1,
    num_views=2,
    num_workers=4,
    pin_memory=True,
    pin_memory_device=None,
    persistent_workers=True,
    prefetch_factor=2,
):
    if num_views < 2:
        raise ValueError("num_views must be at least 2")

    transform = get_transforms()
    dataset = H5SliceDataset(
        "output/mprage_data.h5",
        transform=transform,
        num_views=num_views
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
