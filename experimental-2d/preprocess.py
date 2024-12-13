import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import h5py
from random import shuffle, seed, sample
import numpy as np

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
        
        # Convert to tensor and add channel dimension
        images = {k: torch.from_numpy(v).float().unsqueeze(0) for k, v in images.items()}
        
        # Apply transforms
        if self.transform:
            images = {k: self.transform(v) for k, v in images.items()}
        
        return images

def get_transforms():
    return T.v2.Compose([
        T.v2.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            fill=0
        ),
        T.v2.RandomHorizontalFlip(p=0.5),
        T.v2.RandomVerticalFlip(p=0.5),
        T.v2.RandomRotation(15),
        T.v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        T.v2.Normalize(mean=[0.5], std=[0.5]),
        T.v2.GaussianNoise(mean=0, sigma=0.1),
    ])

def get_bloch_loader(
    batch_size=1,
    same_contrast=False,
    num_views=2,
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
        num_workers=4,
        pin_memory=True,
    )

def get_mprage_loader(
    batch_size=1,
    num_views=2,
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
        num_workers=4,
        pin_memory=True,
    )
