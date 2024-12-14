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

class RandGaussianNoise(Transform):
    def __init__(self, sigma_range=(0.001, 0.2), mean=0.0, clip=True):
        super().__init__()
        self.sigma_range = sigma_range
        self.mean = mean
        self.clip = clip

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sigma = torch.empty(1).uniform_(self.sigma_range[0], self.sigma_range[1]).item()
        params = dict(sigma=sigma)
        return params

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.gaussian_noise, inpt, mean=self.mean, sigma=params["sigma"], clip=self.clip)
        

class GaussianNoise(Transform):
    """Add gaussian noise to images or videos.

    The input tensor is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    Each image or frame in a batch will be transformed independently i.e. the
    noise added to each image will be different.

    The input tensor is also expected to be of float dtype in ``[0, 1]``.
    This transform does not support PIL images.

    Args:
        mean (float): Mean of the sampled normal distribution. Default is 0.
        sigma (float): Standard deviation of the sampled normal distribution. Default is 0.1.
        clip (bool, optional): Whether to clip the values in ``[0, 1]`` after adding noise. Default is True.
    """

    def __init__(self, mean: float = 0.0, sigma: float = 0.1, clip=True) -> None:
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.clip = clip

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.gaussian_noise, inpt, mean=self.mean, sigma=self.sigma, clip=self.clip)


def get_transforms():
    """
    Returns a composition of augmentations for MRI slices using v2 API:
    1. RandomResizedCrop: Maintains local structure while providing different views
    2. RandomRotation90: Valid anatomical orientations for axial slices
    3. RandomFlip: Valid due to approximate bilateral symmetry
    4. RandomAffine: Provides small rotation and scaling variations
    5. GaussianBlur: Simulates resolution variations
    6. RandomAdjustSharpness: Simulates focus variations
    7. GaussianNoise: Simulates scanner noise
    8. Normalize: Standardizes the input
    """
    return v2.Compose([
        # Geometric transformations
        v2.RandomResizedCrop(
            size=224,
            scale=(0.8, 1.0),  # Less aggressive scale variation for medical images
            ratio=(0.9, 1.1),  # Keep aspect ratio close to original
            antialias=True
        ),
        # 90-degree rotations and flips
        v2.RandomHorizontalFlip(p=0.5),  # Valid due to bilateral symmetry
        v2.RandomVerticalFlip(p=0.5),    # Valid for axial slices
        v2.RandomRotation(
            degrees=180,  # Only 90-degree rotations
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
        # Normalization
        v2.Normalize(
            mean=[0.0],
            std=[1.0]
        ),
        # RandGaussianNoise(
        #     sigma_range=(0.001, 0.2),
        #     mean=0.0,
        #     clip=False,
        #     p=1.0
        # ),
        GaussianNoise(
            mean=0.0,
            sigma=0.1,
            clip=False
        ),
        
        # Normalization
        v2.Normalize(
            mean=[0.5],
            std=[0.5]
        )
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
