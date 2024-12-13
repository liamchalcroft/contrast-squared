import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import glob
import os
from random import shuffle, seed, sample
from PIL import Image
import numpy as np

seed(786)

class SliceDataset(Dataset):
    def __init__(self, file_dict, transform=None):
        self.file_dict = file_dict
        self.transform = transform

    def __len__(self):
        return len(self.file_dict)

    def __getitem__(self, idx):
        item = self.file_dict[idx]
        images = {}
        
        for key, filepath in item.items():
            # Images are already grayscale, just load directly
            image = Image.open(filepath)
            if self.transform:
                image = self.transform(image)
            images[key] = image
            
        return images

def get_transforms():
    return T.Compose([
        T.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            fill=0
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(15),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

def get_bloch_loader(
    batch_size=1,
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
        # Parse filename to get subject and slice number
        parts = os.path.basename(f).split('_')
        base_name = f"{parts[0]}_slice{parts[1]}"  # subject_id + slice number
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

    transform = get_transforms()
    dataset = SliceDataset(train_dict, transform=transform)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, transform

def get_mprage_loader(
    batch_size=1,
    num_views=2,
):
    if num_views < 2:
        raise ValueError("num_views must be at least 2")

    # Get all slice files
    slice_files = glob.glob(os.path.join("output/mprage_slices/*.png"))
    
    # Group files by slice
    slice_groups = {}
    for f in slice_files:
        slice_groups[f] = f

    # Create training dictionary - use the same slice for all views
    train_dict = [{f"image{i+1}": f for i in range(num_views)} for f in slice_groups.keys()]
    shuffle(train_dict)

    transform = get_transforms()
    dataset = SliceDataset(train_dict, transform=transform)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, transform
