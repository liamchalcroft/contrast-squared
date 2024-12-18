import torch
import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F, Transform
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from random import shuffle, seed, sample
import torch
from sklearn.model_selection import train_test_split

seed(786)

def get_data_chunks(h5_path, task='denoising', split='train', train_ratio=0.6, val_ratio=0.2, modality='t1', site='GST'):
    with h5py.File(h5_path, 'r') as f:
        task_group = f[task]
        subjects = list(task_group.keys())

        # Filter subjects by modality and site
        filtered_subjects = [
            subject for subject in subjects
            if task_group[subject].attrs.get('modality') == modality and
               task_group[subject].attrs.get('site') == site
        ]

        if train_ratio + val_ratio == 0:
            test_subjects = filtered_subjects
        else:
            # Split subjects into train, val, and test
            train_subjects, temp_subjects = train_test_split(filtered_subjects, train_size=train_ratio)
            val_subjects, test_subjects = train_test_split(temp_subjects, test_size=(1 - train_ratio - val_ratio) / (val_ratio + (1 - train_ratio - val_ratio)))

        if split == 'train':
            selected_subjects = train_subjects
        elif split == 'val':
            selected_subjects = val_subjects
        elif split == 'test':
            selected_subjects = test_subjects
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load data into memory
        data_chunks = {}
        for subject in selected_subjects:
            data_chunks[subject] = {
                'image': task_group[subject]['image'][:]
            }
            if task == 'segmentation':
                data_chunks[subject]['label'] = task_group[subject]['label'][:]
            # Add metadata if available
            if task == 'classification':
                data_chunks[subject]['age'] = task_group[subject].attrs.get('age', None)
                data_chunks[subject]['sex'] = task_group[subject].attrs.get('sex', None)

    return data_chunks

class H5SliceDataset(Dataset):
    def __init__(self, data_chunks, task='denoising', transform=None):
        self.transform = transform
        self.task = task
        self.data_chunks = data_chunks
        
        # Create patient-wise index mapping
        self.patient_slices = {}
        for subject in self.data_chunks.keys():
            num_slices = self.data_chunks[subject]['image'].shape[0]
            self.patient_slices[subject] = list(range(num_slices))
        
        # Create index mapping that groups by patient
        self.index_map = []
        for subject, slices in self.patient_slices.items():
            for slice_idx in slices:
                self.index_map.append((subject, slice_idx))
        
        # Sort by patient ID to ensure grouping
        self.index_map.sort(key=lambda x: x[0])

    def __getitem__(self, idx):
        subject, slice_idx = self.index_map[idx]
        data = self.data_chunks[subject]
        
        image = data['image'][slice_idx]
        if self.task == 'segmentation':
            label = data['label'][slice_idx]
        
        # Convert to tensor
        image = torch.from_numpy(image.copy())
        if self.task == 'segmentation':
            label = torch.from_numpy(label.copy())
        
        # Apply transforms
        if self.image_label_transform:
            if self.task == 'segmentation':
                # Apply the same transform to both image and label
                image, label = self.image_label_transform((image, label))
            else:
                image = self.image_transform(image)
        if self.image_transform:
            image = self.image_transform(image)
        
        # Retrieve additional metadata
        if self.task == 'classification':
            age = data.get('age', None)
            sex = data.get('sex', None)
        
        if self.task == 'segmentation':
            return {'image': image, 'label': label, 'patient_id': subject}
        elif self.task == 'classification':
            return {'image': image, 'patient_id': subject, 'age': age, 'sex': sex}
        else:
            return {'image': image, 'patient_id': subject}

    def __len__(self):
        return len(self.index_map)

class PatientBatchSampler:
    """Ensures each batch contains only one slice per patient"""
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.patient_indices = {}
        
        # Group indices by patient
        for idx, (subject, _) in enumerate(dataset.index_map):
            if subject not in self.patient_indices:
                self.patient_indices[subject] = []
            self.patient_indices[subject].append(idx)
        
        # Calculate exact number of valid batches
        all_indices = [(idx, subject) for subject, indices in self.patient_indices.items() 
                      for idx in indices]
        self.num_samples = len(all_indices)
        
        # Calculate full batches
        self.num_full_batches = self.num_samples // batch_size
        
        # Calculate if there's a valid partial batch (more than 1 item)
        remaining_items = self.num_samples % batch_size
        self.has_partial_batch = remaining_items > 1
        
        # Total number of valid batches
        self.batches_per_epoch = self.num_full_batches + (1 if self.has_partial_batch else 0)
    
    def __iter__(self):
        # Create a list of all indices and their corresponding patients
        all_indices = [(idx, subject) for subject, indices in self.patient_indices.items() 
                      for idx in indices]
        
        # Shuffle all indices
        shuffle(all_indices)
        
        # Create batches ensuring no patient appears twice in same batch
        current_batch = []
        current_patients = set()
        
        for idx, patient in all_indices:
            if patient not in current_patients:
                current_batch.append(idx)
                current_patients.add(patient)
                
                if len(current_batch) == self.batch_size:
                    yield current_batch
                    current_batch = []
                    current_patients.clear()
            
        # Only yield the last batch if it has more than one item
        if len(current_batch) > 1:
            yield current_batch
    
    def __len__(self):
        return self.batches_per_epoch

def get_train_val_transforms():
    image_label_transform = v2.Compose([
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
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            fill=0,
            interpolation=v2.InterpolationMode.BILINEAR
        ),
        
    ])

    image_transform = v2.Compose([
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
    ])

    return image_label_transform, image_transform

def get_test_transforms():
    image_label_transform = None
    image_transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.0],
            std=[1.0]
        ),
    ])
    return image_label_transform, image_transform

def get_train_val_loaders(
        batch_size=1, 
        task='denoising',
        modality='t1',
        site='GST',
        num_workers=4, 
        pin_memory=True, 
        pin_memory_device=None, 
        persistent_workers=True, 
        prefetch_factor=3
    ):
    h5_path = 'task_data/ixi.h5'
    train_ratio = 0.6
    val_ratio = 0.2
    
    image_label_transform, image_transform = get_train_val_transforms()
    train_data_chunks = get_data_chunks(h5_path, task=task, split='train', modality=modality, site=site, train_ratio=train_ratio, val_ratio=val_ratio)
    val_data_chunks = get_data_chunks(h5_path, task=task, split='val', modality=modality, site=site, train_ratio=train_ratio, val_ratio=val_ratio)

    train_dataset = H5SliceDataset(
        train_data_chunks,
        image_label_transform=image_label_transform,
        image_transform=image_transform,
        modality=modality,
    )
    val_dataset = H5SliceDataset(
        val_data_chunks,
        image_label_transform=image_label_transform,
        image_transform=image_transform,
        modality=modality,
    )

    # Use custom sampler instead of shuffle
    train_sampler = PatientBatchSampler(train_dataset, batch_size)
    val_sampler = PatientBatchSampler(val_dataset, batch_size)

    # Only include pin_memory_device if it's specified
    train_dataloader_kwargs = {
        'batch_sampler': train_sampler,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
        'prefetch_factor': prefetch_factor,
    }
    val_dataloader_kwargs = {
        'batch_sampler': val_sampler,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
        'prefetch_factor': prefetch_factor,
    }
    
    if pin_memory_device is not None:
        train_dataloader_kwargs['pin_memory_device'] = pin_memory_device
        val_dataloader_kwargs['pin_memory_device'] = pin_memory_device

    train_loader = DataLoader(
        train_dataset,
        **train_dataloader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        **val_dataloader_kwargs
    )
    return train_loader, val_loader

def get_test_loader(
        batch_size=1, 
        task='denoising',
        modality='t1',
        site='GST',
        num_workers=4, 
        pin_memory=True, 
        pin_memory_device=None, 
        persistent_workers=True, 
        prefetch_factor=3
    ):
    h5_path = 'task_data/ixi.h5'
    if site == 'GST':
        train_ratio = 0.6
        val_ratio = 0.2
    else:
        train_ratio = 0.
        val_ratio = 0.

    image_label_transform, image_transform = get_test_transforms()
    test_data_chunks = get_data_chunks(h5_path, task=task, split='test', modality=modality, site=site, train_ratio=train_ratio, val_ratio=val_ratio)

    test_dataset = H5SliceDataset(
        test_data_chunks,
        image_label_transform=image_label_transform,
        image_transform=image_transform,
        modality=modality,
    )

    test_dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
        'prefetch_factor': prefetch_factor,
    }
    
    if pin_memory_device is not None:
        test_dataloader_kwargs['pin_memory_device'] = pin_memory_device
    
    test_loader = DataLoader(
        test_dataset,
        **test_dataloader_kwargs
    )
    return test_loader