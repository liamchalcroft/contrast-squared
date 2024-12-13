import monai as mn
import glob
import os
import h5py
import torch
from random import seed
import numpy as np
from tqdm import tqdm
from bloch import MONAIBlochTransformD
from utils import ClipPercentilesD

seed(786)

def rescale_mpm(mpm):
    pd = mpm[0]
    r1 = mpm[1]
    r2s = mpm[2]
    mt = mpm[3]
    return torch.stack([pd, r1, r2s, mt], dim=0)

def rescale_to_uint8(data):
    """Rescale data to [0, 255] range and convert to uint8"""
    data_min = data.min()
    data_max = data.max()
    if data_max > data_min:
        data = 255 * (data - data_min) / (data_max - data_min)
    return data.astype(np.uint8)

def is_valid_slice(slice_data, min_percentage=0.05):
    """
    Check if slice contains enough non-zero pixels
    min_percentage: minimum percentage of non-zero pixels required
    """
    total_pixels = slice_data.size
    non_zero_pixels = np.count_nonzero(slice_data)
    return (non_zero_pixels / total_pixels) > min_percentage

def generate_qmri_slices(input_files, output_path, num_contrasts=100, slice_range=(100, 200)):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    prepare_mpm = [
        mn.transforms.LoadImageD(keys=["image"], image_only=True),
        mn.transforms.EnsureChannelFirstD(keys=["image"]),
        mn.transforms.OrientationD(keys=["image"], axcodes="RAS"),
        mn.transforms.LambdaD(keys=["image"], func=mn.transforms.SignalFillEmpty()),
        mn.transforms.LambdaD(keys=["image"], func=rescale_mpm),
        ClipPercentilesD(keys=["image"], lower=0.5, upper=99.5),
        mn.transforms.ResizeWithPadOrCropD(keys=["image"], spatial_size=(224, 224, -1)),
    ]
    
    transform = mn.transforms.Compose(prepare_mpm)
    
    # Configure HDF5 for compression
    compression_opts = {
        'compression': 'gzip',
        'compression_opts': 4,
        'dtype': np.uint8
    }
    
    with h5py.File(output_path, 'w') as f:
        for file_path in tqdm(input_files):
            subject_id = os.path.basename(os.path.dirname(file_path))
            data_dict = {
                "image": [
                    file_path,
                    file_path.replace("masked_pd.nii", "masked_r1.nii"),
                    file_path.replace("masked_pd.nii", "masked_r2s.nii"),
                    file_path.replace("masked_pd.nii", "masked_mt.nii"),
                ]
            }
            
            # Load and transform the data
            data = transform(data_dict)
            volume = data["image"]  # Shape: [4, H, W, D]
            
            # Find valid slices within the range
            valid_slices = []
            for slice_idx in range(slice_range[0], slice_range[1]):
                slice_data = volume[0, :, :, slice_idx].numpy()  # Check PD contrast
                if is_valid_slice(slice_data):
                    valid_slices.append(slice_idx)
            
            if not valid_slices:
                print(f"Warning: No valid slices found for subject {subject_id}")
                continue
                
            print(f"Found {len(valid_slices)} valid slices for subject {subject_id}")
            
            # Create subject group
            subj_group = f.create_group(subject_id)
            
            # Pre-allocate space for all valid slices and contrasts
            contrasts_dataset = subj_group.create_dataset(
                "contrasts", 
                shape=(num_contrasts, len(valid_slices), 224, 224),
                chunks=(1, len(valid_slices), 224, 224),
                **compression_opts
            )
            
            # Generate random contrasts
            for i in tqdm(range(num_contrasts), leave=False):
                bloch_transform = MONAIBlochTransformD(keys=["image"], num_ch=1)
                clip_transform = ClipPercentilesD(keys=["image"], lower=0.5, upper=99.5)
                contrast_volume = clip_transform(bloch_transform({"image": volume}))["image"]
                
                # Extract and store valid slices for this contrast
                for slice_pos, slice_idx in enumerate(valid_slices):
                    slice_data = contrast_volume[0, :, :, slice_idx].numpy()
                    contrasts_dataset[i, slice_pos] = rescale_to_uint8(slice_data)

def generate_mprage_slices(input_files, output_path, slice_range=(100, 200)):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    prepare_mprage = [
        mn.transforms.LoadImageD(keys=["image"], image_only=True),
        mn.transforms.EnsureChannelFirstD(keys=["image"]),
        mn.transforms.OrientationD(keys=["image"], axcodes="RAS"),
        ClipPercentilesD(keys=["image"], lower=0.5, upper=99.5),
        mn.transforms.ResizeWithPadOrCropD(keys=["image"], spatial_size=(224, 224, -1)),
    ]
    
    transform = mn.transforms.Compose(prepare_mprage)
    
    # Configure HDF5 for compression
    compression_opts = {
        'compression': 'gzip',
        'compression_opts': 4,
        'dtype': np.uint8
    }
    
    with h5py.File(output_path, 'w') as f:
        for file_path in tqdm(input_files):
            subject_id = os.path.basename(os.path.dirname(file_path))
            data_dict = {"image": file_path}
            
            # Load and transform the data
            data = transform(data_dict)
            volume = data["image"]  # Shape: [1, H, W, D]
            
            # Find valid slices within the range
            valid_slices = []
            for slice_idx in range(slice_range[0], slice_range[1]):
                slice_data = volume[0, :, :, slice_idx].numpy()
                if is_valid_slice(slice_data):
                    valid_slices.append(slice_idx)
            
            if not valid_slices:
                print(f"Warning: No valid slices found for subject {subject_id}")
                continue
                
            print(f"Found {len(valid_slices)} valid slices for subject {subject_id}")
            
            # Create subject group and store valid slices
            subj_group = f.create_group(subject_id)
            slices = np.stack([volume[0, :, :, idx].numpy() for idx in valid_slices])
            subj_group.create_dataset(
                "slices", 
                data=rescale_to_uint8(slices),
                chunks=(1, 224, 224),
                **compression_opts
            )

if __name__ == "__main__":
    qmri_files = glob.glob(os.path.join("/home/lchalcroft/MPM_DATA/*/*/masked_pd.nii"))
    mprage_files = glob.glob(os.path.join("/home/lchalcroft/MPM_DATA/*/*/sim_mprage.nii"))
    
    generate_mprage_slices(mprage_files, "output/mprage_data.h5", slice_range=(100, 200))
    generate_qmri_slices(qmri_files, "output/qmri_data.h5", num_contrasts=100, slice_range=(100, 200))
