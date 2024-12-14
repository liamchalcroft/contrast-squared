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

def rescale_to_uint8(data, dtype=np.uint8):
    """Rescale data to [0, 255] range and convert to uint8"""
    data_min = data.min()
    data_max = data.max()
    if data_max > data_min:
        data = 255 * (data - data_min) / (data_max - data_min)
    return data.astype(dtype)

def generate_qmri_slices(input_files, output_path, num_contrasts=100, slice_range=(50, 150)):
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
            
            # Create subject group
            subj_group = f.create_group(subject_id)
            
            # Pre-allocate space for all slices and contrasts
            num_slices = slice_range[1] - slice_range[0]
            contrasts_dataset = subj_group.create_dataset(
                "contrasts", 
                shape=(num_contrasts, num_slices, 224, 224),
                chunks=(1, num_slices, 224, 224),
                **compression_opts
            )
            
            # Generate random contrasts first (more efficient)
            for i in tqdm(range(num_contrasts), leave=False):
                # Generate random contrast for the whole volume
                bloch_transform = MONAIBlochTransformD(keys=["image"], num_ch=1)
                clip_transform = ClipPercentilesD(keys=["image"], lower=0.5, upper=99.5)
                contrast_volume = clip_transform(bloch_transform({"image": volume}))["image"]
                # contrast_volume = bloch_transform({"image": volume})["image"]
                contrast_volume = rescale_to_uint8(contrast_volume.numpy())
                
                # Extract and store all slices for this contrast
                for slice_idx in range(slice_range[0], slice_range[1]):
                    slice_pos = slice_idx - slice_range[0]
                    slice_data = contrast_volume[0, :, :, slice_idx]
                    contrasts_dataset[i, slice_pos] = slice_data

def generate_mprage_slices(input_files, output_path, slice_range=(50, 150)):
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
            
            # Create subject group and store slices
            subj_group = f.create_group(subject_id)
            slices = volume[0, :, :, slice_range[0]:slice_range[1]].numpy()
            slices = np.moveaxis(slices, -1, 0)  # Move slices to first dimension
            data = rescale_to_uint8(slices)
            subj_group.create_dataset(
                "slices", 
                data=data,
                chunks=(1, 224, 224),
                **compression_opts
            )

if __name__ == "__main__":
    qmri_files = glob.glob(os.path.join("/home/lchalcroft/MPM_DATA/*/*/masked_pd.nii"))
    mprage_files = glob.glob(os.path.join("/home/lchalcroft/MPM_DATA/*/*/sim_mprage.nii"))
    
    generate_mprage_slices(mprage_files, "output/mprage_data.h5", slice_range=(100, 200))
    generate_qmri_slices(qmri_files, "output/qmri_data.h5", num_contrasts=100, slice_range=(100, 200))
