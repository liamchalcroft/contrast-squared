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
            volume = data["image"]  # Shape: [4, D, H, W]
            
            # Create subject group
            subj_group = f.create_group(subject_id)
            
            # Pre-allocate space for all slices and contrasts
            num_slices = slice_range[1] - slice_range[0]
            contrasts_dataset = subj_group.create_dataset(
                "contrasts", 
                shape=(num_contrasts, num_slices, 224, 224),
                dtype=np.float32
            )
            
            # Generate random contrasts first (more efficient)
            for i in tqdm(range(num_contrasts), leave=False):
                # Generate random contrast for the whole volume
                bloch_transform = MONAIBlochTransformD(keys=["image"], num_ch=1)
                clip_transform = ClipPercentilesD(keys=["image"], lower=0.5, upper=99.5)
                contrast_volume = clip_transform(bloch_transform({"image": volume}))["image"]
                
                # Extract and store all slices for this contrast
                for slice_idx in range(slice_range[0], slice_range[1]):
                    slice_pos = slice_idx - slice_range[0]
                    contrasts_dataset[i, slice_pos] = contrast_volume[0, :, :, slice_idx].numpy()

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
    
    with h5py.File(output_path, 'w') as f:
        for file_path in tqdm(input_files):
            subject_id = os.path.basename(os.path.dirname(file_path))
            data_dict = {"image": file_path}
            
            # Load and transform the data
            data = transform(data_dict)
            volume = data["image"]  # Shape: [1, D, H, W]
            
            # Create subject group and store slices
            subj_group = f.create_group(subject_id)
            slices = volume[0, :, :, slice_range[0]:slice_range[1]].numpy()
            subj_group.create_dataset("slices", data=slices)

if __name__ == "__main__":
    qmri_files = glob.glob(os.path.join("/home/lchalcroft/MPM_DATA/*/*/masked_pd.nii"))
    mprage_files = glob.glob(os.path.join("/home/lchalcroft/MPM_DATA/*/*/sim_mprage.nii"))
    
    generate_qmri_slices(qmri_files, "data/qmri_data.h5", num_contrasts=100, slice_range=(50, 150))
    generate_mprage_slices(mprage_files, "data/mprage_data.h5", slice_range=(50, 150))
