import monai as mn
import glob
import os
from torch.utils.data import DataLoader
import torch
from random import shuffle, seed, randint
import numpy as np
from PIL import Image
import tqdm
from bloch import MONAIBlochTransformD
from utils import ClipPercentilesD

seed(786)

def rescale_mpm(mpm):
    pd = mpm[0]
    r1 = mpm[1]
    r2s = mpm[2]
    mt = mpm[3]
    # r1 = r1 * 10
    # r2s = r2s * 10
    return torch.stack([pd, r1, r2s, mt], dim=0)

def save_slice_as_png(slice_data, output_path):
    # Normalize to 0-255 range
    slice_data = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
    img = Image.fromarray(slice_data)
    img.save(output_path)

def generate_qmri_slices(input_files, output_dir, num_contrasts=100, slice_range=(50, 150)):
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    for file_path in tqdm.tqdm(input_files):
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
        
        # Generate random contrasts
        for i in range(num_contrasts):
            # For each slice in the specified range
            for slice_idx in range(slice_range[0], slice_range[1]):
                slice_data = volume[:, slice_idx, :, :]
                
                # Generate random contrast
                bloch_transform = MONAIBlochTransformD(keys=["image"], num_ch=1)
                clip_transform = ClipPercentilesD(keys=["image"], lower=0.5, upper=99.5)
                contrast = clip_transform(bloch_transform({"image": slice_data}))["image"]
                
                # Save the slice
                output_path = os.path.join(
                    output_dir,
                    f"{subject_id}_slice{slice_idx:03d}_contrast{i:03d}.png"
                )
                save_slice_as_png(contrast[0].numpy(), output_path)

def generate_mprage_slices(input_files, output_dir, slice_range=(50, 150)):
    os.makedirs(output_dir, exist_ok=True)
    
    prepare_mprage = [
        mn.transforms.LoadImageD(keys=["image"], image_only=True),
        mn.transforms.EnsureChannelFirstD(keys=["image"]),
        mn.transforms.OrientationD(keys=["image"], axcodes="RAS"),
        ClipPercentilesD(keys=["image"], lower=0.5, upper=99.5),
        mn.transforms.ResizeWithPadOrCropD(keys=["image"], spatial_size=(224, 224, -1)),
    ]
    
    transform = mn.transforms.Compose(prepare_mprage)
    
    for file_path in tqdm.tqdm(input_files):
        subject_id = os.path.basename(os.path.dirname(file_path))
        data_dict = {"image": file_path}
        
        # Load and transform the data
        data = transform(data_dict)
        volume = data["image"]  # Shape: [1, D, H, W]
        
        # Save each slice in the specified range
        for slice_idx in range(slice_range[0], slice_range[1]):
            slice_data = volume[:, slice_idx, :, :]
            
            output_path = os.path.join(
                output_dir,
                f"{subject_id}_slice{slice_idx:03d}.png"
            )
            save_slice_as_png(slice_data[0], output_path)

# Add this at the bottom of the file to run the generation
if __name__ == "__main__":
    qmri_files = glob.glob(os.path.join("/home/lchalcroft/MPM_DATA/*/*/masked_pd.nii"))
    mprage_files = glob.glob(os.path.join("/home/lchalcroft/MPM_DATA/*/*/sim_mprage.nii"))
    
    generate_qmri_slices(qmri_files, "output/qmri_slices", num_contrasts=100, slice_range=(50, 150))
    generate_mprage_slices(mprage_files, "output/mprage_slices", slice_range=(50, 150))
