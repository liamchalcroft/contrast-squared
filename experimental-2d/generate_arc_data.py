import monai as mn
import glob
import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple

def generate_arc_dataset(
    input_dir: str, 
    output_path: str, 
    slice_range: Tuple[int, int] = (100, 200),
    image_size: Tuple[int, int] = (224, 224)
):
    """Generate H5 dataset from ARC data.
    
    Args:
        input_dir: Directory containing ARC dataset
        output_path: Path to output H5 file
        slice_range: Range of slices to extract (min, max)
        image_size: Size to resize images to (height, width)
    """
    transforms = [
        mn.transforms.LoadImageD(keys=["dwi", "mask"], image_only=True),
        mn.transforms.EnsureChannelFirstD(keys=["dwi", "mask"]),
        mn.transforms.OrientationD(keys=["dwi", "mask"], axcodes="RAS"),
        mn.transforms.ResizeWithPadOrCropD(
            keys=["dwi", "mask"], 
            spatial_size=(image_size[0], image_size[1], -1)
        ),
    ]
    
    transform = mn.transforms.Compose(transforms)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        stroke_group = f.create_group("stroke_segmentation")
        
        # Store dataset metadata
        f.attrs['slice_range'] = slice_range
        f.attrs['image_size'] = image_size
        
        # Process each subject
        subject_paths = sorted(glob.glob(os.path.join(input_dir, "sub-*")))
        for subject_path in tqdm(subject_paths, desc="Processing subjects"):
            subject_id = os.path.basename(subject_path)
            
            try:
                data = transform({
                    "dwi": os.path.join(subject_path, "dwi.nii.gz"),
                    "mask": os.path.join(subject_path, "stroke_mask.nii.gz")
                })
                
                # Extract relevant slices
                dwi_slices = data["dwi"][0, :, :, slice_range[0]:slice_range[1]]
                mask_slices = data["mask"][0, :, :, slice_range[0]:slice_range[1]]
                
                # Store in H5 file
                subj_group = stroke_group.create_group(subject_id)
                subj_group.create_dataset("image", data=dwi_slices)
                subj_group.create_dataset("mask", data=mask_slices)
                
                # Store metadata if available
                clinical_data_path = os.path.join(subject_path, "clinical_data.json")
                if os.path.exists(clinical_data_path):
                    import json
                    with open(clinical_data_path, 'r') as clin_file:
                        clinical_data = json.load(clin_file)
                        for key, value in clinical_data.items():
                            subj_group.attrs[key] = value
                
            except Exception as e:
                print(f"Error processing subject {subject_id}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate H5 dataset from ARC data')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing ARC dataset')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to output H5 file')
    parser.add_argument('--slice_range', type=int, nargs=2, default=[100, 200],
                      help='Range of slices to extract (min max)')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                      help='Size to resize images to (height width)')
    
    args = parser.parse_args()
    
    generate_arc_dataset(
        input_dir=args.input_dir,
        output_path=args.output_path,
        slice_range=tuple(args.slice_range),
        image_size=tuple(args.image_size)
    )
