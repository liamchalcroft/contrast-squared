import monai as mn
import glob
import os
import h5py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional, List

def generate_ixi_dataset(
    input_dir: str, 
    output_path: str,
    metadata_path: str,
    slice_range: Tuple[int, int] = (100, 200),
    image_size: Tuple[int, int] = (224, 224),
    modalities: Optional[List[str]] = None
):
    """Generate H5 dataset from IXI data.
    
    Args:
        input_dir: Directory containing IXI dataset
        output_path: Path to output H5 file
        metadata_path: Path to IXI metadata CSV
        slice_range: Range of slices to extract (min, max)
        image_size: Size to resize images to (height, width)
        modalities: List of modalities to include. If None, uses ['t1', 't2', 'pd']
    """
    if modalities is None:
        modalities = ['t1', 't2', 'pd']

    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    
    # Site mapping
    SITE_NAMES = {
        'guys': 'GST',
        'hh': 'HH',
        'iop': 'IOP'
    }
    
    # Set up transforms
    transform = mn.transforms.Compose([
        mn.transforms.LoadImageD(keys=modalities + [f"label{i}" for i in range(1,4)]),
        mn.transforms.EnsureChannelFirstD(keys=modalities + [f"label{i}" for i in range(1,4)]),
        mn.transforms.OrientationD(keys=modalities + [f"label{i}" for i in range(1,4)], axcodes="RAS"),
        mn.transforms.SpacingD(keys=modalities + [f"label{i}" for i in range(1,4)], pixdim=(1.0, 1.0, 1.0)),
        mn.transforms.ResizeD(keys=modalities + [f"label{i}" for i in range(1,4)], spatial_size=image_size),
        mn.transforms.NormalizeIntensityD(keys=modalities),
    ])

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Create groups for different tasks
        denoising = f.create_group("denoising")
        segmentation = f.create_group("segmentation")
        classification = f.create_group("classification")
        
        # Store dataset metadata
        f.attrs['slice_range'] = slice_range
        f.attrs['image_size'] = image_size
        f.attrs['modalities'] = [m.encode('utf-8') for m in modalities]
        
        # Process each site
        for site in ['guys', 'hh', 'iop']:
            site_path = os.path.join(input_dir, site)
            if not os.path.exists(site_path):
                continue
                
            # Process each subject in this site
            subject_paths = sorted(glob.glob(os.path.join(site_path, "IXI*")))
            for subject_path in tqdm(subject_paths, desc=f"Processing {site} subjects"):
                subject_id = os.path.basename(subject_path)
                
                try:
                    # Get subject metadata
                    ixi_id = int(subject_id.split("-")[1])
                    subject_meta = metadata_df[metadata_df['IXI_ID'] == ixi_id]
                    
                    if subject_meta.empty:
                        print(f"No metadata found for subject {subject_id}")
                        continue
                    
                    # Prepare data dictionary for transform
                    data_dict = {
                        mod: os.path.join(subject_path, f"{mod.upper()}.nii.gz")
                        for mod in modalities
                    }
                    
                    # Add segmentation labels (3 classes + background)
                    for i in range(1, 4):
                        data_dict[f"label{i}"] = os.path.join(subject_path, f"c{i}p_IXI{subject_id}-T1.nii.gz")
                    
                    # Apply transforms
                    data = transform(data_dict)
                    
                    # Extract relevant slices and combine labels
                    slices_dict = {
                        mod: data[mod][0, :, :, slice_range[0]:slice_range[1]]
                        for mod in modalities
                    }
                    
                    # Combine segmentation labels with background
                    labels = []
                    for i in range(1, 4):
                        label = data[f"label{i}"][0, :, :, slice_range[0]:slice_range[1]]
                        labels.append(label)
                    
                    # Create background as 1 - sum of other labels
                    background = 1 - sum(labels)
                    labels.insert(0, background)  # Add background as first channel
                    
                    # Stack labels into single tensor
                    combined_labels = np.stack(labels, axis=0)
                    slices_dict["label"] = combined_labels
                    
                    # Store in H5 groups with site information
                    for group_name, group in [
                        ("denoising", denoising),
                        ("segmentation", segmentation),
                        ("classification", classification)
                    ]:
                        subj_group = group.create_group(subject_id)
                        
                        if group_name == "denoising":
                            for mod in modalities:
                                subj_group.create_dataset(mod, data=slices_dict[mod])
                                # Add modality and site metadata
                                subj_group[mod].attrs['modality'] = mod
                                subj_group[mod].attrs['site'] = SITE_NAMES[site]
                        
                        elif group_name == "segmentation":
                            subj_group.create_dataset("image", data=slices_dict["t1"])
                            subj_group.create_dataset("mask", data=slices_dict["mask"])
                            # Add modality and site metadata
                            subj_group['image'].attrs['modality'] = 't1'
                            subj_group['image'].attrs['site'] = SITE_NAMES[site]
                        
                        elif group_name == "classification":
                            for mod in modalities:
                                subj_group.create_dataset(f"image_{mod}", data=slices_dict[mod])
                                # Add modality and site metadata
                                subj_group[f"image_{mod}"].attrs['modality'] = mod
                                subj_group[f"image_{mod}"].attrs['site'] = SITE_NAMES[site]
                                # Add demographic data
                                subj_group[f"image_{mod}"].attrs['age'] = float(subject_meta['AGE'].iloc[0])
                                subj_group[f"image_{mod}"].attrs['sex'] = subject_meta['SEX'].iloc[0]
                    
                except Exception as e:
                    print(f"Error processing subject {subject_id}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate H5 dataset from IXI data')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing IXI dataset')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to output H5 file')
    parser.add_argument('--metadata_path', type=str, required=True,
                      help='Path to IXI metadata CSV')
    parser.add_argument('--slice_range', type=int, nargs=2, default=[100, 200],
                      help='Range of slices to extract (min max)')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                      help='Size to resize images to (height width)')
    parser.add_argument('--modalities', type=str, nargs='+',
                      default=['t1', 't2', 'pd'],
                      help='Modalities to include')
    
    args = parser.parse_args()
    
    generate_ixi_dataset(
        input_dir=args.input_dir,
        output_path=args.output_path,
        metadata_path=args.metadata_path,
        slice_range=tuple(args.slice_range),
        image_size=tuple(args.image_size),
        modalities=args.modalities
    )