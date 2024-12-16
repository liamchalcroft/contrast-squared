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
from utils import ClipPercentilesD

def rescale_to_uint8(data, dtype=np.uint8):
    """Rescale data to [0, 255] range and convert to uint8"""
    data_min = data.min()
    data_max = data.max()
    if data_max > data_min:
        data = 255 * (data - data_min) / (data_max - data_min)
    return data.astype(dtype)

def generate_ixi_dataset(
    input_dir: str, 
    output_path: str,
    metadata_path: str,
    slice_range: Tuple[int, int] = (100, 200),
    image_size: Tuple[int, int, int] = (224, 224, -1),
    modalities: Optional[List[str]] = None
):
    """Generate H5 dataset from IXI data.
    
    Args:
        input_dir: Directory containing IXI dataset (e.g. '/path/to/IXI/')
        output_path: Path to output H5 file
        metadata_path: Path to IXI metadata XLS file
        slice_range: Range of slices to extract (min, max)
        image_size: Size to resize images to (height, width)
        modalities: List of modalities to include. If None, uses ['t1', 't2', 'pd']
    """
    if modalities is None:
        modalities = ['t1', 't2', 'pd']

    # Load metadata from Excel file
    metadata_df = pd.read_excel(metadata_path)
    
    # Site mapping
    SITE_NAMES = {
        'guys': 'GST',
        'hh': 'HH',
        'iop': 'IOP'
    }
    
    # Configure HDF5 for compression
    compression_opts = {
        'compression': 'gzip',
        'compression_opts': 4,
        'dtype': np.uint8
    }

    with h5py.File(output_path, 'w') as f:
        # Create groups for different tasks
        denoising = f.create_group("denoising")
        segmentation = f.create_group("segmentation")
        classification = f.create_group("classification")
        
        # Store dataset metadata
        f.attrs['slice_range'] = slice_range
        f.attrs['image_size'] = image_size
        f.attrs.create('modalities', [m.encode('ascii') for m in modalities], dtype='S10')
        
        # Process each site
        for site in ['guys', 'hh', 'iop']:
            for modality in modalities:
                # Construct path to preprocessed data
                data_path = os.path.join(input_dir, site, modality, "preprocessed")
                if not os.path.exists(data_path):
                    continue
                
                # Get all preprocessed images
                image_paths = sorted(glob.glob(os.path.join(data_path, "p_*.nii.gz")))
                
                # Set up transforms for this modality
                transform_keys = [modality] + [f"label{i}" for i in range(1,4)]
                
                transform = mn.transforms.Compose([
                    mn.transforms.LoadImageD(keys=transform_keys),
                    mn.transforms.EnsureChannelFirstD(keys=transform_keys),
                    mn.transforms.OrientationD(keys=transform_keys, axcodes="RAS"),
                    mn.transforms.SpacingD(keys=transform_keys, pixdim=(1.0, 1.0, 1.0)),
                    mn.transforms.ResizeD(keys=transform_keys, spatial_size=image_size),
                    ClipPercentilesD(keys=[modality], lower=0.5, upper=99.5),
                ])
                
                for image_path in tqdm(image_paths, desc=f"Processing {site}/{modality}"):
                    # Extract subject ID from filename
                    subject_id = os.path.basename(image_path).split("-")[0][5:]
                    
                    # Get subject metadata
                    ixi_id = int(subject_id)
                    subject_meta = metadata_df[metadata_df['IXI_ID'] == ixi_id]
                    
                    if subject_meta.empty:
                        print(f"No metadata found for subject {subject_id}")
                        continue
                    
                    # Prepare data dictionary for transform
                    data_dict = {
                        modality: image_path,
                    }
                    
                    # Add segmentation labels
                    for i in range(1, 4):
                        label_path = image_path.replace("p_IXI", f"c{i}p_IXI")
                        data_dict[f"label{i}"] = label_path
                    
                    # Apply transforms
                    data = transform(data_dict)
                    
                    # Extract relevant slices and convert to uint8
                    slices = data[modality][0, :, :, slice_range[0]:slice_range[1]]
                    slices = rescale_to_uint8(slices.numpy())
                    
                    # Process segmentation labels
                    labels = []
                    for i in range(1, 4):
                        label = data[f"label{i}"][0, :, :, slice_range[0]:slice_range[1]]
                        labels.append(label)
                    
                    # Create background as 1 - sum of other labels
                    background = 1 - sum(labels)
                    labels.insert(0, background)
                    combined_labels = np.stack(labels, axis=0)
                    combined_labels = (combined_labels * 255).astype(np.uint8)
                    
                    # Store in appropriate H5 groups with compression
                    for group_name, group in [
                        ("denoising", denoising),
                        ("segmentation", segmentation),
                        ("classification", classification)
                    ]:
                        subj_group = group.create_group(f"IXI{subject_id}")
                        
                        if group_name == "denoising":
                            subj_group.create_dataset(modality, data=slices, **compression_opts)
                            subj_group[modality].attrs['modality'] = modality
                            subj_group[modality].attrs['site'] = SITE_NAMES[site]
                        
                        elif group_name == "segmentation":
                            subj_group.create_dataset(f"image_{modality}", data=slices, **compression_opts)
                            subj_group.create_dataset(f"label_{modality}", data=combined_labels, **compression_opts)
                            subj_group[f"image_{modality}"].attrs['modality'] = modality
                            subj_group[f"image_{modality}"].attrs['site'] = SITE_NAMES[site]
                        
                        elif group_name == "classification":
                            subj_group.create_dataset(f"image_{modality}", data=slices, **compression_opts)
                            subj_group[f"image_{modality}"].attrs['modality'] = modality
                            subj_group[f"image_{modality}"].attrs['site'] = SITE_NAMES[site]
                            subj_group[f"image_{modality}"].attrs['age'] = float(subject_meta['AGE'].iloc[0])
                            subj_group[f"image_{modality}"].attrs['sex'] = subject_meta['SEX_ID (1=m, 2=f)'].iloc[0]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate H5 dataset from IXI data')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing IXI dataset')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to output H5 file')
    parser.add_argument('--metadata_path', type=str, required=True,
                      help='Path to IXI metadata XLS file')
    parser.add_argument('--slice_range', type=int, nargs=2, default=[100, 200],
                      help='Range of slices to extract (min max)')
    parser.add_argument('--image_size', type=int, nargs=3, default=[224, 224, -1],
                      help='Size to resize images to (height width depth)')
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