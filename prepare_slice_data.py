import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm

def load_and_preprocess(file_path):
    # Load the image
    img = nib.load(file_path)

    # Ensure RAS orientation using nibabel
    img = nib.as_closest_canonical(img)

    # Get the data
    data = img.get_fdata().astype(np.uint16)
    
    return data

def find_brain_slices(data, threshold=0.05):
    # Normalize data
    data_norm = (data - data.min()) / (data.max() - data.min())
    
    # Find slices with brain tissue
    brain_slices = []
    for i in range(data_norm.shape[2]):
        if np.mean(data_norm[:,:,i]) > threshold:
            brain_slices.append(i)
    
    return brain_slices

def save_2d_nifti(data, output_path, affine):
    # Create a new affine for 2D
    affine_2d = affine.copy()
    # affine_2d = affine_2d[:3, :3]  # Remove z-dimension
    
    # Save as 2D NIfTI
    img_2d = nib.Nifti1Image(data, affine_2d)
    nib.save(img_2d, output_path)

def process_files(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all NIfTI files
    nifti_files = glob.glob(
        os.path.join(input_dir, "*/*/fg_mask.nii"),
    )
    nifti_dicts = [
        {
            "mask": f,
            "pd": f.replace("fg_mask.nii", "masked_pd.nii"),
            "r1": f.replace("fg_mask.nii", "masked_r1.nii"),
            "r2s": f.replace("fg_mask.nii", "masked_r2s.nii"),
            "mt": f.replace("fg_mask.nii", "masked_mt.nii"),
            "mprage": f.replace("fg_mask.nii", "sim_mprage.nii"),
            "seg": f.replace("fg_mask.nii", "mb_labels.nii"),
            "lesion": f.replace("fg_mask.nii", "lesion.nii"),
        }
        for f in nifti_files
    ]

    
    for nifti_dict in tqdm(nifti_dicts, desc="Processing files", total=len(nifti_dicts)):
        # Get file paths
        mask_path = nifti_dict["mask"]
        pd_path = nifti_dict["pd"]
        r1_path = nifti_dict["r1"]
        r2s_path = nifti_dict["r2s"]
        mt_path = nifti_dict["mt"]
        mprage_path = nifti_dict["mprage"]
        seg_path = nifti_dict["seg"]
        lesion_path = nifti_dict["lesion"]

        # Get name for subject - want to just have the diagnosis and the scan number separated by an underscore
        subject_name = mask_path.split("/fg_mask.nii")[0].split("MPM_DATA/")[1].replace("/", "_")

        # Load and preprocess the image
        mask_data = load_and_preprocess(mask_path)
        pd_data = load_and_preprocess(pd_path)
        r1_data = load_and_preprocess(r1_path)
        r2s_data = load_and_preprocess(r2s_path)
        mt_data = load_and_preprocess(mt_path)
        mprage_data = load_and_preprocess(mprage_path)
        seg_data = load_and_preprocess(seg_path)
        if os.path.exists(lesion_path):
            lesion_data = load_and_preprocess(lesion_path)
        else:
            lesion_data = np.zeros_like(mask_data)
        # Get affine matrices
        orig_affine = nib.load(mask_path).affine
        
        # Find brain slices
        brain_slices = find_brain_slices(mask_data)
        
        # Save each brain slice as a 2D NIfTI
        for i, slice_idx in enumerate(brain_slices):
            # PD
            slice_data = pd_data[:,:,slice_idx]
            output_filename = f"{subject_name}_slice_{i:03d}_pd.nii.gz"
            output_path = os.path.join(output_dir, output_filename)
            save_2d_nifti(slice_data, output_path, orig_affine)

            # R1
            slice_data = r1_data[:,:,slice_idx]
            output_filename = f"{subject_name}_slice_{i:03d}_r1.nii.gz"
            output_path = os.path.join(output_dir, output_filename)
            save_2d_nifti(slice_data, output_path, orig_affine)

            # R2s
            slice_data = r2s_data[:,:,slice_idx]
            output_filename = f"{subject_name}_slice_{i:03d}_r2s.nii.gz"
            output_path = os.path.join(output_dir, output_filename)
            save_2d_nifti(slice_data, output_path, orig_affine)

            # MT
            slice_data = mt_data[:,:,slice_idx]
            output_filename = f"{subject_name}_slice_{i:03d}_mt.nii.gz"
            output_path = os.path.join(output_dir, output_filename)
            save_2d_nifti(slice_data, output_path, orig_affine)

            # MPRAGE
            slice_data = mprage_data[:,:,slice_idx]
            output_filename = f"{subject_name}_slice_{i:03d}_mprage.nii.gz"
            output_path = os.path.join(output_dir, output_filename)
            save_2d_nifti(slice_data, output_path, orig_affine)

            # SEG
            slice_data = seg_data[:,:,slice_idx]
            output_filename = f"{subject_name}_slice_{i:03d}_seg.nii.gz"
            output_path = os.path.join(output_dir, output_filename)
            save_2d_nifti(slice_data, output_path, orig_affine)

            # LESION
            slice_data = lesion_data[:,:,slice_idx]
            output_filename = f"{subject_name}_slice_{i:03d}_lesion.nii.gz"
            output_path = os.path.join(output_dir, output_filename)
            save_2d_nifti(slice_data, output_path, orig_affine)

if __name__ == "__main__":
    input_directory = "/home/lchalcroft/MPM_DATA/"
    output_directory = "/home/lchalcroft/MPM_DATA/slices"
    process_files(input_directory, output_directory)
