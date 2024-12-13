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

def get_first_channel(x):
    return x[[0]]

def get_second_channel(x):
    return x[[1]]

def get_augmentations(keys, ptch):
    spatial_augs = [
        mn.transforms.Rand3DElasticD(keys=keys, sigma_range=(5,7), magnitude_range=(50,150),
                                        rotate_range=15, shear_range=0.012, scale_range=0.15,
                                        padding_mode='zeros', prob=0.8),
        mn.transforms.RandAxisFlipd(keys=keys, prob=1),
        mn.transforms.RandAxisFlipd(keys=keys, prob=1),
        mn.transforms.RandAxisFlipd(keys=keys, prob=1),
        mn.transforms.RandSpatialCropD(
            keys=keys, roi_size=ptch, random_size=False
        ),
        mn.transforms.ResizeWithPadOrCropD(
            keys=keys, spatial_size=(ptch, ptch, ptch)
        ),
    ]

    recon_store = [
        mn.transforms.CopyItemsD(keys=keys, names=[key+"_recon" for key in keys])
    ]
    
    intensity_augs = [
        mn.transforms.RandBiasFieldD(keys=keys, prob=1, coeff_range=(0, 0.3)),
        mn.transforms.RandGibbsNoised(keys=keys, prob=1, alpha=(0, 0.8)),
        mn.transforms.RandRicianNoised(keys=keys, prob=1, mean=0.0, std=0.2, relative=False, sample_std=True),
        mn.transforms.RandCoarseDropoutD(keys=keys, prob=1, holes=1, max_holes=10, spatial_size=5, max_spatial_size=ptch//4, fill_value=0)
          ]
    
    augment_image = [*spatial_augs, *recon_store, *intensity_augs]

    return augment_image

def get_prepare_data(keys, ptch):
    prepare_data = [
        mn.transforms.LambdaD(
                keys=keys, func=mn.transforms.SignalFillEmpty()
        ),
        mn.transforms.NormalizeIntensityD(keys=keys, channel_wise=True),
        mn.transforms.ResizeD(
            keys=keys, spatial_size=(ptch, ptch, ptch)
        ),
        mn.transforms.ToTensorD(
            dtype=torch.float32, keys=keys
        ),
    ]
    return prepare_data

def get_bloch_loader(
    batch_size=1,
    device="cpu",
    lowres=False,
    same_contrast=False,
):
    train_files = glob.glob(
        os.path.join("/home/lchalcroft/MPM_DATA/*/*/masked_pd.nii"),
    )
    train_dict = [
        {
            "image1": [
                f,
                f.replace("masked_pd.nii", "masked_r1.nii"),
                f.replace("masked_pd.nii", "masked_r2s.nii"),
                f.replace("masked_pd.nii", "masked_mt.nii"),
            ],
        }
        for f in train_files
    ]

    shuffle(train_dict)

    ptch = 48 if lowres else 96

    prepare_mpm = [mn.transforms.CopyItemsD(keys=["image1"], names=["path"]),
            mn.transforms.LoadImageD(keys=["image1"], image_only=True),
            mn.transforms.EnsureChannelFirstD(keys=["image1"]),
            mn.transforms.LambdaD(keys=["image1"], func=mn.transforms.SignalFillEmpty()),
            mn.transforms.LambdaD(keys=["image1"], func=rescale_mpm),
            ClipPercentilesD(
                keys=["image1"],
                lower=0.5,
                upper=99.5,
            ),  # just clip extreme values, don't rescale
            mn.transforms.LambdaD(keys=["image1"], func=mn.transforms.SignalFillEmpty()),
            mn.transforms.RandSpatialCropD(
                keys=["image1"], roi_size=int(1.2*ptch), random_size=False
            ),
            # mn.transforms.SpacingD(keys=["image1"], pixdim=2 if lowres else 1),
            mn.transforms.ToTensorD(dtype=torch.float32, keys=["image1"], device=device)]
    
    if same_contrast:
       generate_bloch = [
          MONAIBlochTransformD(keys=["image1"], num_ch=1),
          mn.transforms.ScaleIntensityRangePercentilesD(keys=["image1"],
            lower=0.5,upper=95,b_min=0,b_max=1,clip=True,channel_wise=True),
          mn.transforms.CopyItemsD(keys=["image1"], names=["image2"]),
          mn.transforms.HistogramNormalizeD(keys=["image1", "image2"], min=0, max=1),
                        ]
    else:
      generate_bloch = [
          MONAIBlochTransformD(keys=["image1"], num_ch=2),
          mn.transforms.ScaleIntensityRangePercentilesD(keys=["image1"],
            lower=0.5,upper=95,b_min=0,b_max=1,clip=True,channel_wise=True),
          mn.transforms.CopyItemsD(keys=["image1"], names=["image2"]),
          mn.transforms.LambdaD(keys=["image1"], func=get_first_channel),
          mn.transforms.LambdaD(keys=["image2"], func=get_second_channel),
          mn.transforms.HistogramNormalizeD(keys=["image1", "image2"], min=0, max=1),
                        ]

    augment_image1 = get_augmentations(["image1"], ptch)
    augment_image2 = get_augmentations(["image2"], ptch)

    prepare_images = get_prepare_data(["image1", "image2", "image1_recon", "image2_recon"], ptch)

    train_transform = mn.transforms.Compose(
        transforms=[
            *prepare_mpm,
            *generate_bloch,
            *augment_image1,
            *augment_image2,
            *prepare_images,
        ]
    )

    train_data = mn.data.Dataset(train_dict, transform=train_transform)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
    )

    return train_loader, train_transform

def get_mprage_loader(
    batch_size=1,
    device="cpu",
    lowres=False,
):
    train_files = glob.glob(
        os.path.join("/home/lchalcroft/MPM_DATA/*/*/sim_mprage.nii"),
    )
    train_dict = [
        {"image1": f,}
        for f in train_files
    ]

    shuffle(train_dict)

    ptch = 48 if lowres else 96

    prepare_mprage = [mn.transforms.CopyItemsD(keys=["image1"], names=["path"]),
            mn.transforms.LoadImageD(keys=["image1"], image_only=True),
            mn.transforms.EnsureChannelFirstD(keys=["image1"]),
            mn.transforms.LambdaD(keys=["image1"], func=mn.transforms.SignalFillEmpty()),
            ClipPercentilesD(
                keys=["image1"],
                lower=0.5,
                upper=99.5,
            ),  # just clip extreme values, don't rescale
            mn.transforms.LambdaD(keys=["image1"], func=mn.transforms.SignalFillEmpty()),
            # mn.transforms.SpacingD(keys=["image1"], pixdim=2 if lowres else 1),
            mn.transforms.RandSpatialCropD(
                keys=["image1"], roi_size=int(1.2*ptch), random_size=False
            ),
            mn.transforms.ToTensorD(dtype=torch.float32, keys=["image1"], device=device)]
    
    generate_views = [
      mn.transforms.ScaleIntensityRangePercentilesD(keys=["image1"],
            lower=0.5,upper=99.5,b_min=0,b_max=1,clip=True,channel_wise=True),
      mn.transforms.CopyItemsD(keys=["image1"], names=["image2"]),
      mn.transforms.HistogramNormalizeD(keys=["image1", "image2"], min=0, max=1),
    ]

    augment_image1 = get_augmentations(["image1"], ptch)
    augment_image2 = get_augmentations(["image2"], ptch)

    prepare_images = get_prepare_data(["image1", "image2", "image1_recon", "image2_recon"], ptch)

    train_transform = mn.transforms.Compose(
        transforms=[
            *prepare_mprage,
            *generate_views,
            *augment_image1,
            *augment_image2,
            *prepare_images,
        ]
    )

    train_data = mn.data.Dataset(train_dict, transform=train_transform)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
    )

    return train_loader, train_transform

def save_slice_as_png(slice_data, output_path):
    # Normalize to 0-255 range
    slice_data = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
    img = Image.fromarray(slice_data)
    img.save(output_path)

def generate_qmri_slices(input_files, output_dir, num_contrasts=100):
    os.makedirs(output_dir, exist_ok=True)
    
    prepare_mpm = [
        mn.transforms.LoadImageD(keys=["image"], image_only=True),
        mn.transforms.EnsureChannelFirstD(keys=["image"]),
        mn.transforms.LambdaD(keys=["image"], func=mn.transforms.SignalFillEmpty()),
        mn.transforms.LambdaD(keys=["image"], func=rescale_mpm),
        ClipPercentilesD(keys=["image"], lower=0.5, upper=99.5),
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
        
        # Generate random contrasts and slices
        for i in range(num_contrasts):
            # Random slice index for each axis
            for axis in range(3):
                slice_idx = randint(0, volume.shape[axis + 1] - 1)
                if axis == 0:
                    slice_data = volume[:, slice_idx, :, :]
                elif axis == 1:
                    slice_data = volume[:, :, slice_idx, :]
                else:
                    slice_data = volume[:, :, :, slice_idx]
                
                # Generate random contrast
                bloch_transform = MONAIBlochTransformD(keys=["image"], num_ch=1)
                contrast = bloch_transform({"image": slice_data})["image"]
                
                # Save the slice
                output_path = os.path.join(
                    output_dir,
                    f"{subject_id}_contrast{i}_axis{axis}.png"
                )
                save_slice_as_png(contrast[0].numpy(), output_path)

def generate_mprage_slices(input_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    prepare_mprage = [
        mn.transforms.LoadImageD(keys=["image"], image_only=True),
        mn.transforms.EnsureChannelFirstD(keys=["image"]),
        ClipPercentilesD(keys=["image"], lower=0.5, upper=99.5),
    ]
    
    transform = mn.transforms.Compose(prepare_mprage)
    
    for file_path in tqdm.tqdm(input_files):
        subject_id = os.path.basename(os.path.dirname(file_path))
        data_dict = {"image": file_path}
        
        # Load and transform the data
        data = transform(data_dict)
        volume = data["image"]  # Shape: [1, D, H, W]
        
        # Generate slices for each axis
        for axis in range(3):
            for slice_idx in range(volume.shape[axis + 1]):
                if axis == 0:
                    slice_data = volume[:, slice_idx, :, :]
                elif axis == 1:
                    slice_data = volume[:, :, slice_idx, :]
                else:
                    slice_data = volume[:, :, :, slice_idx]
                
                output_path = os.path.join(
                    output_dir,
                    f"{subject_id}_axis{axis}_slice{slice_idx}.png"
                )
                save_slice_as_png(slice_data[0], output_path)

# Add this at the bottom of the file to run the generation
if __name__ == "__main__":
    qmri_files = glob.glob(os.path.join("/home/lchalcroft/MPM_DATA/*/*/masked_pd.nii"))
    mprage_files = glob.glob(os.path.join("/home/lchalcroft/MPM_DATA/*/*/sim_mprage.nii"))
    
    generate_qmri_slices(qmri_files, "output/qmri_slices", num_contrasts=100)
    generate_mprage_slices(mprage_files, "output/mprage_slices")
