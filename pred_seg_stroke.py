import glob
import os
import model
import preprocess_2d
import torch
import wandb
import logging
import argparse
import monai as mn
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import pandas as pd
logging.getLogger("monai").setLevel(logging.ERROR)
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    ".*pixdim*.",
)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

def add_bg(x):
    return torch.cat([1-x.sum(dim=0, keepdim=True), x], dim=0)

def get_loaders(
    data_dict,
    lowres=False,
):
    print(f"data_dict: {len(data_dict)}")

    test_transform = mn.transforms.Compose(
        transforms=[
            mn.transforms.LoadImageD(
                keys=["image", "seg"], image_only=True, allow_missing_keys=True
            ),
            mn.transforms.EnsureChannelFirstD(
                keys=["image", "seg"], allow_missing_keys=True
            ),
            mn.transforms.LambdaD(keys="seg", func=add_bg),
            mn.transforms.OrientationD(
                keys=["image", "seg"], axcodes="RAS", allow_missing_keys=True
            ),
            mn.transforms.SpacingD(
                keys=["image", "seg"],
                pixdim=1 if not lowres else 2,
                allow_missing_keys=True,
            ),
            # mn.transforms.ResizeWithPadOrCropD(
            #     keys=["image", "seg"],
            #     spatial_size=(256, 256, 256) if not lowres else (128, 128, 128),
            #     allow_missing_keys=True,
            # ),
            mn.transforms.CropForegroundD(
                keys=["image", "seg"],
                source_key="image",
                allow_missing_keys=True,
            ),
            mn.transforms.ToTensorD(
                dtype=float,
                keys=["image", "seg"],
                allow_missing_keys=True,
            ),
            mn.transforms.LambdaD(
                keys=["image", "seg"],
                func=mn.transforms.SignalFillEmpty(),
                allow_missing_keys=True,
            ),
            mn.transforms.ScaleIntensityRangePercentilesD(
                keys=["image"],
                lower=0.5, upper=99.5, b_min=0, b_max=1,
                clip=True, channel_wise=True,
            ),
            mn.transforms.HistogramNormalizeD(keys="image", min=0, max=1, allow_missing_keys=True),
            mn.transforms.NormalizeIntensityD(
                keys="image", nonzero=False, channel_wise=True
            ),
            mn.transforms.ThresholdIntensityD(
                keys=["seg"],
                threshold=0.5,
                above=True,
                cval=0,
                allow_missing_keys=True,
            ),
            mn.transforms.ToTensorD(keys=["image", "seg"], dtype=torch.float32),
        ]
    )

    data = mn.data.Dataset(data_dict, transform=test_transform)

    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=8,
    )
    return loader


def compute_dice(y_pred, y, eps=1e-8):
    y_pred = torch.flatten(y_pred)
    y = torch.flatten(y)
    y = y.float()
    intersect = (y_pred * y).sum(-1)
    denominator = (y_pred * y_pred).sum(-1) + (y * y).sum(-1)
    return 2 * (intersect / denominator.clamp(min=eps))

def run_model(args, device):
    
    if args.net == "cnn":
        net = model.CNNUNet(
            spatial_dims=3, 
            in_channels=1,
            out_channels=2,
            features=(64, 128, 256, 512, 768, 32),
            act="GELU", 
            norm="instance", 
            bias=True, 
            dropout=0.2,
            upsample="deconv",
        ).to(device)
    elif args.net == "vit":
        net = model.ViTUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            img_size=(96 if args.lowres else 192),
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            dropout_rate=0.2,
            qkv_bias=True,
            save_attn=False,
        ).to(device)

    checkpoint = torch.load(args.weights, map_location=device)
    print(
        "\nResuming from epoch #{} with WandB ID {}".format(
            checkpoint["epoch"], checkpoint["wandb"]
        )
    )
    print()

    net.load_state_dict(
        checkpoint["model"], strict=False
    )

    odir = os.path.dirname(args.weights)

    atlas_t1_test_list = np.loadtxt("/home/lchalcroft/git/lab-vae/atlas_test.txt", dtype=str)
    arc_t1_test_list = glob.glob("/home/lchalcroft/Data/ARC/PREPROC/sub-*/ses-*/*_T1w_flirt.nii.gz")
    arc_t2_test_list = glob.glob("/home/lchalcroft/Data/ARC/PREPROC/sub-*/ses-*/*_T2w.nii.gz")
    arc_flair_test_list = glob.glob("/home/lchalcroft/Data/ARC/PREPROC/sub-*/ses-*/*_FLAIR_flirt.nii.gz")

    atlas_t1_test_dict = [
        {
            "seg": f.replace("1mm_sub", "sub"),
            "image": f.replace("_label-L_desc-T1lesion_mask", "_T1w").replace(
                "1mm_sub", "sub"
            ),
            "file": f.replace("1mm_sub", "sub"),
            "dataset": "ATLAS",
            "modality": "T1w"
        }
        for f in atlas_t1_test_list
    ]
    arc_t1_test_dict = [
        {
            "image": f,
            "file": f,
            "seg": glob.glob(os.path.join(os.path.dirname(f), "*_T2w_desc-lesion_mask.nii.gz"))[0],
            "dataset": "ARC",
            "modality": "T1w"
        }
        for f in arc_t1_test_list
    ]
    arc_t2_test_dict = [
        {
            "image": f,
            "file": f,
            "seg": glob.glob(os.path.join(os.path.dirname(f), "*_T2w_desc-lesion_mask.nii.gz"))[0],
            "dataset": "ARC",
            "modality": "T2w"
        }
        for f in arc_t2_test_list
    ]
    arc_flair_test_dict = [
        {
            "image": f,
            "file": f,
            "seg": glob.glob(os.path.join(os.path.dirname(f), "*_T2w_desc-lesion_mask.nii.gz"))[0],
            "dataset": "ARC",
            "modality": "FLAIR"
        }
        for f in arc_flair_test_list
    ]
    
    print(f"Test data: \nATLAS T1w: {len(atlas_t1_test_dict)}\nARC T1w: {len(arc_t1_test_dict)}\nARC T2w: {len(arc_t2_test_dict)}\nARC FLAIR: {len(arc_flair_test_dict)}")
    test_dict = atlas_t1_test_dict + arc_t1_test_dict + arc_t2_test_dict + arc_flair_test_dict
    test_loader = get_loaders(test_dict, lowres=args.lowres)

    net.eval()
    
    window = mn.inferers.SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=2, overlap=0.5, mode="gaussian")

    # Create figure for visualization
    fig_dir = os.path.join(odir, 'debug_visualizations')
    os.makedirs(fig_dir, exist_ok=True)

    datasets = ['ATLAS', 'ARC']
    modalities = ['T1w', 'T2w', 'FLAIR']

    with torch.no_grad():
        for dataset in datasets:
            for modality in modalities:
                # Skip ATLAS combinations that don't exist
                if dataset == 'ATLAS' and modality != 'T1w':
                    continue
                    
                count = 0
                fig, axes = plt.subplots(5, 3, figsize=(15, 25))
                fig.suptitle(f'{dataset} - {modality}')
                
                for batch in test_loader:
                    # Skip if not the current dataset/modality
                    if batch["dataset"][0] != dataset or batch["modality"][0] != modality:
                        continue
                    
                    if count >= 5:
                        break

                    image = batch["image"].to(device)
                    seg = batch["seg"].to(device)
                    seg_argmax = seg.argmax(dim=1)
                    
                    # Run inference
                    with torch.cuda.amp.autocast() if args.amp else nullcontext():
                        pred = window(image, net)
                        pred = torch.softmax(pred, dim=1)
                        pred_argmax = pred.argmax(dim=1)

                    # Get middle slices
                    mid_slice = image.shape[-1] // 2
                    image_slice = image[0, 0, :, :, mid_slice].cpu()
                    seg_slice = seg_argmax[0, :, :, mid_slice].cpu()
                    pred_slice = pred_argmax[0, :, :, mid_slice].cpu()

                    # Plot
                    axes[count, 0].imshow(image_slice, cmap='gray')
                    axes[count, 0].set_title(f'Input - {os.path.basename(batch["file"][0])}')
                    axes[count, 0].axis('off')
                    
                    axes[count, 1].imshow(seg_slice, cmap='tab10', vmin=0, vmax=1)
                    axes[count, 1].set_title('Ground Truth')
                    axes[count, 1].axis('off')
                    
                    axes[count, 2].imshow(pred_slice, cmap='tab10', vmin=0, vmax=1)
                    axes[count, 2].set_title('Prediction')
                    axes[count, 2].axis('off')

                    count += 1

                # If we didn't find any cases, skip saving this figure
                if count == 0:
                    plt.close()
                    continue

                # Fill remaining slots with empty plots if less than 5 cases
                while count < 5:
                    axes[count, 0].axis('off')
                    axes[count, 1].axis('off')
                    axes[count, 2].axis('off')
                    count += 1

                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f'debug_{dataset}_{modality}.png'))
                plt.close()

    print(f"\nDebug visualizations saved to: {fig_dir}")

def set_up():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weights", type=str, help="Path to model weights.")
    parser.add_argument(
        "--net", 
        type=str, 
        help="Encoder network to use. Options: [cnn, vit]. Defaults to cnn.", 
        choices=["cnn", "vit"],
        default="cnn"
    )
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument("--device", type=str, default=None, help="Device to use. If not specified then will check for CUDA.")
    parser.add_argument("--lowres", default=False, action="store_true", help="Train with 2mm resolution images.")
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)
    print()
    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
    return args, device


def main():
    args, device = set_up()
    run_model(args, device)


if __name__ == "__main__":
    main()
