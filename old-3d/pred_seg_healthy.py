import glob
import os
import model
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
            out_channels=4,
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
            out_channels=4,
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

    ## Generate training data for the guys t1 modality
    # Load and prepare data
    # Guy's data
    guys_img_list = glob.glob(f"/home/lchalcroft/Data/IXI/guys/{args.modality}/preprocessed/p_IXI*-{args.modality.upper()}.nii.gz")
    
    # Sort and split data from T1 to get hold-out test data
    guys_img_list.sort()
    total_samples = len(guys_img_list)
    train_size = int(0.7 * total_samples)
    val_size = int(0.1 * total_samples)
    guys_img_list = guys_img_list[train_size+val_size:]
   
    guys_img_dict = [
        {
            "image": f,
            "file": f,
            "seg": [f.replace("p_IXI", "c1p_IXI"), f.replace("p_IXI", "c2p_IXI"), f.replace("p_IXI", "c3p_IXI")],
            "dataset": "IXI",
            "modality": args.modality,
            "site": "guys",
            "IXI_ID": int(os.path.basename(f).split("-")[0][5:])
        }
        for f in guys_img_list
    ]
    
    # HH data
    hh_img_list = glob.glob(f"/home/lchalcroft/Data/IXI/hh/{args.modality}/preprocessed/p_IXI*-{args.modality.upper()}.nii.gz")
    hh_img_list.sort()
    hh_img_dict = [
        {
            "image": f,
            "file": f,
            "seg": [f.replace("p_IXI", "c1p_IXI"), f.replace("p_IXI", "c2p_IXI"), f.replace("p_IXI", "c3p_IXI")],
            "dataset": "IXI",
            "modality": args.modality,
            "site": "hh",
            "IXI_ID": int(os.path.basename(f).split("-")[0][5:])
        }
        for f in hh_img_list
    ]

    # IOP data
    iop_img_list = glob.glob(f"/home/lchalcroft/Data/IXI/iop/{args.modality}/preprocessed/p_IXI*-{args.modality.upper()}.nii.gz")
    iop_img_list.sort()
    iop_img_dict = [
        {
            "image": f,
            "file": f,
            "seg": [f.replace("p_IXI", "c1p_IXI"), f.replace("p_IXI", "c2p_IXI"), f.replace("p_IXI", "c3p_IXI")],
            "dataset": "IXI",
            "modality": args.modality,
            "site": "iop",
            "IXI_ID": int(os.path.basename(f).split("-")[0][5:])
        }
        for f in iop_img_list
    ]

    test_dict = guys_img_dict + hh_img_dict + iop_img_dict
    test_loader = get_loaders(test_dict, lowres=args.lowres)

    net.eval()
    
    window = mn.inferers.SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=2, overlap=0.5, mode="gaussian")

    # Create figure for visualization
    df_path = os.path.join(odir, 'predictions.csv')
    df = pd.read_csv(df_path) if os.path.exists(df_path) else pd.DataFrame(columns=["Dataset", "Modality", "Site", "IXI ID", "Dice", "HD95", "NSD", "Class"])

    pred_dir = os.path.join(odir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    img_saver = mn.transforms.SaveImage(
        output_postfix="img",
        output_dir=pred_dir,
        output_ext=".nii.gz",
        output_dtype=np.float32,
        resample=False,
    )
    gt_saver = mn.transforms.SaveImage(
        output_postfix="gt",
        output_dir=pred_dir,
        output_ext=".nii.gz",
        output_dtype=np.int16,
        resample=False,
    )
    pred_saver = mn.transforms.SaveImage(
        output_postfix="pred",
        output_dir=pred_dir,
        output_ext=".nii.gz",
        output_dtype=np.int16,
        resample=False,
    )

    class_names = ["Background", "Gray Matter", "White Matter", "CSF"]

    with torch.no_grad():
        for ix, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing batches"):
            image = batch["image"].to(device)
            seg = batch["seg"].to(device)
            seg_argmax = seg.argmax(dim=1)

            # Metadata
            dataset = batch["dataset"][0]
            modality = batch["modality"][0]
            site = batch["site"][0]
            ixi_id = batch["IXI_ID"][0]

            # If file already in dataframe, skip
            if df.loc[(df["Dataset"] == dataset) & (df["Modality"] == modality) & (df["Site"] == site) & (df["IXI ID"] == ixi_id)].shape[0] > 0:
                continue
            
            # Run inference
            with torch.cuda.amp.autocast() if args.amp else nullcontext():
                pred = window(image, net)
                pred = torch.softmax(pred, dim=1)
                pred_argmax = pred.argmax(dim=1)

            # For first few batches, save images
            if ix < 5:
                img_saver(image[0].detach().cpu())
                gt_saver(seg_argmax[0].detach().cpu())
                pred_saver(pred_argmax[0].detach().cpu())

            # Per channel: compute dice and hd95
            for i, class_name in enumerate(class_names):
                pred_i = pred_argmax == i
                pred_i = torch.stack([1. - pred_i, pred_i], dim=1)
                seg_i = seg_argmax == i
                seg_i = torch.stack([1. - seg_i, seg_i], dim=1)

                dice = compute_dice(pred_i, seg_i).item()
                hd95 = mn.metrics.compute_hausdorff_distance(pred_i, seg_i, include_background=False, percentile=95).item()
                nsd = mn.metrics.compute_surface_dice(pred_i, seg_i, class_thresholds=0.5, include_background=False).item()

                df = df.append({
                    "Dataset": dataset,
                    "Modality": modality,
                    "Site": site,
                    "IXI ID": ixi_id,
                    "Dice": dice,
                    "HD95": hd95,
                    "NSD": nsd,
                    "Class": class_name,
                }, ignore_index=True)

    df.to_csv(df_path, index=False)

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
    parser.add_argument("--modality", type=str, choices=["t1", "t2", "pd"], help="Modality to use.")
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
