import glob
import os
import model
import preprocess_3d
import torch
import wandb
import logging
import argparse
import monai as mn
import utils
import numpy as np
from monai.inferers import sliding_window_inference
from contextlib import nullcontext
from tqdm import tqdm
from torchvision.utils import make_grid
logging.getLogger("monai").setLevel(logging.ERROR)
import warnings

warnings.filterwarnings(
    "ignore",
    ".*pixdim*.",
)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


def run_model(args, device):

    data_transforms = mn.transforms.Compose([
        mn.transforms.LoadImaged(keys="image"),
        mn.transforms.EnsureChannelFirstD(keys="image"),
        mn.transforms.OrientationD(keys="image", axcodes="RAS"),
        mn.transforms.SpacingD(keys="image", pixdim=(1.0, 1.0, 1.0)),
        mn.transforms.LambdaD(
                keys="image", func=mn.transforms.SignalFillEmpty()
        ),
        mn.transforms.ScaleIntensityRangePercentilesD(keys=["image"],
            lower=0.5,upper=95,b_min=0,b_max=1,clip=True,channel_wise=True),
        mn.transforms.HistogramNormalizeD(keys=["image"], min=0, max=1),
        mn.transforms.NormalizeIntensityD(keys="image", channel_wise=True),
        mn.transforms.LambdaD(
                keys="image", func=mn.transforms.SignalFillEmpty()
        ),
        mn.transforms.ToTensorD(
            dtype=torch.float32, keys="image", device=device
        ),
    ])
    
    if args.net == "cnn":
        encoder = model.CNNEncoder(
            spatial_dims=3, 
            in_channels=1, 
            features=(64, 128, 256, 512, 768), 
            act="GELU", 
            norm="instance", 
            bias=True, 
            dropout=0.2
        ).to(device)
    elif args.net == "vit":
        encoder = model.ViTEncoder(
            spatial_dims=3,
            in_channels=1,
            img_size=(48 if args.lowres else 96),
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            dropout_rate=0.2,
            qkv_bias=True,
            save_attn=False
        ).to(device)

    ckpts = os.path.join(args.logdir, args.name, "checkpoint.pt")

    encoder.load_state_dict(
        torch.load(ckpts, map_location=device)["encoder"], strict=True
    )
    
    encoder.eval()

    # Data setup
    guys_t1 = sorted(glob.glob(os.path.join("/home/lchalcroft/Data/IXI/guys/t1/*-T1.nii.gz")))
    guys_t2 = sorted(glob.glob(os.path.join("/home/lchalcroft/Data/IXI/guys/t2/*-T2.nii.gz")))
    guys_pd = sorted(glob.glob(os.path.join("/home/lchalcroft/Data/IXI/guys/pd/*-PD.nii.gz")))
    guys_t1_dict = [{"image": f} for f in guys_t1]
    guys_t2_dict = [{"image": f} for f in guys_t2]
    guys_pd_dict = [{"image": f} for f in guys_pd]
    hh_t1 = sorted(glob.glob(os.path.join("/home/lchalcroft/Data/IXI/hh/t1/*-T1.nii.gz")))
    hh_t2 = sorted(glob.glob(os.path.join("/home/lchalcroft/Data/IXI/hh/t2/*-T2.nii.gz")))
    hh_pd = sorted(glob.glob(os.path.join("/home/lchalcroft/Data/IXI/hh/pd/*-PD.nii.gz")))
    hh_t1_dict = [{"image": f} for f in hh_t1]
    hh_t2_dict = [{"image": f} for f in hh_t2]
    hh_pd_dict = [{"image": f} for f in hh_pd]
    iop_t1 = sorted(glob.glob(os.path.join("/home/lchalcroft/Data/IXI/iop/t1/*-T1.nii.gz")))
    iop_t2 = sorted(glob.glob(os.path.join("/home/lchalcroft/Data/IXI/iop/t2/*-T2.nii.gz")))
    iop_pd = sorted(glob.glob(os.path.join("/home/lchalcroft/Data/IXI/iop/pd/*-PD.nii.gz")))
    iop_t1_dict = [{"image": f} for f in iop_t1]
    iop_t2_dict = [{"image": f} for f in iop_t2]
    iop_pd_dict = [{"image": f} for f in iop_pd]

    odir = os.path.join(args.logdir, args.name, "ixi-features")
    os.makedirs(os.path.join(odir, "guys", "t1"), exist_ok=True)
    os.makedirs(os.path.join(odir, "guys", "t2"), exist_ok=True)
    os.makedirs(os.path.join(odir, "guys", "pd"), exist_ok=True)
    os.makedirs(os.path.join(odir, "hh", "t1"), exist_ok=True)
    os.makedirs(os.path.join(odir, "hh", "t2"), exist_ok=True)
    os.makedirs(os.path.join(odir, "hh", "pd"), exist_ok=True)
    os.makedirs(os.path.join(odir, "iop", "t1"), exist_ok=True)

    # Loop over all sites
    # for pt_dict in tqdm(guys_t1_dict, desc="Guys T1", total=len(guys_t1_dict)):
    #     data_dict = data_transforms(pt_dict)
    #     with torch.no_grad():
    #         features = sliding_window_inference(data_dict["image"].unsqueeze(0).to(device), (96, 96, 96), 1, encoder)
    #         features = features.reshape(features.shape[1], -1).mean(-1).cpu().numpy()
    #         np.save(os.path.join(odir, "guys", "t1", os.path.basename(pt_dict["image"].split("-")[0])), features)
    # for pt_dict in tqdm(guys_t2_dict, desc="Guys T2", total=len(guys_t2_dict)):
    #     data_dict = data_transforms(pt_dict)
    #     with torch.no_grad():
    #         features = sliding_window_inference(data_dict["image"].unsqueeze(0).to(device), (96, 96, 96), 1, encoder)
    #         features = features.reshape(features.shape[1], -1).mean(-1).cpu().numpy()
    #         np.save(os.path.join(odir, "guys", "t2", os.path.basename(pt_dict["image"].split("-")[0])), features)
    # for pt_dict in tqdm(guys_pd_dict, desc="Guys PD", total=len(guys_pd_dict)):
    #     data_dict = data_transforms(pt_dict)
    #     with torch.no_grad():
    #         features = sliding_window_inference(data_dict["image"].unsqueeze(0).to(device), (96, 96, 96), 1, encoder)
    #         features = features.reshape(features.shape[1], -1).mean(-1).cpu().numpy()
    #         np.save(os.path.join(odir, "guys", "pd", os.path.basename(pt_dict["image"].split("-")[0])), features)
    # for pt_dict in tqdm(hh_t1_dict, desc="HH T1", total=len(hh_t1_dict)):
    #     data_dict = data_transforms(pt_dict)
    #     with torch.no_grad():
    #         features = sliding_window_inference(data_dict["image"].unsqueeze(0).to(device), (96, 96, 96), 1, encoder)
    #         features = features.reshape(features.shape[1], -1).mean(-1).cpu().numpy()
    #         np.save(os.path.join(odir, "hh", "t1", os.path.basename(pt_dict["image"].split("-")[0])), features)
    # for pt_dict in tqdm(hh_t2_dict, desc="HH T2", total=len(hh_t2_dict)):
    #     data_dict = data_transforms(pt_dict)
    #     with torch.no_grad():
    #         features = sliding_window_inference(data_dict["image"].unsqueeze(0).to(device), (96, 96, 96), 1, encoder)
    #         features = features.reshape(features.shape[1], -1).mean(-1).cpu().numpy()
    #         np.save(os.path.join(odir, "hh", "t2", os.path.basename(pt_dict["image"].split("-")[0])), features)
    # for pt_dict in tqdm(hh_pd_dict, desc="HH PD", total=len(hh_pd_dict)):
    #     data_dict = data_transforms(pt_dict)
    #     with torch.no_grad():
    #         features = sliding_window_inference(data_dict["image"].unsqueeze(0).to(device), (96, 96, 96), 1, encoder)
    #         features = features.reshape(features.shape[1], -1).mean(-1).cpu().numpy()
    #         np.save(os.path.join(odir, "hh", "pd", os.path.basename(pt_dict["image"].split("-")[0])), features)
    # for pt_dict in tqdm(iop_t1_dict, desc="IOP T1", total=len(iop_t1_dict)):
    #     data_dict = data_transforms(pt_dict)
    #     with torch.no_grad():
    #         features = sliding_window_inference(data_dict["image"].unsqueeze(0).to(device), (96, 96, 96), 1, encoder)
    #         features = features.reshape(features.shape[1], -1).mean(-1).cpu().numpy()
    #         np.save(os.path.join(odir, "iop", "t1", os.path.basename(pt_dict["image"].split("-")[0])), features)
    for pt_dict in tqdm(iop_t2_dict, desc="IOP T2", total=len(iop_t2_dict)):
        data_dict = data_transforms(pt_dict)
        with torch.no_grad():
            features = sliding_window_inference(data_dict["image"].unsqueeze(0).to(device), (96, 96, 96), 1, encoder)
            features = features.reshape(features.shape[1], -1).mean(-1).cpu().numpy()
            np.save(os.path.join(odir, "iop", "t2", os.path.basename(pt_dict["image"].split("-")[0])), features)
    for pt_dict in tqdm(iop_pd_dict, desc="IOP PD", total=len(iop_pd_dict)):
        data_dict = data_transforms(pt_dict)
        with torch.no_grad():
            features = sliding_window_inference(data_dict["image"].unsqueeze(0).to(device), (96, 96, 96), 1, encoder)
            features = features.reshape(features.shape[1], -1).mean(-1).cpu().numpy()
            np.save(os.path.join(odir, "iop", "pd", os.path.basename(pt_dict["image"].split("-")[0])), features)

def set_up():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, help="Name of WandB run.")
    parser.add_argument(
        "--net", 
        type=str, 
        help="Encoder network to use. Options: [cnn, vit]. Defaults to cnn.", 
        choices=["cnn", "vit"],
        default="cnn"
    )
    parser.add_argument("--logdir", type=str, default="./", help="Path to saved outputs")
    parser.add_argument("--device", type=str, default=None, help="Device to use. If not specified then will check for CUDA.")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
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
