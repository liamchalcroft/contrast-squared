from contextlib import nullcontext
import glob
import os
import model
import torch
import logging
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import warnings

logging.getLogger("monai").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

class IXIDataset(torch.utils.data.Dataset):
    def __init__(self, features_dir, ixi_data, ids):
        self.features_dir = features_dir
        self.ixi_data = ixi_data
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ixi_id = self.ids[idx]
        features = np.load(os.path.join(self.features_dir, f"IXI{ixi_id:03d}.npy"))
        subject_data = self.ixi_data[self.ixi_data['IXI_ID'] == ixi_id].iloc[0]
        sex = subject_data['SEX_ID (1=m, 2=f)']
        return torch.FloatTensor(features), torch.tensor(sex - 1)  # Adjusting sex to be 0 or 1

def get_loaders(features_dir, ixi_data, ids):
    dataset = IXIDataset(features_dir, ixi_data, ids)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
    return loader

def run_model(args, device):
    if args.net == "cnn":
        net = model.CNNClassifier(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,  # binary classification for sex
            features=(64, 128, 256, 512, 768, 32),
            act="GELU",
            norm="instance",
            bias=True,
            dropout=0.2,
        ).to(device)
    elif args.net == "vit":
        net = model.ViTClassifier(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,  # binary classification for sex
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

    net.load_state_dict(checkpoint["model"], strict=False)
    net.eval()

    odir = os.path.dirname(args.weights)

    # Load IXI data
    ixi_data = pd.read_excel('/home/lchalcroft/Data/IXI/IXI.xls')
    all_files = [f for f in os.listdir(args.features_dir) if f.endswith('.npy')]
    all_ids = sorted([int(f.split('.')[0][3:]) for f in all_files])
    valid_ids = [id for id in all_ids if id in ixi_data['IXI_ID'].values]

    test_loader = get_loaders(args.features_dir, ixi_data, valid_ids)

    all_preds = []
    all_labels = []
    results = []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Testing", total=len(test_loader)):
            features = features.to(device)
            labels = labels.to(device)
            
            with torch.cuda.amp.autocast() if args.amp else nullcontext():
                outputs = net(features)
                preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    results = {
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1_score": [f1]
    }

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(odir, 'sex_classification_results.csv'), index=False)

    print("\nResults Summary:")
    print(df)

def set_up():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weights", type=str, help="Path to model weights.")
    parser.add_argument(
        "--net", 
        type=str, 
        help="Network to use. Options: [cnn, vit]. Defaults to cnn.", 
        choices=["cnn", "vit"],
        default="cnn"
    )
    parser.add_argument("--features_dir", type=str, help="Directory containing feature files.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the results CSV.")
    parser.add_argument("--amp", default=False, action="store_true", help="Use automatic mixed precision.")
    parser.add_argument("--device", type=str, default=None, help="Device to use. If not specified then will check for CUDA.")
    parser.add_argument("--lowres", default=False, action="store_true", help="Use low resolution features.")
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)
    print()
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