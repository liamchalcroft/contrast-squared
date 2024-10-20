import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import csv
import monai as mn
import model
import glob

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define dataloaders
def get_loaders(args, train_ids, val_ids):
    train_transform = mn.transforms.Compose(
        transforms=[
            mn.transforms.LoadImageD(
                keys=["image"], image_only=True, allow_missing_keys=True
            ),
            mn.transforms.EnsureChannelFirstD(
                keys=["image"], allow_missing_keys=True
            ),
            mn.transforms.OrientationD(
                keys=["image"], axcodes="RAS", allow_missing_keys=True
            ),
            mn.transforms.SpacingD(
                keys=["image"],
                pixdim=1 if not args.lowres else 2,
                allow_missing_keys=True,
            ),
            mn.transforms.ResizeWithPadOrCropD(
                keys=["image"],
                spatial_size=(256, 256, 256) if not args.lowres else (128, 128, 128),
                allow_missing_keys=True,
            ),
            mn.transforms.ToTensorD(
                dtype=float,
                keys=["image"],
                allow_missing_keys=True,
            ),
            mn.transforms.LambdaD(
                keys=["image"],
                func=mn.transforms.SignalFillEmpty(),
                allow_missing_keys=True,
            ),
            mn.transforms.RandAffineD(
                keys=["image"],
                rotate_range=15, shear_range=0.012, scale_range=0.15,
                prob=0.8, cache_grid=True, spatial_size=(256, 256, 256) if not args.lowres else (128, 128, 128),
                allow_missing_keys=True,
            ),
            mn.transforms.RandBiasFieldD(keys="image", prob=0.8),
            mn.transforms.RandAxisFlipd(
                keys=["image"],
                prob=0.8,
                allow_missing_keys=True,
            ),
            mn.transforms.RandAxisFlipd(
                keys=["image"],
                prob=0.8,
                allow_missing_keys=True,
            ),
            mn.transforms.RandAxisFlipd(
                keys=["image"],
                prob=0.8,
                allow_missing_keys=True,
            ),
            mn.transforms.ScaleIntensityRangePercentilesD(
                keys=["image"],
                lower=0.5, upper=95, b_min=0, b_max=1,
                clip=True, channel_wise=True,
            ),
            mn.transforms.HistogramNormalizeD(keys="image", min=0, max=1, allow_missing_keys=True),
            mn.transforms.NormalizeIntensityD(
                keys="image", nonzero=False, channel_wise=True
            ),
            mn.transforms.RandGaussianNoiseD(keys="image", prob=0.8),
            mn.transforms.RandSpatialCropD(
                keys=["image"],
                roi_size=(96, 96, 96) if not args.lowres else (48, 48, 48),
                random_size=False,
                allow_missing_keys=True,
            ),
            mn.transforms.ResizeD(
                keys=["image"],
                spatial_size=(96, 96, 96) if not args.lowres else (48, 48, 48),
                allow_missing_keys=True,
            ),
            mn.transforms.ToTensorD(keys=["image"], dtype=torch.float32),
        ]
    )

    id = train_ids[0]
    print(f"/home/lchalcroft/Data/IXI/guys/t1/{id:03d}*T1.nii.gz")
    train_data = [
        {"image": glob.glob(f"/home/lchalcroft/Data/IXI/guys/t1/{id:03d}*T1.nii.gz")[0], "IXI_ID": id} for id in train_ids
    ]
    val_data = [
        {"image": glob.glob(f"/home/lchalcroft/Data/IXI/guys/t1/{id:03d}*T1.nii.gz")[0], "IXI_ID": id} for id in val_ids
    ]

    train_dataset = mn.data.Dataset(data=train_data, transform=train_transform)
    val_dataset = mn.data.Dataset(data=val_data, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    return train_loader, val_loader

# Define the model
class BrainAgeRegressor(nn.Module):
    def __init__(self, input_size):
        super(BrainAgeRegressor, self).__init__()
        self.linear = nn.Linear(input_size, 1, bias=True)
    
    def forward(self, x):
        return self.linear(x)

def run_model(args):
    # Load IXI spreadsheet
    ixi_data = pd.read_excel('/home/lchalcroft/Data/IXI/IXI.xls')
    print(ixi_data.head())

    # Load and prepare data
    features_dir = os.path.join(args.logdir, args.name, "ixi-features/guys/t1")
    all_files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
    all_ids = sorted([int(f.split('.')[0][3:]) for f in all_files])

    # Initialise encoder and load weights
    if args.net == "cnn":
        encoder = model.CNNEncoder(
            spatial_dims=3, 
            in_channels=1, 
            features=(64, 128, 256, 512, 768), 
            act="GELU", 
            norm="instance", 
            bias=True, 
            dropout=0.2
        ).to(args.device)
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
        ).to(args.device)
    weights = os.path.join(args.logdir, args.name, "checkpoint.pt")
    encoder.load_state_dict(torch.load(weights, map_location=args.device)["encoder"], strict=True)
    encoder.eval()

    # Ensure all IDs in all_ids are present in ixi_data
    valid_ids = [id for id in all_ids if id in ixi_data['IXI_ID'].values]
    # Remove duplicate IDs, keeping only the first occurrence
    ixi_data = ixi_data.drop_duplicates(subset=['IXI_ID'], keep='first')
    # Filter valid IDs
    valid_ids = [id for id in valid_ids if not pd.isna(ixi_data[ixi_data['IXI_ID'] == id]['AGE'].iloc[0])]

    # Sort and split data
    valid_ids.sort()
    total_samples = len(valid_ids)
    train_size = int(0.7 * total_samples)
    val_size = int(0.1 * total_samples)

    train_ids = valid_ids[:train_size]
    val_ids = valid_ids[train_size:train_size+val_size]
    test_ids = valid_ids[train_size+val_size:]

    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {len(test_ids)}")

    # Create datasets
    train_loader, val_loader = get_loaders(args, train_ids, val_ids)

    # Initialize model, loss function, and optimizer
    net = BrainAgeRegressor(769) # 768 features + 1 sex
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')

    # Create dir for model
    model_dir = os.path.join(args.logdir, args.name, 'ixi-classifier')
    os.makedirs(model_dir, exist_ok=True)

    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []

    def get_metadata(ixi_id, ixi_data):
        subject_data = ixi_data[ixi_data['IXI_ID'] == ixi_id].iloc[0]
        age = subject_data['AGE']
        sex = subject_data['SEX_ID (1=m, 2=f)']
        return age, sex

    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            age, sex = zip(*[get_metadata(ixi_id, ixi_data) for ixi_id in batch["IXI_ID"]])
            sex = torch.FloatTensor(sex, device=args.device)
            age = torch.FloatTensor(age, device=args.device)
            image = batch["image"].to(args.device)
            features = encoder(image).view(features.shape[0], features.shape[1], -1).mean(-1)
            features = torch.cat((features, sex), dim=1)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, age/100)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                age, sex = zip(*[get_metadata(ixi_id, ixi_data) for ixi_id in batch["IXI_ID"]])
                sex = torch.FloatTensor(sex, device=args.device)
                age = torch.FloatTensor(age, device=args.device)
                image = batch["image"].to(args.device)
                features = encoder(image).view(features.shape[0], features.shape[1], -1).mean(-1)
                features = torch.cat((features, sex), dim=1)
                outputs = net(features)
                loss = criterion(outputs, age/100)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Append losses to lists
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save latest model
        torch.save(net.state_dict(), os.path.join(model_dir, 'latest_model.pt'))
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), os.path.join(model_dir, 'best_model.pt'))

        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(model_dir, 'training_curve.png'))
        plt.close()

        # Test the model
        net.load_state_dict(torch.load(os.path.join(model_dir, 'latest_model.pt')))
        net.eval()

def set_up():
    parser = argparse.ArgumentParser(description='Train a brain age regression model')
    parser.add_argument('--logdir', type=str, default='./', help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--name', type=str, default='linear', help='Name of the model')
    parser.add_argument('--net', type=str, default='cnn', help='Network architecture')
    parser.add_argument('--lowres', type=bool, default=False, help='Use low resolution images')
    parser.add_argument('--device', type=str, default=None, help='Device to use for training')
    args = parser.parse_args()

    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create logdir if it doesn't exist
    os.makedirs(args.logdir, exist_ok=True)
    
    return args

if __name__ == '__main__':
    args = set_up()
    run_model(args)
