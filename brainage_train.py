import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load IXI spreadsheet
ixi_data = pd.read_excel('/home/lchalcroft/Data/IXI/IXI.xls')
print(ixi_data.head())

# Prepare data
class IXIDataset(Dataset):
    def __init__(self, features_dir, ixi_data, ids):
        self.features_dir = features_dir
        self.ixi_data = ixi_data
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ixi_id = self.ids[idx]
        file_name = f"IXI{ixi_id:03d}.npy"
        features = np.load(os.path.join(self.features_dir, file_name))
        
        subject_data = self.ixi_data[self.ixi_data['IXI_ID'] == ixi_id].iloc[0]
        age = subject_data['AGE']
        assert not pd.isna(age), f"Age is NaN for IXI_ID {ixi_id}"
        sex = subject_data['SEX_ID (1=m, 2=f)']
        
        # Append sex to features
        features = np.append(features, sex)
        
        return torch.FloatTensor(features), torch.FloatTensor([age])

# Define the model
class BrainAgeRegressor(nn.Module):
    def __init__(self, input_size):
        super(BrainAgeRegressor, self).__init__()
        self.linear = nn.Linear(input_size, 1, bias=True)
    
    def forward(self, x):
        return self.linear(x)

def run_model(args):
    # Load and prepare data
    features_dir = os.path.join(args.logdir, args.name, "ixi-features/guys/t1")
    all_files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
    all_ids = sorted([int(f.split('.')[0][3:]) for f in all_files])

    # Ensure all IDs in all_ids are present in ixi_data
    valid_ids = [id for id in all_ids if id in ixi_data['IXI_ID'].values]
    for id in valid_ids:
        print(ixi_data[ixi_data['IXI_ID'] == id]['AGE'].item())
    valid_ids = [id for id in valid_ids if not pd.isna(ixi_data[ixi_data['IXI_ID'] == id]['AGE'].item())]

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
    train_dataset = IXIDataset(features_dir, ixi_data, train_ids)
    val_dataset = IXIDataset(features_dir, ixi_data, val_ids)
    test_dataset = IXIDataset(features_dir, ixi_data, test_ids)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize model, loss function, and optimizer
    input_size = train_dataset[0][0].shape[0]  # Features + sex
    model = BrainAgeRegressor(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')

    # Create dir for model
    os.makedirs(os.path.join(args.logdir, args.name, 'ixi-classifier'), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features, ages in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, ages/100)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, ages in val_loader:
                outputs = model(features)
                loss = criterion(outputs, ages)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.logdir, args.name, 'ixi-classifier', 'best_model.pt'))

    # Test the model
    model.load_state_dict(torch.load(os.path.join(args.logdir, args.name, 'ixi-classifier', 'best_model.pt')))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for features, ages in test_loader:
            outputs = model(features)
            loss = criterion(outputs, ages)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss/len(test_loader):.4f}")

def set_up():
    parser = argparse.ArgumentParser(description='Train a brain age regression model')
    parser.add_argument('--logdir', type=str, default='./', help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--name', type=str, default='linear', help='Name of the model')
    
    args = parser.parse_args()
    
    # Create logdir if it doesn't exist
    os.makedirs(args.logdir, exist_ok=True)
    
    return args

if __name__ == '__main__':
    args = set_up()
    run_model(args)
