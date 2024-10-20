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
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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
    # Load IXI spreadsheet
    ixi_data = pd.read_excel('/home/lchalcroft/Data/IXI/IXI.xls')
    print(ixi_data.head())

    # Load and prepare data
    features_dir = os.path.join(args.logdir, args.name, "ixi-features/guys/t1")
    all_files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
    all_ids = sorted([int(f.split('.')[0][3:]) for f in all_files])

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
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')

    # Create dir for model
    model_dir = os.path.join(args.logdir, args.name, 'ixi-classifier')
    os.makedirs(model_dir, exist_ok=True)

    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []

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
                loss = criterion(outputs, ages/100)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Append losses to lists
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))

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
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pt')))
    model.eval()

    # Test on multiple modalities and sites
    def test_modality(features_dir, ixi_data, model, criterion, exclude_ids):
        all_files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
        all_ids = sorted([int(f.split('.')[0][3:]) for f in all_files])
        valid_ids = [id for id in all_ids if id in ixi_data['IXI_ID'].values and id not in exclude_ids]
        
        if not valid_ids:
            return None  # Return None if no valid IDs are found
        
        test_dataset = IXIDataset(features_dir, ixi_data, valid_ids)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        all_outputs = []
        all_ages = []
        all_ids = []
        
        with torch.no_grad():
            for features, ages in test_loader:
                outputs = model(features)
                all_outputs.extend(outputs.cpu().numpy() * 100)  # Scale back to original age
                all_ages.extend(ages.cpu().numpy())
                all_ids.extend([test_dataset.ids[i] for i in range(len(ages))])
        
        mse = mean_squared_error(all_ages, all_outputs)
        mae = mean_absolute_error(all_ages, all_outputs)
        
        return all_ids, all_outputs, all_ages, mse, mae

    # Combine train and validation IDs to exclude from testing
    exclude_ids = set(train_ids + val_ids)

    # Prepare CSV file
    csv_file = os.path.join(model_dir, 'test_results.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Site', 'Modality', 'Predicted Age', 'True Age', 'MSE', 'MAE'])

        # Test on all sites and modalities
        for site, modalities in [("Guy's Hospital", ['t1', 't2', 'pd', 'mra']),
                                 ("Hammersmith Hospital", ['t1', 't2', 'pd']),
                                 ("Institute of Psychiatry", ['t1', 't2', 'pd'])]:
            site_short = site.split()[0].lower()
            for modality in modalities:
                features_dir = os.path.join(args.logdir, args.name, f"ixi-features/{site_short}/{modality}")
                result = test_modality(features_dir, ixi_data, model, criterion, exclude_ids)
                
                if result is not None:
                    ids, predicted_ages, true_ages, mse, mae = result
                    for id, pred_age, true_age in zip(ids, predicted_ages, true_ages):
                        writer.writerow([id, site, modality, pred_age[0], true_age[0], mse, mae])
                    
                    print(f"{site} - {modality}: MSE = {mse:.4f}, MAE = {mae:.4f}")

    print(f"Results saved to {csv_file}")

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
