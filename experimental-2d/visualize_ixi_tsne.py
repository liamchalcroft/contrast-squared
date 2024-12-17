import h5py
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import timm
from tqdm import tqdm
import argparse
import os
from torchvision.transforms import Compose, Normalize

def extract_features(model, data_loader, device):
    """Extract features from the model for all data in the loader."""
    model.eval()
    features = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Extracting features"):
            data = data.to(device)
            feature = model(data)
            features.append(feature.cpu().numpy())
    return np.concatenate(features, axis=0)

def create_tsne_plots(h5_path, model_name, weights_path, output_dir, perplexity=30, n_iter=1000, pretrained=False):
    """Create t-SNE plots for IXI dataset slices, colored by site and modality."""
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=1)
    model = torch.compile(model)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device)['model_state_dict'])
    model.to(device)
    
    # Define normalization transform
    transform = Compose([
        Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load data
    print("Loading data...")
    features = []
    site_labels = []
    modality_labels = []
    
    with h5py.File(h5_path, 'r') as f:
        denoising = f['denoising']
        
        # Collect the central slice and its metadata
        for subject in tqdm(denoising.keys(), desc="Loading central slices"):
            for modality in denoising[subject].keys():
                slices = denoising[subject][modality][:]  # Shape: [num_slices, H, W]
                site = denoising[subject][modality].attrs['site']
                
                # Select the central slice
                central_slice = slices[len(slices) // 2]
                central_slice = central_slice[np.newaxis, :, :]  # Add channel dimension
                central_slice_tensor = torch.tensor(central_slice, dtype=torch.float32) / 255.0
                
                # Apply normalization
                central_slice_tensor = transform(central_slice_tensor)
                
                # Extract features
                model.eval()
                with torch.no_grad():
                    feature = model(central_slice_tensor.unsqueeze(0).to(device))
                features.append(feature.cpu().numpy())
                
                # Add metadata for the central slice
                site_labels.append(site)
                modality_labels.append(modality)
    
    # Convert to numpy arrays
    features = np.vstack(features)
    
    # Perform t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings = tsne.fit_transform(features)
    
    # Create plots
    print("Creating plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot by site
    plt.figure(figsize=(10, 10))
    site_colors = {'GST': '#1f77b4', 'HH': '#2ca02c', 'IOP': '#d62728'}
    for site in site_colors:
        mask = np.array(site_labels) == site
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                   c=site_colors[site], label=site, alpha=0.5, s=10)
    plt.title('t-SNE by Site')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ixi_tsne_by_site.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot by modality
    plt.figure(figsize=(10, 10))
    modality_colors = {'t1': '#ff7f0e', 't2': '#9467bd', 'pd': '#8c564b'}
    for modality in modality_colors:
        mask = np.array(modality_labels) == modality
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                   c=modality_colors[modality], label=modality, alpha=0.5, s=10)
    plt.title('t-SNE by Modality')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ixi_tsne_by_modality.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined plot (site and modality)
    plt.figure(figsize=(12, 12))
    for site in site_colors:
        for modality in modality_colors:
            site_mask = np.array(site_labels) == site
            modality_mask = np.array(modality_labels) == modality
            combined_mask = site_mask & modality_mask
            plt.scatter(embeddings[combined_mask, 0], embeddings[combined_mask, 1], 
                       label=f'{site}-{modality}', alpha=0.5, s=10)
    plt.title('t-SNE by Site and Modality')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ixi_tsne_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate t-SNE plots for IXI dataset')
    parser.add_argument('--h5_path', type=str, default='task_data/ixi.h5',
                      help='Path to IXI HDF5 file')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name of the timm model to use')
    parser.add_argument('--weights_path', type=str,
                      help='Path to the model weights file')
    parser.add_argument('--output_dir', type=str, default='task_data',
                      help='Directory to save the t-SNE plots')
    parser.add_argument('--perplexity', type=float, default=30,
                      help='t-SNE perplexity parameter')
    parser.add_argument('--n_iter', type=int, default=1000,
                      help='Number of iterations for t-SNE')
    parser.add_argument('--pretrained', action='store_true',
                      help='Use pretrained model')
    
    args = parser.parse_args()
    create_tsne_plots(args.h5_path, args.model_name, args.weights_path, args.output_dir, args.perplexity, args.n_iter, args.pretrained) 