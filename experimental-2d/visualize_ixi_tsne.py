import h5py
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import timm
from tqdm import tqdm
import argparse
import os
from torchvision.transforms import Compose, Normalize

# Ensure the use of a consistent style
sns.set(style="whitegrid")

def strip_prefix_state_dict(state_dict, prefix_to_remove):
    """Load weights and remove a specific prefix from the keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace(prefix_to_remove, '', 1)  # Remove the prefix
        new_state_dict[new_key] = v
    return new_state_dict

def create_tsne_plots(h5_path, model_name, weights_path, output_dir, perplexity=30, n_iter=1000, pretrained=False, name=''):
    """Create t-SNE plots for IXI dataset slices, colored by site and modality."""
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=1)
    if weights_path:
        model.load_state_dict(strip_prefix_state_dict(torch.load(weights_path, map_location=device)['model_state_dict'], '_orig_mod.encoder.'), strict=False)
    model.to(device)
    model = torch.compile(model)
    
    # Define normalization transform
    transform = Compose([
        Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load or compute features
    features_path = os.path.join(output_dir, 'features.npy')
    embeddings_path = os.path.join(output_dir, 'embeddings.npy')
    if os.path.exists(embeddings_path):
        print("Loading precomputed t-SNE embeddings...")
        embeddings = np.load(embeddings_path)
        with open(os.path.join(output_dir, 'metadata.npy'), 'rb') as f:
            site_labels = np.load(f)
            modality_labels = np.load(f)
    else:
        if os.path.exists(features_path):
            print("Loading precomputed features...")
            features = np.load(features_path)
            with open(os.path.join(output_dir, 'metadata.npy'), 'rb') as f:
                site_labels = np.load(f)
                modality_labels = np.load(f)
        else:
            print("Loading data and extracting features...")
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
                        with torch.no_grad():
                            feature = model(central_slice_tensor.unsqueeze(0).to(device)).cpu().numpy()
                        features.append(feature)
                        
                        # Add metadata for the central slice
                        site_labels.append(site)
                        modality_labels.append(modality)
            
            # Convert to numpy arrays
            features = np.vstack(features)
            
            # Save features and metadata
            np.save(features_path, features)
            with open(os.path.join(output_dir, 'metadata.npy'), 'wb') as f:
                np.save(f, site_labels)
                np.save(f, modality_labels)
        
        # Perform t-SNE
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        embeddings = tsne.fit_transform(features)
        
        # Save t-SNE embeddings
        np.save(embeddings_path, embeddings)
    
    # Create plots
    print("Creating plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define consistent color and marker orders
    modality_order = ['t1', 't2', 'pd']
    site_order = ['GST', 'HH', 'IOP']

    # Define colors and markers
    modality_colors = {'t1': '#1f77b4', 't2': '#ff7f0e', 'pd': '#2ca02c'}
    site_markers = {'GST': 'o', 'HH': 'x', 'IOP': 's'}

    # Mapping for legend labels
    modality_labels_map = {'t1': 'T1w', 't2': 'T2w', 'pd': 'PDw'}

    # # Define fixed axis limits
    # xlim = (-2.5, 2.5)  # Example values, adjust based on your data
    # ylim = (-2.5, 2.5)  # Example values, adjust based on your data

    # Create combined plot (site and modality)
    plt.figure(figsize=(12, 12))
    for site in site_order:
        for modality in modality_order:
            site_mask = np.array(site_labels) == site
            modality_mask = np.array(modality_labels) == modality
            combined_mask = site_mask & modality_mask
            plt.scatter(embeddings[combined_mask, 0], embeddings[combined_mask, 1], 
                       c=[modality_colors[modality]], marker=site_markers[site], alpha=0.7, s=50, edgecolor='w', linewidth=0.5)

    # Set fixed axis limits
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Create custom legend for modalities (colors)
    modality_legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=modality_labels_map[modality],
                                          markerfacecolor=modality_colors[modality], markersize=12) 
                               for modality in modality_order]
    modality_legend = plt.legend(handles=modality_legend_handles, title="Modality", loc='upper left', fontsize=14, title_fontsize=16, framealpha=0.5)

    # Create custom legend for sites (shapes)
    site_legend_handles = [plt.Line2D([0], [0], marker=site_markers[site], color='w', label=site,
                                      markerfacecolor='gray', markersize=12, markeredgecolor='k') 
                             for site in site_order]
    site_legend = plt.legend(handles=site_legend_handles, title="Site", loc='upper right', fontsize=14, title_fontsize=16, framealpha=0.5)

    # Add both legends to the plot
    plt.gca().add_artist(modality_legend)
    plt.gca().add_artist(site_legend)

    plt.axis('off')  # Remove axes
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'tsne_{name}.png'), dpi=600, bbox_inches='tight')
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
    parser.add_argument('--name', type=str, default='',
                      help='Name of the model')
    
    args = parser.parse_args()
    create_tsne_plots(args.h5_path, args.model_name, args.weights_path, args.output_dir, args.perplexity, args.n_iter, args.pretrained, args.name) 