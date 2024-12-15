import matplotlib.pyplot as plt
import torch
from preprocess import get_bloch_loader, get_mprage_loader
import os
from torchvision.utils import make_grid
import numpy as np

def save_batch_images(batch, save_path, title):
    """
    Save a batch of images to disk.
    batch: dict with 'image1', 'image2', etc. keys containing tensors of shape [B, 1, H, W]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get number of views
    num_views = len([k for k in batch.keys() if k.startswith('image')])
    
    # Create figure with subplots for each sample in batch
    batch_size = next(iter(batch.values())).shape[0]
    fig, axes = plt.subplots(batch_size, num_views, figsize=(3*num_views, 3*batch_size))
    
    if batch_size == 1:
        axes = axes[None, :]
    
    # Plot each sample
    for i in range(batch_size):
        for j in range(num_views):
            img = batch[f'image{j+1}'][i, 0].numpy()  # Remove channel dim
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f'View {j+1}')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_loaders():
    # Test Bloch loader with same contrast
    loader = get_bloch_loader(batch_size=4, same_contrast=True, num_views=2, cache_data=False)
    batch = next(iter(loader))
    save_batch_images(
        batch, 
        'output/test/bloch_same_contrast.png',
        'Bloch Loader - Same Contrast'
    )
    
    # Test Bloch loader with different contrasts
    loader = get_bloch_loader(batch_size=4, same_contrast=False, num_views=2, cache_data=False)
    batch = next(iter(loader))
    save_batch_images(
        batch, 
        'output/test/bloch_diff_contrast.png',
        'Bloch Loader - Different Contrasts'
    )
    
    # Test Bloch loader with 5 views
    loader = get_bloch_loader(batch_size=4, same_contrast=False, num_views=5, cache_data=False)
    batch = next(iter(loader))
    save_batch_images(
        batch, 
        'output/test/bloch_five_views.png',
        'Bloch Loader - Five Views'
    )
    
    # Test MPRAGE loader
    loader = get_mprage_loader(batch_size=4, num_views=2, cache_data=False)
    batch = next(iter(loader))
    save_batch_images(
        batch, 
        'output/test/mprage.png',
        'MPRAGE Loader'
    )

if __name__ == "__main__":
    # Create output directory
    os.makedirs("output/test", exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    test_loaders()
    print("Done! Check output/test/ directory for results.") 