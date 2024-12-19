import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from downstream_preprocess import get_test_loader
from models import create_unet_model
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_metrics(clean, noisy, denoised):
    """Calculate PSNR and SSIM for both noisy and denoised images."""
    # Convert to numpy and squeeze if needed
    clean = clean.cpu().numpy().squeeze()
    noisy = noisy.cpu().numpy().squeeze()
    denoised = denoised.cpu().numpy().squeeze()
    
    # Rescale to [0, 1] for metric calculation
    clean = (clean - clean.min()) / (clean.max() - clean.min())
    noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())
    denoised = (denoised - denoised.min()) / (denoised.max() - denoised.min())
    
    # Calculate metrics
    noisy_psnr = peak_signal_noise_ratio(clean, noisy, data_range=1.0)
    denoised_psnr = peak_signal_noise_ratio(clean, denoised, data_range=1.0)
    
    noisy_ssim = structural_similarity(clean, noisy, data_range=1.0)
    denoised_ssim = structural_similarity(clean, denoised, data_range=1.0)

    mae = torch.mean(torch.abs(clean - denoised))
    mse = torch.mean((clean - denoised) ** 2)
    
    return {
        'noisy_psnr': noisy_psnr,
        'denoised_psnr': denoised_psnr,
        'noisy_ssim': noisy_ssim,
        'denoised_ssim': denoised_ssim,
        'mae': mae,
        'mse': mse
    }

def test_denoising(model_dir, model_name, modality, site):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_unet_model(model_name, pretrained=False)
    checkpoint = torch.load(Path(model_dir) / f"denoising_model_{modality}_GST_latest.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get test loader
    test_loader = get_test_loader(
        batch_size=1,  # Process one image at a time for per-item metrics
        task='denoising',
        modality=modality,
        site=site
    )

    # Initialize results list
    results = []

    # Test loop
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Testing {modality} {site}")):
            clean = batch['image'].to(device)
            
            # Add noise
            std = torch.rand(1, dtype=clean.dtype, device=clean.device) * 0.2
            noise = torch.randn_like(clean) * std
            noisy = clean + noise
            
            # Estimate noise
            noise_pred = model(noisy)

            # Estimate denoised image
            denoised = noisy - noise_pred
            
            # Calculate metrics
            metrics = calculate_metrics(clean[0], noisy[0], denoised[0])
            
            # Store results
            results.append({
                'model': model_name,
                'modality': modality,
                'site': site,
                'image_idx': i,
                **metrics
            })

    return pd.DataFrame(results)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test denoising models and generate metrics')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing model checkpoints')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model for logging')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output CSV file')
    parser.add_argument('--modality', type=str, nargs='+', default=['t1', 't2', 'pd'], help='Modalities to test')
    parser.add_argument('--sites', type=str, nargs='+', default=['GST', 'HH', 'IOP'], help='Sites to test')
    
    args = parser.parse_args()
    
    # Initialize empty DataFrame for all results
    all_results = pd.DataFrame()
    
    # Test each modality and site combination
    for modality in args.modality:
        for site in args.sites:
            try:
              results = test_denoising(
                  args.model_dir,
                  args.model_name,
                  modality,
                  site
              )
              all_results = pd.concat([all_results, results], ignore_index=True)
            except Exception as e:
                print(f"Error testing {modality} {site}: {e}")
    
    # Save results
    all_results.to_csv(args.output_file, index=False)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    summary = all_results.groupby(['model', 'modality', 'site']).agg({
        'noisy_psnr': ['mean', 'std'],
        'denoised_psnr': ['mean', 'std'],
        'noisy_ssim': ['mean', 'std'],
        'denoised_ssim': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'mse': ['mean', 'std']
    }).round(3)
    print(summary) 