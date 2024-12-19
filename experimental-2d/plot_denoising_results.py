import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_and_process_results(results_dir):
    """Load and combine all CSV files from results directory."""
    results_dir = Path(results_dir)
    all_files = list(results_dir.glob('*.csv'))
    
    if not all_files:
        raise ValueError(f"No CSV files found in {results_dir}")
    
    # Combine all CSV files
    df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    
    # Clean up model names for better display
    # df['model'] = df['model'].apply(lambda x: x.replace('ResNet-50', '').strip())
    
    return df

def create_boxplots(df, output_dir, metrics=None):
    """Create boxplots for specified metrics."""
    if metrics is None:
        metrics = [
            ('PSNR', ['noisy_psnr', 'denoised_psnr']),
            ('SSIM', ['noisy_ssim', 'denoised_ssim'])
        ]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create plots for each metric
    for metric_name, metric_cols in metrics:
        # Create figure with subplots for each modality
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'{metric_name} by Model and Modality', fontsize=16)
        
        for ax, modality in zip(axes, ['t1', 't2', 'pd']):
            modality_data = df[df['modality'] == modality]
            
            # Reshape data for boxplot
            plot_data = pd.melt(
                modality_data,
                id_vars=['model'],
                value_vars=metric_cols,
                var_name='Type',
                value_name=metric_name
            )
            
            # Create boxplot
            sns.boxplot(
                data=plot_data,
                x='model',
                y=metric_name,
                hue='Type',
                ax=ax
            )
            
            # Customize plot
            ax.set_title(f'{modality.upper()} Images')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_xlabel('')
            
            # Only show legend for first subplot
            if ax != axes[0]:
                ax.get_legend().remove()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric_name.lower()}_comparison.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

def create_site_comparison(df, output_dir):
    """Create plots comparing performance across sites."""
    output_dir = Path(output_dir)
    
    # Create violin plots for site comparison
    plt.figure(figsize=(15, 6))
    sns.violinplot(
        data=df,
        x='model',
        y='denoised_psnr',
        hue='site',
        split=True
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Denoising Performance Across Sites')
    plt.tight_layout()
    plt.savefig(output_dir / 'site_comparison.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def print_summary_stats(df):
    """Print summary statistics for the results."""
    print("\nSummary Statistics:")
    summary = df.groupby(['model', 'modality']).agg({
        'noisy_psnr': ['mean', 'std'],
        'denoised_psnr': ['mean', 'std'],
        'noisy_ssim': ['mean', 'std'],
        'denoised_ssim': ['mean', 'std']
    }).round(3)
    
    print(summary)
    
    # Calculate and print improvement metrics
    improvements = df.groupby(['model', 'modality']).apply(
        lambda x: pd.Series({
            'PSNR_improvement': (x['denoised_psnr'] - x['noisy_psnr']).mean(),
            'SSIM_improvement': (x['denoised_ssim'] - x['noisy_ssim']).mean()
        })
    ).round(3)
    
    print("\nAverage Improvements:")
    print(improvements)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate plots from denoising results')
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Directory containing CSV result files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Load and process results
    df = load_and_process_results(args.results_dir)
    
    # Create plots
    create_boxplots(df, args.output_dir)
    create_site_comparison(df, args.output_dir)
    
    # Print statistics
    print_summary_stats(df) 