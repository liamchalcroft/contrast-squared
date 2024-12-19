import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np

def load_and_process_results(results_dir):
    """Load and combine all CSV files from results directory."""
    results_dir = Path(results_dir)
    all_files = list(results_dir.glob('*.csv'))
    
    if not all_files:
        raise ValueError(f"No CSV files found in {results_dir}")
    
    # Combine all CSV files
    df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    
    # Clean up model names for better display
    df['model'] = df['model'].apply(lambda x: x.replace('ResNet-50', '').strip())
    
    # Create mapping for model name replacements
    name_mapping = {
        'View2': 'SimCLR (n=2)',
        'View5': 'SimCLR (n=5)',
        'Barlow': 'Barlow Twins',
        'VICReg': 'VICReg'
    }
    
    # Apply replacements
    for old, new in name_mapping.items():
        df['model'] = df['model'].apply(lambda x: x.replace(old, new))
    
    # Define consistent model order
    model_order = [
        'Random',
        'ImageNet',
        'MPRAGE SimCLR (n=2)',
        'MPRAGE SimCLR (n=5)',
        'MPRAGE Barlow Twins',
        'MPRAGE VICReg',
        'Bloch SimCLR (n=2)',
        'Bloch SimCLR (n=5)',
        'Bloch Barlow Twins',
        'Bloch VICReg'
    ]
    
    # Convert model column to categorical with specific order
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
    
    # Define modality mapping and order
    modality_mapping = {
        't1': 'T1w',
        't2': 'T2w',
        'pd': 'PDw'
    }
    modality_order = ['t1', 't2', 'pd']
    
    # Apply modality mapping
    df['modality'] = df['modality'].map(modality_mapping)
    
    # Convert modality to categorical with specific order
    df['modality'] = pd.Categorical(
        df['modality'], 
        categories=[modality_mapping[m] for m in modality_order], 
        ordered=True
    )
    
    # Sort the dataframe by model and modality order
    df = df.sort_values(['model', 'modality'])
    
    return df

def get_model_colors():
    """Define consistent colors for model groups."""
    # Use different base colors for each group
    colors = {
        'Random': '#ffffff',  # White
        'ImageNet': '#808080',  # Gray
        # Blues for MPRAGE with different shades
        'MPRAGE SimCLR (n=2)': '#1f77b4',
        'MPRAGE SimCLR (n=5)': '#2f87c4',
        'MPRAGE Barlow Twins': '#3f97d4',
        'MPRAGE VICReg': '#4fa7e4',
        # Oranges for Bloch with different shades
        'Bloch SimCLR (n=2)': '#ff7f0e',
        'Bloch SimCLR (n=5)': '#ff8f1e',
        'Bloch Barlow Twins': '#ff9f2e',
        'Bloch VICReg': '#ffaf3e'
    }
    return colors

def create_boxplots(df, output_dir, metrics=None):
    """Create boxplots for specified metrics."""
    if metrics is None:
        metrics = [
            ('PSNR', 'denoised_psnr'),
            ('SSIM', 'denoised_ssim'),
            ('MSE', 'mse'),
            ('MAE', 'mae')
        ]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_theme()
    
    # Get color palette
    colors = get_model_colors()
    
    # Site labels mapping
    site_labels = {
        'GST': 'Training Site (GST)',
        'HH': 'OOD Site (HH)',
        'IOP': 'OOD Site (IOP)'
    }
    
    # Create plots for each metric
    for metric_name, metric_col in metrics:
        # Create figure with subplots for each site
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'{metric_name} by Model and Site', fontsize=16)
        
        for ax, site in zip(axes, ['GST', 'HH', 'IOP']):
            site_data = df[df['site'] == site]
            
            # Create boxplot with custom colors
            sns.boxplot(
                data=site_data,
                x='modality',
                y=metric_col,
                hue='model',
                ax=ax,
                palette=colors
            )
            
            # Set log scale for MSE and MAE
            if metric_name in ['MSE', 'MAE']:
                ax.set_yscale('log')
            
            # Customize plot
            ax.set_title(site_labels[site])
            ax.set_xlabel('Modality')
            ax.set_ylabel(metric_name)
            
            # Rotate legend labels if needed
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Only show legend for last subplot
            if ax != axes[-1]:
                ax.get_legend().remove()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric_name.lower()}_comparison.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

def create_radar_plots(df, output_dir):
    """Create radar plots comparing models across metrics."""
    output_dir = Path(output_dir)
    
    # Get color palette
    colors = get_model_colors()
    
    # Prepare metrics for radar plot
    metrics = ['denoised_psnr', 'denoised_ssim', 'mse', 'mae']
    metric_names = ['PSNR', 'SSIM', 'MSE', 'MAE']
    
    # For each modality and site
    for modality in ['t1', 't2', 'pd']:
        for site in ['GST', 'HH', 'IOP']:
            # Filter data
            plot_data = df[(df['modality'] == modality) & (df['site'] == site)]
            
            # Calculate mean values for each model and metric
            means = plot_data.groupby('model')[metrics].mean()
            
            # Normalize metrics to [0,1] scale for comparison
            # Note: Invert MSE and MAE since lower is better
            normalized = pd.DataFrame()
            for metric in metrics:
                if metric in ['mse', 'mae']:
                    normalized[metric] = 1 - ((means[metric] - means[metric].min()) / 
                                            (means[metric].max() - means[metric].min()))
                else:
                    normalized[metric] = (means[metric] - means[metric].min()) / \
                                       (means[metric].max() - means[metric].min())
            
            # Calculate min and max values for smart limits
            min_val = normalized.values.min()
            max_val = normalized.values.max()
            
            # Set limits to 95% of min and 105% of max
            ylim_min = min_val * 0.95
            ylim_max = max_val * 1.05
            
            # Set up the angles for the spider plot
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # Close the plot
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Plot each model
            for model in normalized.index:
                values = normalized.loc[model].values
                values = np.concatenate((values, [values[0]]))  # Close the plot
                ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[model])
                ax.fill(angles, values, alpha=0.25, color=colors[model])
            
            # Fix axis to go in the right order and start at 12 o'clock
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Set the ylim
            ax.set_ylim(ylim_min, ylim_max)
            
            # Draw axis lines for each angle and label
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_names, rotation=45)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
            
            # Add title
            plt.title(f"Performance Metrics - {modality.upper()} {site}")
            
            # Save plot
            plt.tight_layout()
            plt.savefig(output_dir / f'radar_plot_{modality}_{site}.png',
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
    # create_radar_plots(df, args.output_dir)
    
    # Print statistics
    print_summary_stats(df) 