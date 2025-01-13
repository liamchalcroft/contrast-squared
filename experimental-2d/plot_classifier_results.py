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

    print(df.head())

    print(df['model'].unique())
    
    # # Clean up model names for better display
    # df['model'] = df['model'].apply(lambda x: x.replace('timm/', '').replace('.a1_in1k', ''))
    
    # # Create mapping for model name replacements
    # name_mapping = {
    #     'random-resnet50': 'Random',
    #     'imagenet-resnet50': 'ImageNet',
    #     'mprage-resnet50-view2': 'MPRAGE SimCLR (n=2)',
    #     'mprage-resnet50-view5': 'MPRAGE SimCLR (n=5)',
    #     'mprage-resnet50-barlow': 'MPRAGE Barlow Twins',
    #     'mprage-resnet50-vicreg': 'MPRAGE VICReg',
    #     'bloch-resnet50-view2': 'Bloch SimCLR (n=2)',
    #     'bloch-resnet50-view5': 'Bloch SimCLR (n=5)',
    #     'bloch-resnet50-barlow': 'Bloch Barlow Twins',
    #     'bloch-resnet50-vicreg': 'Bloch VICReg'
    # }
    
    # # Apply replacements
    # df['model'] = df['model'].map(name_mapping)

    df['model'] = df['model'].apply(lambda x: x.replace(' ResNet-50', ''))
    df['model'] = df['model'].apply(lambda x: x.replace('View2', 'SimCLR (n=2)'))
    df['model'] = df['model'].apply(lambda x: x.replace('View5', 'SimCLR (n=5)'))
    df['model'] = df['model'].apply(lambda x: x.replace('Barlow', 'Barlow Twins'))

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

def create_barplots(df, output_dir, metrics=None):
    """Create bar plots for specified metrics."""
    if metrics is None:
        metrics = [
            ('Accuracy', 'test_accuracy'),
            ('Loss', 'test_loss')
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
            
            # Create bar plot with custom colors, no aggregation
            sns.barplot(
                data=site_data,
                x='modality',
                y=metric_col,
                hue='model',
                ax=ax,
                palette=colors,
                errorbar=None  # No confidence intervals
            )
            
            # Set log scale for Loss
            if metric_name == 'Loss':
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

def print_summary_stats(df):
    """Print summary statistics for the results."""
    print("\nSummary Statistics:")
    summary = df.groupby(['model', 'modality']).agg({
        'test_accuracy': ['mean', 'std'],
        'test_loss': ['mean', 'std']
    }).round(3)
    
    print(summary)
    
    # Print cross-site generalization metrics
    print("\nCross-site Generalization:")
    site_perf = df.groupby(['model', 'site'])['test_accuracy'].mean().unstack()
    site_perf['OOD_drop'] = site_perf['GST'] - site_perf[['HH', 'IOP']].mean(axis=1)
    print(site_perf.round(3))

def create_ood_barplots(df, output_dir):
    """Create bar plots showing OOD performance degradation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_theme()
    
    # Get color palette
    colors = get_model_colors()
    
    # Calculate OOD degradation
    site_perf = df.groupby(['model', 'modality', 'site'])['test_accuracy'].mean().unstack()
    ood_drop = pd.DataFrame({
        'model': site_perf.index.get_level_values(0),
        'modality': site_perf.index.get_level_values(1),
        'HH_drop': site_perf['GST'] - site_perf['HH'],
        'IOP_drop': site_perf['GST'] - site_perf['IOP'],
        'avg_drop': site_perf['GST'] - site_perf[['HH', 'IOP']].mean(axis=1)
    }).reset_index(drop=True)
    
    # Create figure with subplots for each type of drop
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Out-of-Distribution Performance Degradation', fontsize=16)
    
    drop_types = [
        ('HH_drop', 'HH Site Drop'),
        ('IOP_drop', 'IOP Site Drop'),
        ('avg_drop', 'Average OOD Drop')
    ]
    
    for ax, (drop_col, title) in zip(axes, drop_types):
        sns.barplot(
            data=ood_drop,
            x='modality',
            y=drop_col,
            hue='model',
            ax=ax,
            palette=colors,
            errorbar=None  # No confidence intervals
        )
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel('Modality')
        ax.set_ylabel('Accuracy Drop')
        
        # Rotate legend labels if needed
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Only show legend for last subplot
        if ax != axes[-1]:
            ax.get_legend().remove()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / 'ood_degradation.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate plots from classification results')
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Directory containing CSV result files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Load and process results
    df = load_and_process_results(args.results_dir)
    
    # Create plots
    create_barplots(df, args.output_dir)
    create_ood_barplots(df, args.output_dir)
    
    # Print statistics
    print_summary_stats(df) 