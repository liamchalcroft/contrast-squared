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
    df['model_name'] = df['model_name'].apply(lambda x: x.replace('timm/', '').replace('.a1_in1k', ''))
    
    # Create mapping for model name replacements
    name_mapping = {
        'random-resnet50': 'Random',
        'imagenet-resnet50': 'ImageNet',
        'mprage-resnet50-view2': 'MPRAGE SimCLR (n=2)',
        'mprage-resnet50-view5': 'MPRAGE SimCLR (n=5)',
        'mprage-resnet50-barlow': 'MPRAGE Barlow Twins',
        'mprage-resnet50-vicreg': 'MPRAGE VICReg',
        'bloch-resnet50-view2': 'Bloch SimCLR (n=2)',
        'bloch-resnet50-view5': 'Bloch SimCLR (n=5)',
        'bloch-resnet50-barlow': 'Bloch Barlow Twins',
        'bloch-resnet50-vicreg': 'Bloch VICReg'
    }
    
    # Apply replacements
    df['model'] = df['model_name'].map(name_mapping)
    
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

def create_boxplots(df, output_dir):
    """Create boxplots for accuracy across models, modalities, and sites."""
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
    
    # Create figure with subplots for each site
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Classification Accuracy by Model and Site', fontsize=16)
    
    for ax, (site, site_label) in enumerate(site_labels.items()):
        site_data = df[df['site'] == site]
        
        # Create boxplot with explicit hue parameter
        boxprops = dict(alpha=0.8)
        sns.boxplot(
            data=site_data,
            x='modality',
            y='test_accuracy',
            hue='model',  # Explicitly specify hue
            ax=axes[ax],
            palette=colors,
            boxprops=boxprops
        )
        
        # Customize plot
        axes[ax].set_title(site_label)
        axes[ax].set_xlabel('Modality')
        axes[ax].set_ylabel('Accuracy (%)')
        
        # Rotate legend labels if needed
        axes[ax].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Only show legend for last subplot
        if ax != 2:  # Changed from axes[-1] comparison
            axes[ax].get_legend().remove()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def create_heatmap(df, output_dir):
    """Create heatmap showing accuracy across modalities and sites for each model."""
    output_dir = Path(output_dir)
    
    # Pivot data for heatmap
    pivot_data = df.pivot_table(
        values='test_accuracy',
        index='model',
        columns=['modality', 'site'],
        aggfunc='mean'
    )
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlBu_r',
        center=50,  # Center colormap at 50%
        vmin=0,
        vmax=100
    )
    
    plt.title('Classification Accuracy (%) Across Modalities and Sites')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_heatmap.png',
                bbox_inches='tight', dpi=300)
    plt.close()

def print_summary_stats(df):
    """Print summary statistics for the results."""
    print("\nSummary Statistics:")
    
    # Overall performance by model
    print("\nOverall Model Performance:")
    summary = df.groupby('model')['test_accuracy'].agg(['mean', 'std']).round(2)
    print(summary)
    
    # Performance by model and modality
    print("\nPerformance by Model and Modality:")
    modality_summary = df.groupby(['model', 'modality'])['test_accuracy'].agg(['mean', 'std']).round(2)
    print(modality_summary)
    
    # Cross-site generalization
    print("\nCross-site Generalization (Training site vs. OOD sites):")
    site_summary = df.groupby(['model', 'site'])['test_accuracy'].mean().round(2)
    site_summary = site_summary.unstack()
    site_summary['OOD_drop'] = site_summary['GST'] - site_summary[['HH', 'IOP']].mean(axis=1)
    print(site_summary)

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
    create_boxplots(df, args.output_dir)
    create_heatmap(df, args.output_dir)
    
    # Print statistics
    print_summary_stats(df) 