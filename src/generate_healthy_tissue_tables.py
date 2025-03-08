import os
import pandas as pd
import numpy as np

MODEL_ORDER = ['Baseline', 'Sequence-augmented', 'Sequence-invariant']
SITES = ['GST', 'HH', 'IOP']
MODALITIES = ['T1w', 'T2w', 'PDw']
PERCENTAGES = ['1pc', '10pc', '100pc']
TISSUES = ['white matter', 'gray matter', 'csf']  # These names should match your class names in the results

# Define which metrics should be minimized vs maximized
METRIC_ORDERING = {
    'DSC': 'maximize',
    'HD95': 'minimize'
}

def format_mean_std(mean, std, is_best=False, is_second=False):
    """Format mean ± std with appropriate precision and highlighting"""
    if np.isnan(mean) or np.isnan(std):
        return "---"
    
    if mean < 0.01:
        formatted = f"{mean:.2e} ± {std:.2e}"
    elif mean < 1:
        formatted = f"{mean:.3f} ± {std:.3f}"
    else:
        formatted = f"{mean:.2f} ± {std:.2f}"
    
    if is_best:
        formatted = "\\textbf{" + formatted + "}"
    elif is_second:
        formatted = "\\underline{" + formatted + "}"
    
    return formatted

def get_ranking_indices(values, metric):
    """Get indices of best and second best values, handling NaN"""
    valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
    if not valid_indices:
        return None, None
    
    # Determine sort order based on metric
    reverse = METRIC_ORDERING.get(metric, 'maximize') == 'maximize'
    
    sorted_indices = sorted(valid_indices, key=lambda i: values[i], reverse=reverse)
    best_idx = sorted_indices[0] if len(sorted_indices) > 0 else None
    second_idx = sorted_indices[1] if len(sorted_indices) > 1 else None
    
    return best_idx, second_idx

def create_tissue_table(all_data, tissue, metric, percentage):
    """Create a LaTeX table for a specific tissue type"""
    print(f"\nCreating table for {tissue} tissue - {metric} - {percentage}% Training Data")
    
    # Filter data for the specified tissue
    tissue_data = {}
    for pc in all_data:
        try:
            # Try to find the tissue-specific results
            tissue_data[pc] = all_data[pc].xs(tissue, level='Class', drop_level=False)
        except:
            print(f"No data for {tissue} in {pc}")
            continue
    
    if not tissue_data:
        print(f"No data found for {tissue}")
        return None
    
    # Define shortened model names
    MODEL_NAMES = {
        'Baseline': 'Base',
        'Sequence-augmented': 'Seq-aug',
        'Sequence-invariant': 'Seq-inv'
    }
    
    # Create descriptive caption
    metric_descriptions = {
        'DSC': 'Dice Similarity Coefficient (higher is better)',
        'HD95': '95th percentile Hausdorff Distance in mm (lower is better)'
    }
    
    caption = f"{tissue.title()} segmentation performance using {metric_descriptions.get(metric, metric)} with {percentage}\\% training data. "
    caption += "Values show mean ± standard error, with \\textbf{bold} and \\underline{underlined} indicating best and second-best results. "
    caption += "GST represents the training domain."
    
    # Create tabular format
    tabular_format = "l" + "c" * len(MODEL_ORDER)
    
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:healthy_{tissue.replace(' ', '_')}_{metric.lower()}_{percentage}_results}}",
        "\\resizebox{\\textwidth}{!}{",
        f"\\begin{{tabular}}{{{tabular_format}}}",
        "\\toprule"
    ]

    # Add model names header
    bold_models = [f"\\textbf{{{MODEL_NAMES[model]}}}" for model in MODEL_ORDER]
    latex_lines.append("Dataset & " + " & ".join(bold_models) + " \\\\")
    latex_lines.append("\\midrule")

    # Process datasets
    current_domain = None
    ordered_datasets = []
    for site in SITES:
        for modality in MODALITIES:
            ordered_datasets.append(f"{site} [{modality}]")
    
    pc_data = tissue_data[percentage]
    
    for dataset in ordered_datasets:
        if dataset not in pc_data.index.get_level_values(0).unique():
            continue
            
        print(f"\nDataset: {dataset}")
        
        # Add domain header if domain changes
        domain = "In Domain" if "GST" in dataset else "Out of Domain"
        if domain != current_domain:
            current_domain = domain
            if domain == "Out of Domain":
                latex_lines.append("\\midrule")
            latex_lines.append(f"\\multicolumn{{{len(MODEL_ORDER) + 1}}}{{l}}{{\\textbf{{{domain}}}}} \\\\")
            latex_lines.append("\\midrule")
        
        # Make dataset names bold
        bold_dataset = f"\\textbf{{{dataset}}}"
        
        # Process values for each model
        values = []
        for model in MODEL_ORDER:
            try:
                mean = pc_data.loc[dataset].loc[model][(metric, 'mean')]
                values.append(mean)
            except:
                values.append(np.nan)
        
        best_idx, second_idx = get_ranking_indices(values, metric)
        
        row_values = []
        for i, model in enumerate(MODEL_ORDER):
            try:
                mean = pc_data.loc[dataset].loc[model][(metric, 'mean')]
                std = pc_data.loc[dataset].loc[model][(metric, 'sem')]
                is_best = (i == best_idx)
                is_second = (i == second_idx)
                formatted = format_mean_std(mean, std, is_best, is_second)
                row_values.append(formatted)
            except Exception as e:
                print(f"    Error for {model}: {str(e)}")
                row_values.append("---")
                
        latex_lines.append(f"{bold_dataset} & " + " & ".join(row_values) + " \\\\")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}"
    ])

    return "\n".join(latex_lines)

def main():
    # Healthy segmentation directory
    task_dir = os.path.join('plots', 'healthy_segmentation')
    if not os.path.exists(task_dir):
        print(f"No results directory found for healthy segmentation")
        return

    # Load data for each percentage
    all_data = {}
    for pc in PERCENTAGES:
        # Check for tissue-specific statistics
        for tissue in TISSUES:
            tissue_file = os.path.join(task_dir, pc, f"summary_statistics_{tissue.replace(' ', '_')}.csv")
            if os.path.exists(tissue_file):
                if pc not in all_data:
                    all_data[pc] = {}
                print(f"\nLoading {tissue_file}")
                df = pd.read_csv(tissue_file, header=[0,1], index_col=[0,1])
                print("DataFrame shape:", df.shape)
                print("DataFrame columns:", df.columns)
                all_data[pc] = df
                break  # Found a file with tissue info
        
        # If no tissue-specific files, try the main statistics file
        if pc not in all_data:
            stats_file = os.path.join(task_dir, pc, 'summary_statistics.csv')
            if os.path.exists(stats_file):
                print(f"\nLoading {stats_file}")
                df = pd.read_csv(stats_file, header=[0,1], index_col=[0,1])
                print("DataFrame shape:", df.shape)
                print("DataFrame columns:", df.columns)
                all_data[pc] = df
    
    if not all_data:
        print("No data found!")
        return

    # Check the first dataframe to see what metrics are available
    sample_data = next(iter(all_data.values()))
    metrics = sample_data.columns.get_level_values(0).unique()
    print(f"\nMetrics found: {metrics}")
    
    # Check if we have tissue information in the data
    all_tissues = set()
    for pc, data in all_data.items():
        if 'Class' in data.index.names:
            tissues = data.index.get_level_values('Class').unique()
            all_tissues.update(tissues)
    
    if all_tissues:
        print(f"\nTissue classes found: {all_tissues}")
        TISSUES = list(all_tissues)  # Update with actual tissues found
    
    # Generate tables for each tissue, metric, and percentage
    for tissue in TISSUES:
        tissue_dir = os.path.join(task_dir, tissue.replace(' ', '_'))
        os.makedirs(tissue_dir, exist_ok=True)
        
        for metric in metrics:
            for percentage in PERCENTAGES:
                if percentage in all_data:
                    latex_table = create_tissue_table(all_data, tissue, metric, percentage.replace('pc', ''))
                    if latex_table:
                        output_file = os.path.join(tissue_dir, f'{metric.lower()}_{percentage}_table.txt')
                        with open(output_file, 'w') as f:
                            f.write(latex_table)
                        print(f"Generated table for {tissue} - {metric} - {percentage}")

if __name__ == "__main__":
    main() 