import os
import pandas as pd
import numpy as np

MODEL_ORDER = ['Baseline', 'Sequence-augmented', 'Sequence-invariant']
SITES = ['GST', 'HH', 'IOP']
MODALITIES = ['T1w', 'T2w', 'PDw']
PERCENTAGES = ['1pc', '10pc', '100pc']
# These default tissue types will be used if we can't detect them from the data
DEFAULT_TISSUES = ['white matter', 'gray matter', 'csf']

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

def create_combined_tissue_table(all_data, tissue, metric):
    """Create a LaTeX table for a specific tissue type combining all percentages"""
    print(f"\nCreating combined table for {tissue} tissue - {metric}")
    
    # Filter data for the specified tissue
    tissue_data = {}
    for pc in PERCENTAGES:
        if pc not in all_data:
            print(f"No data for percentage {pc}")
            continue
            
        try:
            # Try to find the tissue-specific results
            if 'Class' in all_data[pc].index.names:
                tissue_data[pc] = all_data[pc].xs(tissue, level='Class', drop_level=False)
            else:
                # If there's no Class level, this might be a tissue-specific file
                tissue_data[pc] = all_data[pc]
        except Exception as e:
            print(f"No data for {tissue} in {pc}: {e}")
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
    
    caption = f"{tissue.title()} segmentation performance using {metric_descriptions.get(metric, metric)} across different training data percentages. "
    caption += "Values show mean ± standard error, with \\textbf{bold} and \\underline{underlined} indicating best and second-best results for each dataset and percentage. "
    caption += "GST represents the training domain."
    
    # Calculate total columns (3 models per percentage + 1 for dataset)
    total_cols = len(PERCENTAGES) * len(MODEL_ORDER) + 1
    
    # Create tabular format with vertical lines between percentage groups
    tabular_format = "l" + ("c" * len(MODEL_ORDER) + "|") * (len(PERCENTAGES) - 1) + "c" * len(MODEL_ORDER)
    
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:healthy_{tissue.replace(' ', '_')}_{metric.lower()}_combined_results}}",
        "\\resizebox{\\textwidth}{!}{",
        f"\\begin{{tabular}}{{{tabular_format}}}",
        "\\toprule"
    ]

    # Create more descriptive percentage headers with bold
    header = ["& \\multicolumn{" + str(len(MODEL_ORDER)) + "}{c}{\\textbf{" + str(int(pc.replace('pc',''))) + "\\% Training Data}}" 
             for pc in PERCENTAGES]
    latex_lines.append(" " + " ".join(header) + " \\\\")
    
    # Add bold to shortened model names
    bold_models = [f"\\textbf{{{MODEL_NAMES[model]}}}" for model in MODEL_ORDER]
    model_line = "& " + " & ".join(bold_models * len(PERCENTAGES)) + " \\\\"
    latex_lines.extend([
        f"\\cmidrule(lr){{2-{total_cols}}}",
        model_line,
        "\\midrule"
    ])

    # Process datasets
    current_domain = None
    ordered_datasets = []
    for site in SITES:
        for modality in MODALITIES:
            ordered_datasets.append(f"{site} [{modality}]")
    
    for dataset in ordered_datasets:
        # Check if this dataset exists in any of the percentage data
        dataset_exists = False
        for pc in PERCENTAGES:
            if pc in tissue_data and dataset in tissue_data[pc].index.get_level_values(0).unique():
                dataset_exists = True
                break
                
        if not dataset_exists:
            continue
            
        print(f"\nDataset: {dataset}")
        
        # Add domain header if domain changes
        domain = "In Domain" if "GST" in dataset else "Out of Domain"
        if domain != current_domain:
            current_domain = domain
            if domain == "Out of Domain":
                latex_lines.append("\\midrule")
            latex_lines.append(f"\\multicolumn{{{total_cols}}}{{l}}{{\\textbf{{{domain}}}}} \\\\")
            latex_lines.append("\\midrule")
        
        # Make dataset names bold
        bold_dataset = f"\\textbf{{{dataset}}}"
        
        row_values = []
        
        # Process each percentage
        for pc in PERCENTAGES:
            if pc not in tissue_data or dataset not in tissue_data[pc].index.get_level_values(0).unique():
                # If this dataset doesn't exist for this percentage, add empty cells
                row_values.extend(["---"] * len(MODEL_ORDER))
                continue
                
            pc_data = tissue_data[pc]
            
            # Get values for ranking
            values = []
            for model in MODEL_ORDER:
                try:
                    mean = pc_data.loc[dataset].loc[model][(metric, 'mean')]
                    values.append(mean)
                except:
                    values.append(np.nan)
            
            best_idx, second_idx = get_ranking_indices(values, metric)
            
            # Format values for this percentage
            for i, model in enumerate(MODEL_ORDER):
                try:
                    mean = pc_data.loc[dataset].loc[model][(metric, 'mean')]
                    std = pc_data.loc[dataset].loc[model][(metric, 'sem')]
                    is_best = (i == best_idx)
                    is_second = (i == second_idx)
                    formatted = format_mean_std(mean, std, is_best, is_second)
                    row_values.append(formatted)
                except Exception as e:
                    print(f"    Error for {model} in {pc}: {str(e)}")
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
    tissues_in_data = set()
    
    for pc in PERCENTAGES:
        # First try loading tissue-specific files
        tissue_files_found = False
        for tissue in DEFAULT_TISSUES:
            tissue_file = os.path.join(task_dir, pc, f"summary_statistics_{tissue.replace(' ', '_')}.csv")
            if os.path.exists(tissue_file):
                print(f"\nLoading tissue-specific file: {tissue_file}")
                df = pd.read_csv(tissue_file, header=[0,1], index_col=[0,1])
                print("DataFrame shape:", df.shape)
                print("DataFrame columns:", df.columns)
                all_data[pc] = df
                tissues_in_data.add(tissue)
                tissue_files_found = True
                break  # Found tissue-specific file
        
        # If no tissue-specific files found, try the main summary file
        if not tissue_files_found:
            stats_file = os.path.join(task_dir, pc, 'summary_statistics.csv')
            if os.path.exists(stats_file):
                print(f"\nLoading general statistics file: {stats_file}")
                df = pd.read_csv(stats_file, header=[0,1], index_col=[0,1])
                print("DataFrame shape:", df.shape)
                print("DataFrame columns:", df.columns)
                
                # Check if this file has tissue information
                if 'Class' in df.index.names:
                    class_values = df.index.get_level_values('Class').unique()
                    print(f"Tissue classes found in file: {class_values}")
                    tissues_in_data.update(class_values)
                
                all_data[pc] = df
    
    if not all_data:
        print("No data found!")
        return

    # Determine which tissues to process
    if tissues_in_data:
        print(f"\nTissue classes found in data: {tissues_in_data}")
        tissues_to_process = list(tissues_in_data)
    else:
        print(f"\nNo specific tissue classes found, using defaults: {DEFAULT_TISSUES}")
        tissues_to_process = DEFAULT_TISSUES

    # Check the first dataframe to see what metrics are available
    sample_data = next(iter(all_data.values()))
    metrics = sample_data.columns.get_level_values(0).unique()
    print(f"\nMetrics found: {metrics}")
    
    # Generate combined tables for each tissue and metric
    for tissue in tissues_to_process:
        # Create a subfolder for each tissue type
        tissue_dir = os.path.join(task_dir, tissue.replace(' ', '_'))
        os.makedirs(tissue_dir, exist_ok=True)
        
        for metric in metrics:
            latex_table = create_combined_tissue_table(all_data, tissue, metric)
            if latex_table:
                output_file = os.path.join(tissue_dir, f'{metric.lower()}_combined_table.txt')
                with open(output_file, 'w') as f:
                    f.write(latex_table)
                print(f"Generated combined table for {tissue} - {metric}")

if __name__ == "__main__":
    main() 