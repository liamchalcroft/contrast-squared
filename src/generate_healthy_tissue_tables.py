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

def create_combined_tissue_table(all_tissue_data, tissue, metric):
    """Create a LaTeX table for a specific tissue type combining all percentages"""
    print(f"\nCreating combined table for {tissue} tissue - {metric}")
    
    # Check if we have data for this tissue
    if tissue not in all_tissue_data:
        print(f"No data found for {tissue}")
        return None
    
    tissue_data = all_tissue_data[tissue]
    
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

    # Create a nested structure to store all tissue data by tissue type and percentage
    all_tissue_data = {}
    
    # First, let's see if we have class-specific data inside the main statistics file
    has_class_specific_data = False
    for pc in PERCENTAGES:
        stats_file = os.path.join(task_dir, pc, 'summary_statistics.csv')
        if os.path.exists(stats_file):
            print(f"\nChecking main statistics file: {stats_file}")
            df = pd.read_csv(stats_file, header=[0,1], index_col=[0,1])
            
            # Check if this has Class in the index
            if 'Class' in df.index.names:
                has_class_specific_data = True
                class_values = df.index.get_level_values('Class').unique()
                print(f"Found tissue classes in main file: {class_values}")
                
                # Store data for each tissue type
                for tissue in class_values:
                    if tissue not in all_tissue_data:
                        all_tissue_data[tissue] = {}
                    
                    try:
                        tissue_df = df.xs(tissue, level='Class', drop_level=False)
                        all_tissue_data[tissue][pc] = tissue_df
                        print(f"  Loaded data for {tissue} from {pc}")
                    except Exception as e:
                        print(f"  Error extracting {tissue} from {pc}: {e}")
                break
    
    # If no class-specific data found in main files, look for tissue-specific files
    if not has_class_specific_data:
        # Try to load tissue-specific summary files
        for tissue in DEFAULT_TISSUES:
            tissue_name = tissue.replace(' ', '_')
            found_data = False
            
            for pc in PERCENTAGES:
                tissue_file = os.path.join(task_dir, pc, f"summary_statistics_{tissue_name}.csv")
                if os.path.exists(tissue_file):
                    if tissue not in all_tissue_data:
                        all_tissue_data[tissue] = {}
                    
                    print(f"\nLoading tissue-specific file: {tissue_file}")
                    df = pd.read_csv(tissue_file, header=[0,1], index_col=[0,1])
                    all_tissue_data[tissue][pc] = df
                    found_data = True
                    print(f"  Loaded data for {tissue} from {pc}")
            
            if found_data:
                print(f"Found data for tissue: {tissue}")
    
    if not all_tissue_data:
        print("No tissue-specific data found!")
        return
    
    print(f"\nFound data for {len(all_tissue_data)} tissues: {list(all_tissue_data.keys())}")
    
    # Get metrics from the first available dataframe
    sample_tissue = next(iter(all_tissue_data.keys()))
    sample_pc = next(iter(all_tissue_data[sample_tissue].keys()))
    sample_df = all_tissue_data[sample_tissue][sample_pc]
    metrics = sample_df.columns.get_level_values(0).unique()
    print(f"\nMetrics found: {metrics}")
    
    # Generate combined tables for each tissue and metric
    for tissue in all_tissue_data:
        # Create a subfolder for each tissue type
        tissue_dir = os.path.join(task_dir, tissue.replace(' ', '_'))
        os.makedirs(tissue_dir, exist_ok=True)
        
        for metric in metrics:
            latex_table = create_combined_tissue_table(all_tissue_data, tissue, metric)
            if latex_table:
                output_file = os.path.join(tissue_dir, f'{metric.lower()}_combined_table.txt')
                with open(output_file, 'w') as f:
                    f.write(latex_table)
                print(f"Generated combined table for {tissue} - {metric}")

if __name__ == "__main__":
    main() 