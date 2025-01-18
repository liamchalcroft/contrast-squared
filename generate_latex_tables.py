import os
import pandas as pd
import numpy as np

TASKS = ['healthy_segmentation', 'denoise', 'stroke_segmentation']
PERCENTAGES = ['1pc', '10pc', '100pc']
MODEL_ORDER = ['Baseline', 'Sequence-augmented', 'Sequence-invariant']

def format_mean_std(mean, std, is_best=False, is_second=False):
    """Format mean ± std with appropriate precision and highlighting"""
    if np.isnan(mean) or np.isnan(std):
        return "---"
    
    if mean < 0.01:
        formatted = f"{mean:.2e} ± {std:.2e}"
    elif mean < 1:
        formatted = f"{mean:.3f} ± {std:.3f}"
    else:
        formatted = f"{mean:.1f} ± {std:.1f}"
    
    if is_best:
        formatted = "\\textbf{" + formatted + "}"
    elif is_second:
        formatted = "\\underline{" + formatted + "}"
    
    return formatted

def get_ranking_indices(values):
    """Get indices of best and second best values, handling NaN"""
    valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
    if not valid_indices:
        return None, None
    
    sorted_indices = sorted(valid_indices, key=lambda i: values[i], reverse=True)
    best_idx = sorted_indices[0] if len(sorted_indices) > 0 else None
    second_idx = sorted_indices[1] if len(sorted_indices) > 1 else None
    
    return best_idx, second_idx

def create_latex_table(all_data, metric, task):
    """Create a LaTeX table for a given metric"""
    print(f"\nCreating table for {task} - {metric}")
    
    total_cols = len(PERCENTAGES) * len(MODEL_ORDER) + 1
    
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l" + "c" * (total_cols-1) + "}",
        "\\toprule"
    ]

    header = ["& \\multicolumn{" + str(len(MODEL_ORDER)) + "}{c}{" + str(int(pc.replace('pc',''))) + "\\%}" 
             for pc in PERCENTAGES]
    latex_lines.append("Dataset " + " ".join(header) + " \\\\")
    
    model_line = "& " + " & ".join(MODEL_ORDER * len(PERCENTAGES)) + " \\\\"
    latex_lines.extend([
        f"\\cmidrule(lr){{2-{total_cols}}}",
        model_line,
        "\\midrule"
    ])

    print("\nProcessing datasets:")
    datasets = all_data[PERCENTAGES[-1]].index.get_level_values(0).unique()
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        row_values = []
        
        for pc in PERCENTAGES:
            print(f"  Processing {pc}")
            df = all_data[pc]
            
            # Get values for all models for this dataset and percentage
            values = []
            for model in MODEL_ORDER:
                try:
                    mean = df.loc[dataset].loc[model][(metric, 'mean')]
                    values.append(mean)
                except:
                    values.append(np.nan)
            
            # Determine best and second best
            best_idx, second_idx = get_ranking_indices(values)
            
            # Format each model's value with appropriate highlighting
            for i, model in enumerate(MODEL_ORDER):
                try:
                    mean = df.loc[dataset].loc[model][(metric, 'mean')]
                    std = df.loc[dataset].loc[model][(metric, 'std')]
                    is_best = (i == best_idx)
                    is_second = (i == second_idx)
                    formatted = format_mean_std(mean, std, is_best, is_second)
                    print(f"    {model}: {formatted}")
                    row_values.append(formatted)
                except Exception as e:
                    print(f"    Error for {model}: {str(e)}")
                    row_values.append("---")
                    
        latex_lines.append(dataset + " & " + " & ".join(row_values) + " \\\\")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}"
    ])

    return "\n".join(latex_lines)

def main():
    for task in TASKS:
        print(f"\nProcessing task: {task}")
        task_dir = os.path.join('plots', task)
        if not os.path.exists(task_dir):
            print(f"No results directory found for {task}")
            continue

        all_data = {}
        
        for pc in PERCENTAGES:
            stats_file = os.path.join(task_dir, pc, 'summary_statistics.csv')
            if os.path.exists(stats_file):
                print(f"\nLoading {stats_file}")
                df = pd.read_csv(stats_file, header=[0,1], index_col=[0,1])
                print("DataFrame shape:", df.shape)
                print("DataFrame columns:", df.columns)
                all_data[pc] = df
            else:
                print(f"No summary statistics found for {task} {pc}")
                continue

        if not all_data:
            continue

        metrics = all_data[PERCENTAGES[0]].columns.get_level_values(0).unique()
        print(f"\nMetrics found: {metrics}")
        
        for metric in metrics:
            latex_table = create_latex_table(all_data, metric, task)
            
            output_file = os.path.join(task_dir, f'{metric.lower()}_table.txt')
            with open(output_file, 'w') as f:
                f.write(latex_table)
            print(f"Generated table for {task} {metric}")

if __name__ == "__main__":
    main() 