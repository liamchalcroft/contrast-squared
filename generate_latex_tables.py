import os
import pandas as pd
import numpy as np

TASKS = ['healthy_segmentation', 'denoise', 'stroke_segmentation']
PERCENTAGES = ['100pc', '10pc', '1pc']
MODEL_ORDER = ['Baseline', 'Sequence-augmented', 'Sequence-invariant']

def format_mean_std(mean, std):
    """Format mean ± std with appropriate precision"""
    if np.isnan(mean) or np.isnan(std):
        return "---"
    if mean < 0.01:
        return f"{mean:.2e} ± {std:.2e}"
    elif mean < 1:
        return f"{mean:.3f} ± {std:.3f}"
    else:
        return f"{mean:.1f} ± {std:.1f}"

def create_latex_table(all_data, metric, task):
    """Create a LaTeX table for a given metric"""
    print(f"\nCreating table for {task} - {metric}")
    
    # Start the table
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{" + f"{metric} results for {task.replace('_', ' ').title()}" + "}",
        "\\label{tab:" + f"{task}_{metric.lower()}" + "}",
        "\\begin{tabular}{l" + "c" * len(PERCENTAGES) * len(MODEL_ORDER) + "}",
        "\\toprule"
    ]

    # Create the header
    header = ["& \\multicolumn{" + str(len(MODEL_ORDER)) + "}{c}{" + str(pc) + "\\%}" 
             for pc in [100, 10, 1]]
    latex_lines.append("Dataset " + " ".join(header) + " \\\\")
    
    # Add model names
    model_line = "& " + " & ".join(MODEL_ORDER * len(PERCENTAGES)) + " \\\\"
    latex_lines.extend([
        "\\cmidrule(lr){2-" + str(len(MODEL_ORDER) + 1) + "}" * len(PERCENTAGES),
        model_line,
        "\\midrule"
    ])

    # Add data rows
    print("\nProcessing datasets:")
    datasets = all_data[PERCENTAGES[0]].index.get_level_values(0).unique()
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        row_values = []
        for pc in PERCENTAGES:
            print(f"  Processing {pc}")
            df = all_data[pc]
            for model in MODEL_ORDER:
                try:
                    mean = df.loc[dataset].loc[model][(metric, 'mean')]
                    std = df.loc[dataset].loc[model][(metric, 'std')]
                    formatted = format_mean_std(mean, std)
                    print(f"    {model}: {formatted}")
                    row_values.append(formatted)
                except Exception as e:
                    print(f"    Error for {model}: {str(e)}")
                    row_values.append("---")
        latex_lines.append(dataset + " & " + " & ".join(row_values) + " \\\\")

    # Close the table
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
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

        # Create dictionary to store data for each percentage
        all_data = {}
        
        # Load data from each percentage
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

        # Get metrics from the first percentage's data
        metrics = all_data[PERCENTAGES[0]].columns.get_level_values(0).unique()
        print(f"\nMetrics found: {metrics}")
        
        # Generate table for each metric
        for metric in metrics:
            latex_table = create_latex_table(all_data, metric, task)
            
            # Save table to file in the same directory as the plots
            output_file = os.path.join(task_dir, f'{metric.lower()}_table.txt')
            with open(output_file, 'w') as f:
                f.write(latex_table)
            print(f"Generated table for {task} {metric}")

if __name__ == "__main__":
    main() 