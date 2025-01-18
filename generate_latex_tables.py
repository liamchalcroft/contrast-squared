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

def create_latex_table(data, metric, task):
    """Create a LaTeX table for a given metric"""
    # Start the table
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{" + f"{metric} results for {task.replace('_', ' ').title()}" + "}",
        "\\label{tab:" + f"{task}_{metric.lower()}" + "}",
        "\\begin{tabular}{l" + "c" * len(PERCENTAGES) * len(MODEL_ORDER) + "}",
        "\\toprule"
    ]

    # Create the header - convert integers to strings
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
    for dataset in data.index.get_level_values(0).unique():
        row_values = []
        for pc in PERCENTAGES:
            for model in MODEL_ORDER:
                try:
                    mean = data.loc[(dataset, model), (metric, 'mean')][pc]
                    std = data.loc[(dataset, model), (metric, 'std')][pc]
                    row_values.append(format_mean_std(mean, std))
                except:
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
                df = pd.read_csv(stats_file, header=[0,1,2], index_col=[0,1])
                all_data[pc] = df
            else:
                print(f"No summary statistics found for {task} {pc}")
                continue

        if not all_data:
            continue

        # Combine data from all percentages
        metrics = all_data[PERCENTAGES[0]].columns.get_level_values(0).unique()

        print(all_data.head())
        
        # Generate table for each metric
        for metric in metrics:
            latex_table = create_latex_table(all_data[PERCENTAGES[0]], metric, task)
            
            # Save table to file in the same directory as the plots
            output_file = os.path.join(task_dir, f'{metric.lower()}_table.txt')
            with open(output_file, 'w') as f:
                f.write(latex_table)
            print(f"Generated table for {task} {metric}")

if __name__ == "__main__":
    main() 