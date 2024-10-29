import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def get_results_df(results_dir):
    results_file = os.path.join(results_dir, "test_results.csv")
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        print(f"results_dir: {results_dir}")
        run_name = results_dir.split("/")[0]
        print(f"run_name: {run_name}")
        data_pc = int(run_name.split("pc")[1])
        df["% Training Data"] = data_pc
        method = run_name.split("simclr-")[1].split("-pc")[0]
        df["Method"] = method.upper()
        return df
    else:
        print(f"Results file not found: {results_file}")
        return None
    
def spider_plot(results_df, metric="DSC", class_name=None):
    # Filter by class if specified
    if class_name is not None:
        results_df = results_df[results_df['Class'] == class_name]
        
    # Calculate mean DSC for each modality dataset and method
    spider_data = results_df.groupby(['Modality Dataset', 'Method'])[metric].mean().unstack()
    
    # Set up the angles for the spider plot
    angles = np.linspace(0, 2*np.pi, len(spider_data.index), endpoint=False)
    
    # Close the plot by appending first value
    angles = np.concatenate((angles, [angles[0]]))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    for method in spider_data.columns:
        values = spider_data[method].values
        # Close the plot by appending first value
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, 'o-', linewidth=2, label=method)
        ax.fill(angles, values, alpha=0.25)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(spider_data.index, rotation=45)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    title = f"Mean {metric} by Modality Dataset and Method"
    if class_name:
        title += f" for {class_name}"
    plt.title(title)
    return fig

# List of model configurations to check
results_dirs = glob.glob("spm-cnn-simclr-*/")

# Create results directory
plot_dir = "plots/healthy_segmentation"
os.makedirs(plot_dir, exist_ok=True)

# Collect all results into single dataframe
all_results = []
for results_dir in results_dirs:
    df = get_results_df(results_dir)
    if df is not None:
        all_results.append(df)

if not all_results:
    print("No results found!")
    exit()

results_df = pd.concat(all_results, ignore_index=True)

results_df = results_df[results_df["% Training Data"] == 100]

# Tidy up names of columns
results_df.rename(columns={"dice": "DSC", "hd95": "HD95", "class": "Class", "modality": "Modality", "dataset": "Dataset"}, inplace=True)

# Create a merged column for modality and dataset
results_df["Modality Dataset"] = results_df["Dataset"] + " [" + results_df["Modality"] + "]"

# Get unique classes
classes = results_df['Class'].unique()

# Generate plots for all data and per-class
for class_name in [None] + list(classes):
    class_suffix = f"_{class_name.lower()}" if class_name else ""
    class_data = results_df[results_df['Class'] == class_name] if class_name else results_df
    
    # 1. Boxplot of Dice scores by model and dataset
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=class_data, x="Modality Dataset", y="DSC", hue="Method")
    title = "Dice Scores by Model and Dataset"
    if class_name:
        title += f" for {class_name}"
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"dice_by_modality_dataset{class_suffix}.png"))
    plt.close()

    # 2. Boxplot of HD95 scores by model and dataset
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=class_data, x="Modality Dataset", y="HD95", hue="Method")
    title = "HD95 Scores by Model and Dataset"
    if class_name:
        title += f" for {class_name}"
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"hd95_by_modality_dataset{class_suffix}.png"))
    plt.close()

    # 3. Spider plot of Dice scores by modality and dataset
    plt.figure(figsize=(10, 10))
    spider_plot(results_df, metric="DSC", class_name=class_name)
    plt.savefig(os.path.join(plot_dir, f"dice_spider_plot{class_suffix}.png"))
    plt.close()

    # 4. Spider plot of HD95 scores by modality and dataset
    plt.figure(figsize=(10, 10))
    spider_plot(results_df, metric="HD95", class_name=class_name)
    plt.savefig(os.path.join(plot_dir, f"hd95_spider_plot{class_suffix}.png"))
    plt.close()

    # 5. Summary statistics table
    summary_stats = class_data.groupby(['Modality Dataset', 'Method'])[['DSC', 'HD95']].agg(['mean', 'std', 'median', 'min', 'max', 'sem']).round(3)
    if class_name:
        summary_stats.to_csv(os.path.join(plot_dir, f"summary_statistics_{class_name.lower()}.csv"))
    else:
        summary_stats.to_csv(os.path.join(plot_dir, "summary_statistics.csv"))

    # Print summary for this class
    print(f"\nResults Summary{' for ' + class_name if class_name else ''}:")
    print(summary_stats)

print(f"\nPlots saved in: {plot_dir}")
