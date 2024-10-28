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
        data_pc = int(results_dir.split("pc")[1])
        df["% Training Data"] = data_pc
        method = results_dir.split("simclr-")[1].split("-")[0]
        df["Method"] = method.upper()
        return df
    else:
        print(f"Results file not found: {results_file}")
        return None
    
def spider_plot(results_df, metric="DSC"):
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
    
    plt.title(f"Mean {metric} by Modality Dataset and Method")
    return fig

# List of model configurations to check
results_dirs = glob.glob("stroke-cnn-simclr-*/")

# Create results directory
plot_dir = "plots/stroke_segmentation"
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

# Tidy up names of columns
results_df.rename(columns={"dice": "DSC", "hd95": "HD95", "class": "Class", "modality": "Modality", "dataset": "Dataset"}, inplace=True)

# Create a merged column for modality and dataset
results_df["Modality Dataset"] = results_df["Dataset"] + " [" + results_df["Modality"] + "]"
# Generate plots

# 1. Boxplot of Dice scores by model and dataset
plt.figure(figsize=(12, 6))
sns.boxplot(data=results_df, x="Modality Dataset", y="DSC", hue="Method")
plt.title("Dice Scores by Model and Dataset")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "dice_by_modality_dataset.png"))
plt.close()

# 2. Boxplot of HD95 scores by model and dataset
plt.figure(figsize=(12, 6))
sns.boxplot(data=results_df, x="Modality Dataset", y="HD95", hue="Method")
plt.title("HD95 Scores by Model and Dataset")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "hd95_by_modality_dataset.png"))
plt.close()

# 3. Spider plot of Dice scores by modality and dataset
plt.figure(figsize=(10, 10))
spider_plot(results_df, metric="DSC")
plt.savefig(os.path.join(plot_dir, "dice_spider_plot.png"))
plt.close()

# 4. Spider plot of HD95 scores by modality and dataset
plt.figure(figsize=(10, 10))
spider_plot(results_df, metric="HD95")
plt.savefig(os.path.join(plot_dir, "hd95_spider_plot.png"))
plt.close()

# 5. Summary statistics table
summary_stats = results_df.groupby(['Modality Dataset', 'Method'])[['DSC', 'HD95']].agg(['mean', 'std', 'median', 'min', 'max', 'sem']).round(3)
summary_stats.to_csv(os.path.join(plot_dir, "summary_statistics.csv"))

# Print overall summary
print("\nOverall Results Summary:")
print(summary_stats)
print(f"\nPlots saved in: {plot_dir}")
