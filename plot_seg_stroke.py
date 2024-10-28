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

# Set style for fancy plots
plt.style.use('seaborn-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2

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

# Multiply DSC by 100 to get percentage
results_df["DSC"] = results_df["DSC"] * 100

# Create a merged column for modality and dataset
results_df["Modality Dataset"] = results_df["Dataset"] + " [" + results_df["Modality"] + "]"

# Generate plots with enhanced styling
# 1. Boxplot of Dice scores by model and dataset
plt.figure(figsize=(15, 8))
ax = sns.boxplot(data=results_df, x="Modality Dataset", y="DSC", hue="Method",
                 boxprops={'alpha': 0.8, 'linewidth': 2},
                 showfliers=False, # Hide outliers for cleaner look
                 width=0.8)
plt.title("Dice Similarity Coefficient by Modality and Dataset", pad=20, fontsize=16, fontweight='bold')
plt.xlabel("Modality Dataset", fontsize=14, labelpad=15)
plt.ylabel("DSC (%)", fontsize=14, labelpad=15)
plt.xticks(rotation=45, ha='right')
ax.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Method", title_fontsize=12, fontsize=11, bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "dice_by_modality_dataset.png"), dpi=300, bbox_inches='tight')
plt.close()

# 2. Boxplot of HD95 scores by model and dataset
plt.figure(figsize=(15, 8))
ax = sns.boxplot(data=results_df, x="Modality Dataset", y="HD95", hue="Method",
                 boxprops={'alpha': 0.8, 'linewidth': 2},
                 showfliers=False,
                 width=0.8)
plt.title("95% Hausdorff Distance by Modality and Dataset", pad=20, fontsize=16, fontweight='bold')
plt.xlabel("Modality Dataset", fontsize=14, labelpad=15)
plt.ylabel("HD95 (mm)", fontsize=14, labelpad=15)
plt.xticks(rotation=45, ha='right')
ax.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Method", title_fontsize=12, fontsize=11, bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "hd95_by_modality_dataset.png"), dpi=300, bbox_inches='tight')
plt.close()

# 3. Spider plot of Dice scores by modality and dataset
plt.figure(figsize=(10, 10))
spider_plot(results_df, metric="DSC")
plt.savefig(os.path.join(plot_dir, "dice_spider_plot.png"), dpi=300, bbox_inches='tight')
plt.close()

# 4. Spider plot of HD95 scores by modality and dataset
plt.figure(figsize=(10, 10))
spider_plot(results_df, metric="HD95")
plt.savefig(os.path.join(plot_dir, "hd95_spider_plot.png"), dpi=300, bbox_inches='tight')
plt.close()

# 5. Summary statistics table
summary_stats = results_df.groupby(['Modality Dataset', 'Method'])[['DSC', 'HD95']].agg(['mean', 'std', 'median', 'min', 'max', 'sem']).round(1)
summary_stats.to_csv(os.path.join(plot_dir, "summary_statistics.csv"))

# Print overall summary
print("\nOverall Results Summary:")
print(summary_stats)
print(f"\nPlots saved in: {plot_dir}")
