import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_scores(model_names, base_dir):
    # Create a directory for saving plots
    plot_dir = os.path.join(base_dir, 'plots', 'brainage')
    os.makedirs(plot_dir, exist_ok=True)

    # Load and combine data from all models
    all_data = []
    for model in model_names:
        csv_file = os.path.join(base_dir, model, 'ixi-classifier', 'test_results.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['Model'] = model
            all_data.append(df)
        else:
            print(f"Warning: CSV file not found for model {model}")

    if not all_data:
        print("No data found for any of the specified models.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Fix site names to look nicer
    combined_df['Site'] = combined_df['Site'].replace({'guys': 'Guys', 'hh': 'HH', 'iop': 'IOP'})
    # Fix modality names to look nicer
    combined_df['Modality'] = combined_df['Modality'].replace({'t1': 'T1w', 't2': 'T2w', 'pd': 'PDw'})

    # Create a new column combining Site and Modality
    combined_df['Site'] = combined_df['Site'] + ' - ' + combined_df['Modality']
    # Drop the Modality column
    combined_df = combined_df.drop(columns=['Modality'])

    # 1. Scatter plot of Predicted Age vs True Age
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=combined_df, x='True Age', y='Predicted Age', hue='Model', style='Site')
    plt.title('Predicted Age vs True Age')
    plt.plot([combined_df['True Age'].min(), combined_df['True Age'].max()], 
             [combined_df['True Age'].min(), combined_df['True Age'].max()], 
             'r--', lw=2)
    plt.savefig(os.path.join(plot_dir, 'predicted_vs_true_age_comparison.png'))
    plt.close()

    # 2. Box plot of Absolute Error by Site and Model
    plt.figure(figsize=(14, 8))
    combined_df['Absolute Error'] = abs(combined_df['Predicted Age'] - combined_df['True Age'])
    sns.boxplot(data=combined_df, x='Site', y='Absolute Error', hue='Model')
    plt.title('Absolute Error by Site and Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'absolute_error_boxplot_comparison.png'))
    plt.close()

    # 3. Bar plot of MSE by Site and Model
    plt.figure(figsize=(14, 8))
    sns.barplot(data=combined_df, x='Site', y='MSE', hue='Model')
    plt.title('Mean Squared Error by Site and Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mse_barplot_comparison.png'))
    plt.close()

    # 4. Bar plot of MAE by Site and Model
    plt.figure(figsize=(14, 8))
    sns.barplot(data=combined_df, x='Site', y='MAE', hue='Model')
    plt.title('Mean Absolute Error by Site and Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mae_barplot_comparison.png'))
    plt.close()

    # 5. Histogram of Age Distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(data=combined_df, x='True Age', hue='Site', multiple='stack', bins=20)
    plt.title('Age Distribution by Site')
    plt.savefig(os.path.join(plot_dir, 'age_distribution.png'))
    plt.close()

    # 6. Radar plot of MAE by Site and Model
    plt.figure(figsize=(12, 10))
    
    # Prepare data for radar plot
    sites = combined_df['Site'].unique()
    angles = np.linspace(0, 2*np.pi, len(sites), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # complete the circle
    
    ax = plt.subplot(111, polar=True)
    
    for model in model_names:
        model_data = combined_df[combined_df['Model'] == model].groupby('Site')['MAE'].mean()
        values = [model_data[site] for site in sites]
        values = np.concatenate((values, [values[0]]))  # complete the circle
        
        ax.plot(angles, values, '-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)
    
    # Set the labels and their positions
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(sites)
    
    # Move the labels outward
    ax.xaxis.set_tick_params(pad=10)
    
    # Remove the radial labels (y-axis labels)
    ax.set_yticklabels([])
    
    # Add subtle gridlines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Make the plot square
    ax.set_aspect('equal', 'box')
    
    ax.set_title('Mean Absolute Error by Site and Model', y=1.08)
    
    # Adjust the subplot to make room for the legend
    plt.subplots_adjust(bottom=0.2)
    
    # Place the legend below the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mae_radar_comparison.png'), bbox_inches='tight')
    plt.close()

    print(f"Comparison plots saved in {plot_dir}")

if __name__ == '__main__':
    # Specify the base directory where model folders are located
    base_dir = './'

    # List of model names to compare
    model_names = [
        '3d-cnn-simclr-mprage',
        '3d-cnn-simclr-bloch',
        ]

    plot_scores(model_names, base_dir)
