import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

    # 1. Scatter plot of Predicted Age vs True Age
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=combined_df, x='True Age', y='Predicted Age', hue='Model', style='Site')
    plt.title('Predicted Age vs True Age')
    plt.plot([combined_df['True Age'].min(), combined_df['True Age'].max()], 
             [combined_df['True Age'].min(), combined_df['True Age'].max()], 
             'r--', lw=2)
    plt.savefig(os.path.join(plot_dir, 'predicted_vs_true_age_comparison.png'))
    plt.close()

    # 2. Box plot of Absolute Error by Model and Site
    plt.figure(figsize=(14, 8))
    combined_df['Absolute Error'] = abs(combined_df['Predicted Age'] - combined_df['True Age'])
    sns.boxplot(data=combined_df, x='Model', y='Absolute Error', hue='Site')
    plt.title('Absolute Error by Model and Site')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'absolute_error_boxplot_comparison.png'))
    plt.close()

    # 3. Bar plot of MSE by Model and Site
    plt.figure(figsize=(14, 8))
    sns.barplot(data=combined_df, x='Model', y='MSE', hue='Site')
    plt.title('Mean Squared Error by Model and Site')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mse_barplot_comparison.png'))
    plt.close()

    # 4. Bar plot of MAE by Model and Site
    plt.figure(figsize=(14, 8))
    sns.barplot(data=combined_df, x='Model', y='MAE', hue='Site')
    plt.title('Mean Absolute Error by Model and Site')
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
