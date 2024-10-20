import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_scores(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create a directory for saving plots
    plot_dir = os.path.dirname(csv_file)
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Scatter plot of Predicted Age vs True Age
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='True Age', y='Predicted Age', hue='Site', style='Modality')
    plt.title('Predicted Age vs True Age')
    plt.plot([df['True Age'].min(), df['True Age'].max()], 
             [df['True Age'].min(), df['True Age'].max()], 
             'r--', lw=2)
    plt.savefig(os.path.join(plot_dir, 'predicted_vs_true_age.png'))
    plt.close()

    # 2. Box plot of Absolute Error by Site and Modality
    plt.figure(figsize=(12, 6))
    df['Absolute Error'] = abs(df['Predicted Age'] - df['True Age'])
    sns.boxplot(data=df, x='Site', y='Absolute Error', hue='Modality')
    plt.title('Absolute Error by Site and Modality')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'absolute_error_boxplot.png'))
    plt.close()

    # 3. Bar plot of MSE by Site and Modality
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Site', y='MSE', hue='Modality')
    plt.title('Mean Squared Error by Site and Modality')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mse_barplot.png'))
    plt.close()

    # 4. Bar plot of MAE by Site and Modality
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Site', y='MAE', hue='Modality')
    plt.title('Mean Absolute Error by Site and Modality')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mae_barplot.png'))
    plt.close()

    # 5. Histogram of Age Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='True Age', hue='Site', multiple='stack', bins=20)
    plt.title('Age Distribution by Site')
    plt.savefig(os.path.join(plot_dir, 'age_distribution.png'))
    plt.close()

    print(f"Plots saved in {plot_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot brain age prediction scores')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file with brain age prediction results')
    args = parser.parse_args()

    plot_scores(args.csv_file)
