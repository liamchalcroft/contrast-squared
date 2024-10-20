import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from brainage_train import BrainAgeDataset, BrainAgeModel, EncoderModel

def test_brain_age_model(model_path, test_data_dir):
    # Load the trained model
    encoder = EncoderModel()
    model = BrainAgeModel(encoder)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create test dataset and dataloader
    test_dataset = BrainAgeDataset(test_data_dir, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    true_ages = []
    predicted_ages = []

    with torch.no_grad():
        for brain_images, ages in test_loader:
            outputs = model(brain_images)
            true_ages.extend(ages.numpy())
            predicted_ages.extend(outputs.squeeze().numpy())

    # Calculate metrics
    mae = mean_absolute_error(true_ages, predicted_ages)
    mse = mean_squared_error(true_ages, predicted_ages)
    r2 = r2_score(true_ages, predicted_ages)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

    # Plot predicted vs true ages
    plt.figure(figsize=(10, 10))
    plt.scatter(true_ages, predicted_ages, alpha=0.5)
    plt.plot([min(true_ages), max(true_ages)], [min(true_ages), max(true_ages)], 'r--')
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title("Predicted vs True Age")
    plt.savefig("predicted_vs_true_age.png")
    plt.close()

    # Save results to CSV
    results_df = pd.DataFrame({
        "True Age": true_ages,
        "Predicted Age": predicted_ages
    })
    results_df.to_csv("brain_age_results.csv", index=False)

if __name__ == "__main__":
    model_path = "brain_age_model.pth"
    test_data_dir = "path/to/your/test/data"
    test_brain_age_model(model_path, test_data_dir)
