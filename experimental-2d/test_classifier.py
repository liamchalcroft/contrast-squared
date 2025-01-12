import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from models import create_classification_model
from downstream_preprocess import get_test_loader
import argparse
import os

def test_model(model, loader, criterion, device):
    """Evaluate model on test data."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Testing'):
            images = batch['image'].to(device)
            labels = batch['sex'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(loader)
    
    return avg_loss, accuracy, all_preds, all_labels

def test_classifier(model_dir, model_name, modality, site):
    """Test classifier for a specific modality and site."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = create_classification_model(
        model_name=model_name,
        num_classes=2,
        pretrained=False
    ).to(device)
    
    # Load model weights
    checkpoint_path = Path(model_dir) / f"classifier_{modality}_GST_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get test loader
    test_loader = get_test_loader(
        batch_size=32,
        task='classification',
        modality=modality,
        site=site
    )
    
    # Test model
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, predictions, labels = test_model(
        model, test_loader, criterion, device
    )
    
    # Return results as a single row DataFrame
    return pd.DataFrame([{
        'model_name': model_name,
        'modality': modality,
        'site': site,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'epoch': checkpoint['epoch']
    }])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test classification models and generate metrics')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing model checkpoints')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model for logging')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output CSV file')
    parser.add_argument('--modality', type=str, nargs='+', default=['t1', 't2', 'pd'], help='Modalities to test')
    parser.add_argument('--sites', type=str, nargs='+', default=['GST', 'HH', 'IOP'], help='Sites to test')
    
    args = parser.parse_args()
    
    # Initialize empty DataFrame for all results
    all_results = pd.DataFrame()
    
    # Test each modality and site combination
    for modality in args.modality:
        for site in args.sites:
            try:
                results = test_classifier(
                    args.model_dir,
                    args.model_name,
                    modality,
                    site
                )
                all_results = pd.concat([all_results, results], ignore_index=True)
            except Exception as e:
                print(f"Error testing {modality} {site}: {e}")
    
    # Save results
    all_results.to_csv(args.output_file, index=False)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    summary = all_results.groupby(['model_name', 'modality', 'site']).agg({
        'test_accuracy': ['mean', 'std'],
        'test_loss': ['mean', 'std']
    }).round(3)
    print(summary) 