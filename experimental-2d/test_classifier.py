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

def test_classifier(model_dir, model_name, modality, site, t1_cross_modal=False):
    """Test classifier on specified modality and site.
    
    Args:
        model_dir: Directory containing model checkpoint
        model_name: Name of model for logging
        modality: Modality to test on ('t1', 't2', 'pd')
        site: Site to test on ('GST', 'HH', 'IOP')
        t1_cross_modal: If True, use T1w-finetuned model for testing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_classification_model(
        model_name='timm/resnet50.a1_in1k',  # Always use ResNet-50 architecture
        num_classes=2,
        pretrained=False
    ).to(device)
    
    # Load checkpoint - use T1w checkpoint if doing cross-modal testing
    checkpoint_modality = 't1' if t1_cross_modal else modality
    checkpoint_path = Path(model_dir) / f"classifier_model_{checkpoint_modality}_GST_final.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get test loader for target modality
    test_loader = get_test_loader(
        batch_size=32,
        task='classification',
        modality=modality,
        site=site
    )
    
    # Test loop
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing {modality} {site}"):
            images = batch['image'].to(device)
            labels = batch['sex'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
    
    accuracy = correct / total
    avg_loss = running_loss / len(test_loader)
    
    return pd.DataFrame([{
        'model': model_name,
        'modality': modality,
        'site': site,
        't1_finetuned': t1_cross_modal,
        'test_accuracy': accuracy,
        'test_loss': avg_loss
    }])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test classification models')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--modality', type=str, nargs='+', default=['t1', 't2', 'pd'])
    parser.add_argument('--sites', type=str, nargs='+', default=['GST', 'HH', 'IOP'])
    
    args = parser.parse_args()
    
    # Initialize results DataFrame
    all_results = pd.DataFrame()
    
    # Standard testing on each modality
    for modality in args.modality:
        for site in args.sites:
            try:
                results = test_classifier(
                    args.model_dir,
                    args.model_name,
                    modality,
                    site,
                    t1_cross_modal=False
                )
                all_results = pd.concat([all_results, results], ignore_index=True)
            except Exception as e:
                print(f"Error testing {modality} {site}: {e}")
    
    # Cross-modal testing using T1w-finetuned model
    for modality in args.modality:
        for site in args.sites:
            if modality != 't1':  # Skip T1w as it's already tested
                try:
                    results = test_classifier(
                        args.model_dir,
                        args.model_name,
                        modality,
                        site,
                        t1_cross_modal=True
                    )
                    all_results = pd.concat([all_results, results], ignore_index=True)
                except Exception as e:
                    print(f"Error in cross-modal testing {modality} {site}: {e}")
    
    # Save results
    all_results.to_csv(args.output_file, index=False)
    
    # Print summary
    print("\nSummary Statistics:")
    summary = all_results.groupby(['model', 'modality', 't1_finetuned']).agg({
        'test_accuracy': ['mean', 'std'],
        'test_loss': ['mean', 'std']
    }).round(3)
    print(summary) 