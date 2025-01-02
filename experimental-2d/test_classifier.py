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

def main():
    parser = argparse.ArgumentParser(description='Test a classification model on brain data')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory containing model checkpoints')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--modality', type=str, default='t1', choices=['t1', 't2', 'pd'], help='Image modality')
    parser.add_argument('--site', type=str, default='GST', choices=['GST', 'HH', 'IOP'], help='Test site')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(args.checkpoint_dir)
    results_file = checkpoint_dir / f'test_results_{args.modality}_{args.site}.csv'
    
    # Initialize model
    model = create_classification_model(
        model_name=args.model_name,
        num_classes=2,
        pretrained=False
    ).to(device)
    
    # Load model weights
    checkpoint_path = checkpoint_dir / f"classifier_{args.modality}_{args.site}_best.pth"
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Get test loader
    test_loader = get_test_loader(
        batch_size=args.batch_size,
        task='classification',
        modality=args.modality,
        site=args.site
    )
    
    # Test model
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, predictions, labels = test_model(
        model, test_loader, criterion, device
    )
    
    # Save results
    results = {
        'model_name': [args.model_name],
        'modality': [args.modality],
        'site': [args.site],
        'test_loss': [test_loss],
        'test_accuracy': [test_acc],
        'epoch': [checkpoint['epoch']]
    }
    
    df = pd.DataFrame(results)
    print("\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.2f}%")
    
    # Append results if file exists, otherwise create new file
    if results_file.exists():
        df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        df.to_csv(results_file, index=False)
    
    print(f"\nResults saved to {results_file}")

if __name__ == '__main__':
    main() 