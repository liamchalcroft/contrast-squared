import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from pathlib import Path
from models import create_classification_model
from downstream_preprocess import get_train_val_loaders, get_test_loader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import os
import matplotlib.pyplot as plt

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['sex'].to(device)  # Assuming binary classification for sex
        
        optimizer.zero_grad()
        outputs = model(images)
        print(f"outputs: {outputs.shape}, labels: {labels.shape}")
        print(labels.unique())
        loss = criterion(outputs.cpu(), labels.cpu().long())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
    
    return running_loss/len(loader), 100.*correct/total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['sex'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(loader), 100.*correct/total

def plot_training_curves(train_losses, val_accs, output_dir, modality, site):
    """Plot training loss and validation accuracy curves."""
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    val_epochs = list(range(0, len(train_losses), 2))
    plt.plot(val_epochs, val_accs, label='Validation Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'classification_curves_{modality}_{site}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

def train_classifier(model_name, output_dir, weights_path=None, pretrained=False, 
                    epochs=50, batch_size=32, learning_rate=1e-3, 
                    modality='t1', site='GST', amp=False, resume=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(project="brain-classification", 
               name=f"sex-classification-{modality}-{site}",
               config={
                   "model": model_name,
                   "epochs": epochs,
                   "batch_size": batch_size,
                   "learning_rate": learning_rate,
                   "modality": modality,
                   "site": site
               })
    
    # Get dataloaders
    train_loader, val_loader = get_train_val_loaders(
        batch_size=batch_size,
        task='classification',
        modality=modality,
        site=site
    )
    
    # Initialize model and move to device
    model = create_classification_model(
        model_name=model_name,
        num_classes=2,
        weights_path=weights_path,
        pretrained=pretrained
    ).to(device)
    
    # Freeze encoder weights
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
    print("Encoder weights frozen")
    
    # Only optimize decoder parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Set up learning rate scheduler
    warmup_epochs = 10
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=learning_rate * 0.01
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    if amp:
        scaler = torch.amp.GradScaler('cuda')
    
    # Initialize tracking
    train_losses = []
    val_accs = []
    best_val_acc = 0
    start_epoch = 0
    
    if resume:
        checkpoint = torch.load(output_dir / f"classifier_{modality}_{site}_best.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_accs = checkpoint['val_accs']
        best_val_acc = checkpoint['val_acc']
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        train_losses.append(train_loss)
        
        # Validate every 2 epochs
        if epoch % 2 == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            val_accs.append(val_acc)
            
            # Log metrics
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_acc': val_acc,
                    'train_losses': train_losses,
                    'val_accs': val_accs,
                }, output_dir / f"classifier_{modality}_{site}_best.pth")
            
            # Plot training curves
            plot_training_curves(train_losses, val_accs, output_dir, modality, site)
        
        # Learning rate scheduling
        scheduler.step()
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_acc': val_acc,
        'train_losses': train_losses,
        'val_accs': val_accs,
    }, output_dir / f"classifier_{modality}_{site}_final.pth")
    
    # Final plot
    plot_training_curves(train_losses, val_accs, output_dir, modality, site)
    
    # Test final model
    test_loader = get_test_loader(
        batch_size=batch_size,
        task='classification',
        modality=modality,
        site=site
    )
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    wandb.log({'test_loss': test_loss, 'test_acc': test_acc})

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a classification model on brain data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the model')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--weights_path', type=str, help='Path to the weights file')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--modality', type=str, default='t1', choices=['t1', 't2', 'pd'], help='Image modality')
    parser.add_argument('--site', type=str, default='GST', choices=['GST', 'HH', 'IOP'], help='Training site')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    
    args = parser.parse_args()
    
    if args.resume and os.path.exists(os.path.join(args.output_dir, f"classifier_{args.modality}_{args.site}_final.pth")):
        print("Final weights already exist, skipping training")
    else:
        if args.resume and not os.path.exists(os.path.join(args.output_dir, f"classifier_{args.modality}_{args.site}_best.pth")):
            print("Resume flag set but no best checkpoint found, starting from scratch")
            args.resume = False
        train_classifier(
            args.model_name,
            args.output_dir,
            args.weights_path,
            args.pretrained,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.modality,
            args.site,
            args.amp,
            args.resume
        ) 