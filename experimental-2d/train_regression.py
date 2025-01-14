import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from downstream_preprocess import get_train_val_loaders
from models import create_classification_model
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def plot_training_curves(train_losses, val_losses, output_dir, modality, site):
    """Plot training and validation loss curves."""
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
    plt.plot(val_epochs, val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'regression_curves_{modality}_{site}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

def train_regression(model_name, output_dir, weights_path, pretrained, epochs, batch_size, 
                    learning_rate, modality, site, amp=False, resume=False):
    """Train regression model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_classification_model(model_name, num_classes=1, weights_path=weights_path, pretrained=pretrained).to(device)
    
    # Get data loaders
    train_loader, val_loader = get_train_val_loaders(
        batch_size=batch_size,
        task='classification',
        modality=modality,
        site=site
    )
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler with warmup
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs-5)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[5])
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    if resume:
        checkpoint_path = Path(output_dir) / f"regression_model_{modality}_{site}_best.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    train_losses = []
    val_losses = []
    scaler = torch.cuda.amp.GradScaler() if amp else None
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        
        # Training
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in pbar:
                images = batch['image'].to(device)
                ages = batch['age'].float().to(device)
                
                optimizer.zero_grad()
                
                if amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(images).squeeze()
                        print(f"images: {images.shape}, outputs: {outputs.shape}, ages: {ages.shape}")
                        loss = criterion(outputs, ages)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, ages)
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                ages = batch['age'].float().to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, ages)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, Path(output_dir) / f"regression_model_{modality}_{site}_best.pth")
        
        scheduler.step()
    
    # Save final model
    torch.save({
        'epoch': epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }, Path(output_dir) / f"regression_model_{modality}_{site}_final.pth")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, output_dir, modality, site)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train regression model')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name of the model architecture')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save model checkpoints')
    parser.add_argument('--weights_path', type=str, default=None,
                      help='Path to pretrained weights')
    parser.add_argument('--pretrained', action='store_true',
                      help='Use pretrained weights')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--modality', type=str, default='t1',
                      choices=['t1', 't2', 'pd'],
                      help='Image modality')
    parser.add_argument('--site', type=str, default='GST',
                      choices=['GST', 'HH', 'IOP'],
                      help='Training site')
    parser.add_argument('--amp', action='store_true',
                      help='Use automatic mixed precision')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.resume and os.path.exists(os.path.join(args.output_dir, f"regression_model_{args.modality}_{args.site}_final.pth")):
        print("Final weights already exist, skipping training")
    else:
        if args.resume and not os.path.exists(os.path.join(args.output_dir, f"regression_model_{args.modality}_{args.site}_best.pth")):
            print("Resume flag set but no best checkpoint found, starting from scratch")
            args.resume = False
        train_regression(
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