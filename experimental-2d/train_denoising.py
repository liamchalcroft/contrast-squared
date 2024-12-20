import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from downstream_preprocess import get_train_val_loaders
from models import create_unet_model
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def strip_prefix_state_dict(state_dict, substring_to_remove):
    """Load weights and remove a specific prefix from the keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace(substring_to_remove, '', 1)  # Remove the prefix
        new_state_dict[new_key] = v
    return new_state_dict

def plot_training_curves(train_losses, val_losses, output_dir, modality, site):
    """Plot training and validation curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    
    # Plot validation points (every 2nd epoch)
    val_epochs = list(range(0, len(train_losses), 2))
    plt.plot(val_epochs, val_losses, label='Validation Loss', marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Losses - {modality.upper()} {site}')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Save plot
    plt.savefig(output_dir / f'training_curves_{modality}_{site}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

def train_denoising(model_name, output_dir, weights_path=None, pretrained=False, epochs=10, batch_size=16, learning_rate=1e-3, modality='t1', site='GST', amp=False, resume=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get data loaders
    train_loader, val_loader = get_train_val_loaders(
        batch_size=batch_size,
        task='denoising',
        modality=modality,
        site=site
    )
    
    # Initialize model, loss, and optimizer
    model = create_unet_model(model_name, weights_path, pretrained).to(device)
    
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
    criterion = nn.MSELoss()

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
        eta_min=learning_rate * 0.01  # Minimum LR will be 1% of initial LR
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    if amp:
        scaler = torch.amp.GradScaler('cuda')

    # Initialize loss tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_epoch = 0

    if resume:
        checkpoint = torch.load(output_dir / f"denoising_model_{modality}_{site}_best.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        best_val_loss = checkpoint['val_loss']
    
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            inputs = batch['image'].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda' if amp else None):
                std = torch.rand(1, dtype=inputs.dtype, device=inputs.device) * 0.2
                noise = torch.randn_like(inputs) * std
                inputs = inputs + noise
                outputs = model(inputs)
                loss = criterion(outputs, noise)
            
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Step the scheduler
        scheduler.step()
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase and checkpoint saving
        if epoch % 2 == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                    inputs = batch['image'].to(device)
                    std = torch.rand(1, dtype=inputs.dtype, device=inputs.device) * 0.2
                    noise = torch.randn_like(inputs) * std
                    inputs = inputs + noise
                    outputs = model(inputs)
                    loss = criterion(outputs, noise)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}")
            print(f"Training Loss: {avg_train_loss:.6f}")
            print(f"Validation Loss: {avg_val_loss:.6f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                }, output_dir / f"denoising_model_{modality}_{site}_best.pth")
            
            # Plot training curves
            plot_training_curves(train_losses, val_losses, output_dir, modality, site)

    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, output_dir / f"denoising_model_{modality}_{site}_final.pth")
    
    # Final plot
    plot_training_curves(train_losses, val_losses, output_dir, modality, site)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a denoising model on IXI data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the model')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--weights_path', type=str, help='Path to the weights file')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--modality', type=str, default='t1', choices=['t1', 't2', 'pd'], help='Image modality')
    parser.add_argument('--site', type=str, default='GST', choices=['GST', 'HH', 'IOP'], help='Training site')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')

    args = parser.parse_args()
    if args.resume and os.path.exists(os.path.join(args.output_dir, f"denoising_model_{args.modality}_{args.site}_final.pth")):
        print("Final weights already exist, skipping training")
    else:
        if args.resume and not os.path.exists(os.path.join(args.output_dir, f"denoising_model_{args.modality}_{args.site}_best.pth")):
            print("Resume flag set but no best checkpoint found, starting from scratch")
            args.resume = False
        train_denoising(
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