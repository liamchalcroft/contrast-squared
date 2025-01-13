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

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

def plot_training_curves(train_losses, val_losses, val_dices, output_dir, modality, site):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    val_epochs = list(range(0, len(train_losses), 2))
    ax1.plot(val_epochs, val_losses, label='Validation Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training and Validation Losses - {modality.upper()} {site}')
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale('log')
    
    # Plot Dice scores
    ax2.plot(val_epochs, val_dices, label='Validation Dice', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title(f'Validation Dice Scores - {modality.upper()} {site}')
    ax2.legend()
    ax2.grid(True)
    
    plt.savefig(output_dir / f'training_curves_{modality}_{site}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

def train_segmentation(model_name, output_dir, weights_path=None, pretrained=False, epochs=10, 
                      batch_size=16, learning_rate=1e-3, modality='t1', site='GST', 
                      amp=False, resume=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get data loaders
    train_loader, val_loader = get_train_val_loaders(
        batch_size=batch_size,
        task='segmentation',
        modality=modality,
        site=site
    )
    
    # Initialize model
    model = create_unet_model(
        model_name=model_name,
        weights_path=weights_path,
        pretrained=pretrained,
        in_channels=1,
        out_channels=1  # Binary segmentation
    ).to(device)
    
    # Freeze encoder weights
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
    print("Encoder weights frozen")
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate
    )
    criterion = DiceLoss()

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
        scaler = torch.amp.GradScaler()

    # Initialize tracking
    train_losses = []
    val_losses = []
    val_dices = []
    best_dice = 0.0
    start_epoch = 0

    if resume:
        checkpoint = torch.load(output_dir / f"seg_healthy_{modality}_{site}_best.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        val_dices = checkpoint['val_dices']
        best_dice = checkpoint['best_dice']
    
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images = batch['image'].to(device)
            masks = batch['label'].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda' if amp else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        scheduler.step()
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if epoch % 2 == 0:
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                    images = batch['image'].to(device)
                    masks = batch['label'].to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    # Calculate Dice score
                    pred_masks = (outputs > 0.5).float()
                    dice = 1 - criterion(pred_masks, masks)
                    
                    val_loss += loss.item()
                    val_dice += dice.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            avg_val_dice = val_dice / val_batches
            val_losses.append(avg_val_loss)
            val_dices.append(avg_val_dice)
            
            print(f"Epoch {epoch+1}")
            print(f"Training Loss: {avg_train_loss:.6f}")
            print(f"Validation Loss: {avg_val_loss:.6f}")
            print(f"Validation Dice: {avg_val_dice:.6f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if avg_val_dice > best_dice:
                best_dice = avg_val_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_dice': avg_val_dice,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_dices': val_dices,
                    'best_dice': best_dice
                }, output_dir / f"seg_healthy_{modality}_{site}_best.pth")
            
            # Plot training curves
            plot_training_curves(train_losses, val_losses, val_dices, output_dir, modality, site)

    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'val_dice': avg_val_dice,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_dices': val_dices,
        'best_dice': best_dice
    }, output_dir / f"seg_healthy_{modality}_{site}_final.pth")
    
    # Final plot
    plot_training_curves(train_losses, val_losses, val_dices, output_dir, modality, site)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a healthy tissue segmentation model')
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
    
    if args.resume and os.path.exists(os.path.join(args.output_dir, f"seg_healthy_{args.modality}_{args.site}_final.pth")):
        print("Final weights already exist, skipping training")
    else:
        if args.resume and not os.path.exists(os.path.join(args.output_dir, f"seg_healthy_{args.modality}_{args.site}_best.pth")):
            print("Resume flag set but no best checkpoint found, starting from scratch")
            args.resume = False
        train_segmentation(
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