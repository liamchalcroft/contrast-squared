import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from downstream_preprocess import get_train_val_loaders
from models import create_unet_model

def train_denoising(model_name, output_dir, weights_path=None, pretrained=False, epochs=10, batch_size=16, learning_rate=1e-3, modality='t1', site='GST', amp=False):
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
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    if amp:
        scaler = torch.cuda.amp.GradScaler()

    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            inputs = batch['image'].to(device)
            # For denoising, target is the same as input
            targets = inputs.clone()
            
            optimizer.zero_grad()
            with torch.amp.autocast(enabled=amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                inputs = batch['image'].to(device)
                targets = inputs.clone()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        print(f"Epoch {epoch+1}")
        print(f"Training Loss: {avg_train_loss:.6f}")
        print(f"Validation Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, output_dir / f"denoising_model_{modality}_{site}_best.pth")

    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, output_dir / f"denoising_model_{modality}_{site}_final.pth")

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

    args = parser.parse_args()
    train_denoising(
        args.model_name,
        args.output_dir,
        args.weights_path,
        args.pretrained,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.modality,
        args.site
    ) 