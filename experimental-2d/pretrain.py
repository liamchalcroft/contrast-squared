import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.cuda.amp as amp
import timm
from preprocess import get_bloch_loader, get_mprage_loader
from tqdm import tqdm
import wandb
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning"""
    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.layer3(x)
        return F.normalize(x, dim=1)

class ContrastiveModel(nn.Module):
    """Encoder + projection head for contrastive learning"""
    def __init__(self, model_name='resnet18', pretrained=False):
        super().__init__()
        
        # Initialize encoder from timm
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            in_chans=1,     # Single channel input
        )
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 224, 224)
            output = self.encoder(dummy_input)
            self.encoder_dim = output.shape[1]
        
        # Projection head
        self.projector = ProjectionHead(self.encoder_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

def nt_xent_loss(z1, z2, temperature=0.1):
    """
    Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
    """
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    
    # Compute cosine similarity
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    
    # Mask out self-similarity
    sim_ij = torch.diag(sim, batch_size)
    sim_ji = torch.diag(sim, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)
    
    # Mask out self-similarity in denominator
    mask = (~torch.eye(2 * batch_size, dtype=bool, device=sim.device)).float()
    negatives = sim * mask
    
    # Compute log_prob
    exp_pos = torch.exp(positives)
    exp_neg = torch.exp(negatives)
    log_prob = positives - torch.log(exp_pos + exp_neg.sum(dim=1))
    
    # Compute final loss
    loss = -log_prob.mean()
    
    return loss

def nt_xent_loss_multi_view(zs, temperature=0.1):
    """
    Standard multi-view NT-Xent loss with improved numerical stability
    Args:
        zs: list of tensors, each of shape [batch_size, dim]
        temperature: temperature parameter
    """
    batch_size = zs[0].shape[0]
    num_views = len(zs)
    
    # Concatenate all views
    cat_zs = torch.cat(zs, dim=0)  # [num_views * batch_size, dim]
    
    # Compute cosine similarity with numerical stability
    cat_zs = F.normalize(cat_zs, dim=1)  # Ensure normalized vectors
    sim = torch.matmul(cat_zs, cat_zs.T) / temperature  # [num_views * batch_size, num_views * batch_size]
    
    # Create mask for positive pairs
    pos_mask = torch.zeros_like(sim)
    for i in range(num_views):
        for j in range(num_views):
            if i != j:
                pos_mask[
                    i * batch_size:(i + 1) * batch_size,
                    j * batch_size:(j + 1) * batch_size
                ] = torch.eye(batch_size, device=sim.device)
    
    # Create mask for valid negatives (excluding self-comparisons)
    neg_mask = (~torch.eye(batch_size * num_views, dtype=bool, device=sim.device)).float()
    
    # Compute log_prob with improved numerical stability
    max_val, _ = torch.max(sim * neg_mask, dim=1, keepdim=True)
    numerator = torch.exp(sim - max_val)
    
    # Compute positive and negative scores
    pos_scores = (numerator * pos_mask).sum(dim=1)
    neg_scores = (numerator * neg_mask).sum(dim=1)
    
    # Compute final loss with numerical stability
    eps = 1e-8
    loss = -torch.log((pos_scores + eps) / (neg_scores + eps))
    
    # Check for invalid values
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print("Warning: Invalid values in loss calculation")
        print(f"pos_scores min/max: {pos_scores.min():.4f}/{pos_scores.max():.4f}")
        print(f"neg_scores min/max: {neg_scores.min():.4f}/{neg_scores.max():.4f}")
        # Return a valid loss value to prevent training breakdown
        return torch.tensor(10.0, device=loss.device, requires_grad=True)
    
    loss = loss.mean() / (num_views - 1)
    return loss

class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

def save_checkpoint(state, is_best, checkpoint_dir):
    """Save checkpoint and keep best model separately"""
    checkpoint_path = checkpoint_dir / 'latest_model.pt'
    best_path = checkpoint_dir / 'best_model.pt'
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, best_path)
        logging.info(f"Saved new best model with loss: {state['loss']:.6f}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load checkpoint and return training state"""
    if not checkpoint_path.exists():
        return 0, float('inf')
    
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'] + 1, checkpoint['best_loss']

def vicreg_loss(z1, z2, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    """
    VICReg loss from "VICReg: Variance-Invariance-Covariance Regularization" paper
    """
    # Invariance loss (MSE)
    sim_loss = F.mse_loss(z1, z2)
    
    # Variance loss
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-04)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-04)
    var_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    
    # Covariance loss
    N, D = z1.shape
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)
    diag_mask = ~torch.eye(D, device=z1.device, dtype=torch.bool)
    cov_loss = (cov_z1[diag_mask]**2).mean() + (cov_z2[diag_mask]**2).mean()
    
    loss = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
    return loss

def barlow_twins_loss(z1, z2, lambda_param=5e-3):
    """
    Barlow Twins loss from "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"
    """
    # Normalize representations along batch dimension
    z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-6)
    z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-6)
    
    N = z1.size(0)
    D = z1.size(1)
    
    # Cross-correlation matrix
    c = (z1_norm.T @ z2_norm) / N
    
    # Loss
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.flatten()[1:].view(D-1, D+1)[:,:-1].pow_(2).sum()
    loss = on_diag + lambda_param * off_diag
    
    return loss

def train_epoch(model, loader, optimizer, device, epoch, scaler=None, ema=None, loss_type='nt_xent'):
    model.train()
    total_loss = 0
    num_batches = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        # Extract views from batch dictionary, excluding patient_id
        views = [batch[f'image{i+1}'].to(device, non_blocking=True) 
                for i in range(len(batch)-1)]  # -1 to exclude patient_id
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            if loss_type == 'nt_xent':
                zs = [model(view)[1] for view in views]
                loss = nt_xent_loss_multi_view(zs)
            elif loss_type == 'vicreg':
                z1, z2 = [model(view)[1] for view in views[:2]]  # Only use first two views
                loss = vicreg_loss(z1, z2)
            elif loss_type == 'barlow':
                z1, z2 = [model(view)[1] for view in views[:2]]  # Only use first two views
                loss = barlow_twins_loss(z1, z2)
        
        # Check for NaN loss
        if torch.isnan(loss):
            logging.error(f"NaN loss detected at epoch {epoch}, batch {num_batches}")
            raise ValueError("NaN loss detected")
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if ema is not None:
            ema.update()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': total_loss / num_batches})
        
        # Log every batch since we have so few
        wandb.log({
            'batch_loss': loss.item(),
            'epoch': epoch,
            'batch': num_batches,
            'global_step': epoch * len(loader) + num_batches
        })
    
    # Clear CUDA cache at the end of epoch
    torch.cuda.empty_cache()
    
    return total_loss / num_batches

def main(args):
    # Set up checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize wandb with resume support
    run_id = None
    if args.resume:
        # Try to get the run ID from the checkpoint
        checkpoint_path = checkpoint_dir / ('latest_model.pt' if not args.best else 'best_model.pt')
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            if 'wandb_run_id' in checkpoint:
                run_id = checkpoint['wandb_run_id']
                logging.info(f"Resuming wandb run: {run_id}")

    # Initialize wandb
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=args.__dict__,
        resume="allow" if args.resume else None,
        id=run_id,
        settings=wandb.Settings(start_method="fork")
    )
    
    # Set device and enable benchmarking
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
        torch.backends.cuda.matmul.allow_tf32 = True  # For newer NVIDIA GPUs
        torch.backends.cudnn.allow_tf32 = True
    logging.info(f"Using device: {device}")
    
    # Create model
    model = ContrastiveModel(
        model_name=args.model_name,
        pretrained=args.pretrained
    ).to(device)
    
    # Wrap model in DistributedDataParallel if using multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Create optimizer with weight decay
    param_groups = [
        {'params': [], 'weight_decay': 0.0},  # no weight decay for biases and norm layers
        {'params': [], 'weight_decay': args.weight_decay}  # weight decay for other params
    ]
    
    for name, param in model.named_parameters():
        if 'bias' in name or 'bn' in name or 'norm' in name:
            param_groups[0]['params'].append(param)
        else:
            param_groups[1]['params'].append(param)
    
    optimizer = AdamW(param_groups, lr=args.learning_rate)
    
    # Create warmup + cosine scheduler
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, 
                           schedulers=[warmup_scheduler, cosine_scheduler],
                           milestones=[args.warmup_epochs])
    
    # Initialize mixed precision training
    scaler = amp.GradScaler('cuda') if args.mixed_precision else None
    
    # Initialize EMA
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        checkpoint_path = checkpoint_dir / ('latest_model.pt' if not args.best else 'best_model.pt')
        start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        logging.info(f"Resuming from epoch {start_epoch} with best loss: {best_loss:.6f}")
    
    # Get data loader with specified number of views
    if args.dataset == 'bloch':
        loader = get_bloch_loader(
            batch_size=args.batch_size,
            same_contrast=args.same_contrast,
            num_views=args.num_views,
            num_workers=args.num_workers,
            pin_memory=True,
            pin_memory_device=f'cuda:{args.gpu_id}',
            persistent_workers=True,
            prefetch_factor=2,
        )
    else:
        loader = get_mprage_loader(
            batch_size=args.batch_size,
            num_views=args.num_views,
            num_workers=args.num_workers,
            pin_memory=True,
            pin_memory_device=f'cuda:{args.gpu_id}',
            persistent_workers=True,
            prefetch_factor=2,
        )
    
    # After model creation
    model = torch.compile(model)  # Uses TorchDynamo for optimization
    
    # Training loop
    try:
        for epoch in range(start_epoch, args.epochs):
            loss = train_epoch(model, loader, optimizer, device, epoch, scaler, ema, args.loss_type)
            scheduler.step()
            
            # Use EMA model for evaluation and saving
            if ema is not None:
                ema.apply_shadow()
            
            # Log epoch metrics
            wandb.log({
                'epoch_loss': loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            # Save checkpoints
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'ema_shadow': ema.shadow if ema is not None else None,
                'loss': loss,
                'best_loss': best_loss,
                'args': args.__dict__,
                'wandb_run_id': run.id  # Save the run ID
            }, is_best, checkpoint_dir)
            
            if ema is not None:
                ema.restore()
            
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.exception("Training failed with exception")
        raise
    finally:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500,
                      help='Number of epochs to train')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                      help='Number of epochs for learning rate warmup')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--dataset', type=str, choices=['bloch', 'mprage'], default='bloch')
    parser.add_argument('--same_contrast', action='store_true',
                      help='For Bloch data, use same contrast for both views')
    parser.add_argument('--model_name', type=str, default='resnet18',
                      help='Model name from timm library')
    parser.add_argument('--pretrained', action='store_true',
                      help='Use pretrained weights')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from checkpoint')
    parser.add_argument('--best', action='store_true',
                      help='Resume from best checkpoint instead of latest')
    parser.add_argument('--mixed_precision', action='store_true',
                      help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--num_views', type=int, default=2,
                      help='Number of views for contrastive learning')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help='Number of steps to accumulate gradients')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                      help='Maximum gradient norm for clipping')
    parser.add_argument('--use_ema', action='store_true',
                      help='Use Exponential Moving Average of model weights')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                      help='EMA decay rate')
    parser.add_argument('--loss_type', type=str, default='nt_xent',
                      choices=['nt_xent', 'vicreg', 'barlow'],
                      help='Type of loss function to use')
    parser.add_argument('--wandb_project', type=str, default="mri-ssl",
                      help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                      help='WandB entity (username or team name)')
    parser.add_argument('--wandb_name', type=str, default=None,
                      help='WandB run name')
    parser.add_argument('--gpu_id', type=int, default=0,
                      help='GPU ID to use for training')
    
    args = parser.parse_args()
    main(args) 