#!/usr/bin/env python3
"""
Simple training script for cross-modal projection heads.
All configuration is done through config.yaml file.

Usage:
    python train.py [config_file]
    
Examples:
    python train.py                    # Uses config.yaml
    python train.py my_config.yaml     # Uses custom config
"""

import sys
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Import model and loss definitions
from models import create_model
from losses import compute_loss


class EmbeddingDataset(Dataset):
    """Simple dataset for pre-encoded embeddings."""
    def __init__(self, image_path, text_path):
        self.image_embeddings = torch.load(image_path, map_location='cpu')
        self.text_embeddings = torch.load(text_path, map_location='cpu')
        assert len(self.image_embeddings) == len(self.text_embeddings)
        
    def __len__(self):
        return len(self.image_embeddings)
    
    def __getitem__(self, idx):
        return self.image_embeddings[idx], self.text_embeddings[idx]


# Model definitions moved to models.py


def plot_training_curves(train_losses, val_losses, output_dir):
    """Plot and save training curves."""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add best epoch marker
    best_epoch = val_losses.index(min(val_losses)) + 1
    best_val_loss = min(val_losses)
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, 
                label=f'Best Epoch ({best_epoch})')
    plt.scatter([best_epoch], [best_val_loss], color='green', s=100, zorder=5)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {plot_path}")
    
    # Also save loss history as JSON for later analysis
    loss_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }
    
    with open(output_dir / 'loss_history.json', 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    print(f"Loss history saved to: {output_dir / 'loss_history.json'}")



def train_epoch(image_head, text_head, train_loader, optimizer, device, config, is_cross_modal, pin_memory):
    """Train for one epoch."""
    if is_cross_modal:
        image_head.train()
    else:
        image_head.train()
        text_head.train()
    
    total_loss = 0
    for image_emb, text_emb in tqdm(train_loader, desc="Training"):
        # Convert to float32 to ensure dtype compatibility with projection heads
        image_emb = image_emb.to(device, dtype=torch.float32, non_blocking=pin_memory)
        text_emb = text_emb.to(device, dtype=torch.float32, non_blocking=pin_memory)
        
        # Forward pass
        if is_cross_modal:
            image_proj, text_proj = image_head(image_emb, text_emb)
        else:
            image_proj = image_head(image_emb)
            text_proj = text_head(text_emb)
        
        # Handle BLIP momentum features if needed
        momentum_image_proj = None
        momentum_text_proj = None
        if config['model']['type'] == 'blip' and config['loss'].get('alpha', 0) > 0:
            momentum_image_proj = image_head(image_emb, use_momentum=True)
            momentum_text_proj = text_head(text_emb, use_momentum=True)
        
        # Compute loss
        loss = compute_loss(image_proj, text_proj, config, image_head, text_head,
                          momentum_image_proj, momentum_text_proj)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            image_head.parameters() if is_cross_modal else 
            list(image_head.parameters()) + list(text_head.parameters()),
            float(config['training']['max_grad_norm'])
        )
        optimizer.step()
        
        # Update BLIP momentum networks
        if config['model']['type'] == 'blip':
            if not is_cross_modal:
                image_head.update_momentum()
                text_head.update_momentum()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


@torch.no_grad()
def validate(image_head, text_head, val_loader, device, config, is_cross_modal, pin_memory):
    """Validate for one epoch."""
    if is_cross_modal:
        image_head.eval()
    else:
        image_head.eval()
        text_head.eval()
    
    total_loss = 0
    for image_emb, text_emb in tqdm(val_loader, desc="Validation"):
        # Convert to float32 to ensure dtype compatibility with projection heads
        image_emb = image_emb.to(device, dtype=torch.float32, non_blocking=pin_memory)
        text_emb = text_emb.to(device, dtype=torch.float32, non_blocking=pin_memory)
        
        # Forward pass
        if is_cross_modal:
            image_proj, text_proj = image_head(image_emb, text_emb)
        else:
            image_proj = image_head(image_emb)
            text_proj = text_head(text_emb)
        
        # Handle BLIP momentum features if needed (for validation)
        momentum_image_proj = None
        momentum_text_proj = None
        if config['model']['type'] == 'blip' and config['loss'].get('alpha', 0) > 0:
            momentum_image_proj = image_head(image_emb, use_momentum=True)
            momentum_text_proj = text_head(text_emb, use_momentum=True)
        
        # Compute loss
        loss = compute_loss(image_proj, text_proj, config, image_head, text_head,
                          momentum_image_proj, momentum_text_proj)
        total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from: {config_path}")
    print(f"Experiment: {config['experiment_name']}")
    
    # Set seed
    torch.manual_seed(config['seed'])
    
    # Setup device
    if config['device'] == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using device: {device} ({torch.cuda.get_device_name()})")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
            print(f"Using device: {device} (Apple Silicon GPU)")
        else:
            device = torch.device('cpu')
            print(f"Using device: {device}")
    else:
        device = torch.device(config['device'])
        print(f"Using device: {device}")
    
    # Validate data splits are available
    required_files = [
        'train_image_embeddings', 'train_text_embeddings',
        'val_image_embeddings', 'val_text_embeddings'
    ]
    missing_files = [f for f in required_files if f not in config['data']]
    if missing_files:
        print(f"Error: Missing required data files in config: {missing_files}")
        print("Please run 'python split_train_data.py' first to create proper data splits.")
        sys.exit(1)
    
    # Create datasets
    print("Using training split for training (~95K samples)")
    print("Using validation split for validation (~5K samples)")
    train_dataset = EmbeddingDataset(
        config['data']['train_image_embeddings'],
        config['data']['train_text_embeddings']
    )
    val_dataset = EmbeddingDataset(
        config['data']['val_image_embeddings'],
        config['data']['val_text_embeddings']
    )
    
    # Create data loaders with device-optimized settings
    pin_memory = device.type == 'cuda'  # Only use pin_memory for CUDA
    num_workers = 4 if device.type != 'mps' else 0  # MPS works better with num_workers=0
    
    # Set consistent dtype for MPS compatibility
    if device.type == 'mps':
        torch.backends.mps.fallback_enabled = True  # Enable fallback for unsupported ops
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Get dimensions
    image_dim = train_dataset.image_embeddings.shape[1]
    text_dim = train_dataset.text_embeddings.shape[1]
    print(f"Image dim: {image_dim}, Text dim: {text_dim}")
    
    # Create model
    image_head, text_head, is_cross_modal = create_model(config, image_dim, text_dim)
    
    # Move to device with consistent dtype for MPS
    if device.type == 'mps':
        # Force float32 for MPS compatibility
        image_head = image_head.to(device, dtype=torch.float32)
        if text_head is not None:
            text_head = text_head.to(device, dtype=torch.float32)
    else:
        image_head = image_head.to(device)
        if text_head is not None:
            text_head = text_head.to(device)
    
    # Create optimizer
    if is_cross_modal:
        params = image_head.parameters()
    else:
        params = list(image_head.parameters()) + list(text_head.parameters())
    
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # Training loop - create config-specific directory
    config_name = Path(config_path).stem  # Extract filename without extension
    output_dir = Path(config['output_dir']) / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Track losses for plotting
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_loss = train_epoch(image_head, text_head, train_loader, optimizer, device, config, is_cross_modal, pin_memory)
        
        # Validate
        val_loss = validate(image_head, text_head, val_loader, device, config, is_cross_modal, pin_memory)
        
        # Store losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Log epoch results with optional temperature for CLIP
        log_msg = f"Epoch {epoch+1}/{config['training']['num_epochs']} - Train: {train_loss:.4f}, Val: {val_loss:.4f}"
        
        if config['model']['type'] == 'clip' and not is_cross_modal:
            # Log learned temperature for CLIP
            if hasattr(image_head, 'get_temperature'):
                temp = image_head.get_temperature()
                log_msg += f", Temp: {temp:.3f}"
        
        print(log_msg)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'config': config,
                'image_head_state': image_head.state_dict(),
                'val_loss': val_loss,
                'is_cross_modal': is_cross_modal
            }
            if not is_cross_modal:
                checkpoint['text_head_state'] = text_head.state_dict()
            
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"New best model saved! Val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config['training']['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Training completed! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    
    # Plot and save training curves
    print("Generating training curves...")
    plot_training_curves(train_losses, val_losses, output_dir)


if __name__ == "__main__":
    main()