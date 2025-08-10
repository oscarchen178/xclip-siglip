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
from dataset import create_train_dataset, create_eval_dataset
from metrics import compute_validation_metrics




# Model definitions moved to models.py


def plot_training_curves(train_losses, val_losses, val_r1_scores, output_dir):
    """Plot and save training curves with loss and R@1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training and validation loss together
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add best loss epoch marker (lower is better)
    best_loss_epoch = val_losses.index(min(val_losses)) + 1
    best_val_loss = min(val_losses)
    ax1.axvline(x=best_loss_epoch, color='orange', linestyle='--', alpha=0.7, 
                label=f'Best Loss Epoch ({best_loss_epoch})')
    ax1.scatter([best_loss_epoch], [best_val_loss], color='orange', s=100, zorder=5)
    ax1.legend(fontsize=12)
    
    # Plot validation R@1
    ax2.plot(epochs, val_r1_scores, 'g-', label='Validation R@1', linewidth=2)
    ax2.set_title('Validation R@1 (Early Stopping)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('R@1 Score', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add best R@1 epoch marker (higher is better) - this is used for early stopping
    best_r1_epoch = val_r1_scores.index(max(val_r1_scores)) + 1
    best_val_r1 = max(val_r1_scores)
    ax2.axvline(x=best_r1_epoch, color='green', linestyle='--', alpha=0.7, 
                label=f'Best R@1 Epoch ({best_r1_epoch})')
    ax2.scatter([best_r1_epoch], [best_val_r1], color='green', s=100, zorder=5)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {plot_path}")
    
    # Also save training history as JSON for later analysis
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_r1_scores': val_r1_scores,
        'best_r1_epoch': best_r1_epoch,
        'best_val_r1': best_val_r1,
        'best_loss_epoch': best_loss_epoch,
        'best_val_loss': best_val_loss
    }
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Training history saved to: {output_dir / 'training_history.json'}")



def train_epoch(image_head, text_head, train_loader, optimizer, device, config, pin_memory):
    """Train for one epoch."""
    image_head.train()
    text_head.train()
    
    total_loss = 0
    for image_emb, text_emb, img_ids in tqdm(train_loader, desc="Training"):
        # Convert to float32 to ensure dtype compatibility with projection heads
        image_emb = image_emb.to(device, dtype=torch.float32, non_blocking=pin_memory)
        text_emb = text_emb.to(device, dtype=torch.float32, non_blocking=pin_memory)
        
        # Forward pass (simplified, no cross-modal or BLIP support)
        image_proj = image_head(image_emb)
        text_proj = text_head(text_emb)
        
        # Compute loss with image IDs for queue-based losses
        loss = compute_loss(image_proj, text_proj, config, image_head, text_head, img_ids=img_ids)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(image_head.parameters()) + list(text_head.parameters()),
            float(config['training']['max_grad_norm'])
        )
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


@torch.no_grad()
def validate(image_head, text_head, val_loader, device, config, pin_memory):
    """Validate for one epoch."""
    image_head.eval()
    text_head.eval()
    
    total_loss = 0
    for image_emb, text_emb, img_ids in tqdm(val_loader, desc="Validation"):
        # Convert to float32 to ensure dtype compatibility with projection heads
        image_emb = image_emb.to(device, dtype=torch.float32, non_blocking=pin_memory)
        text_emb = text_emb.to(device, dtype=torch.float32, non_blocking=pin_memory)
        
        # Forward pass (simplified)
        image_proj = image_head(image_emb)
        text_proj = text_head(text_emb)
        
        # Compute loss with image IDs
        loss = compute_loss(image_proj, text_proj, config, image_head, text_head, img_ids=img_ids)
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
    base_dir = config.get('base_data_dir', '../pretrain_encoded')  # Fallback for legacy configs
    train_dataset = create_train_dataset(
        base_dir,
        config['data']['train_image_embeddings'],
        config['data']['train_text_embeddings'],
        config['data']['train_metadata']
    )
    val_dataset = create_eval_dataset(
        base_dir, 
        config['data']['val_image_embeddings'],
        config['data']['val_text_embeddings'],
        config['data']['val_metadata']
    )
    
    # Create data loaders with device-optimized settings
    pin_memory = device.type == 'cuda'  # Only use pin_memory for CUDA
    num_workers = 4 if device.type != 'mps' else 0  # MPS works better with num_workers=0
    
    # Force float32 for all devices for consistency
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
    image_head, text_head = create_model(config, image_dim, text_dim)
    
    # Move to device with consistent float32 dtype
    image_head = image_head.to(device, dtype=torch.float32)
    text_head = text_head.to(device, dtype=torch.float32)
    
    # Create optimizer including loss function parameters if learnable scale is enabled
    params = list(image_head.parameters()) + list(text_head.parameters())
    
    # Add loss function parameters if using learnable temperature
    if config['loss'].get('learnable_scale', False):
        from losses import get_loss_instance
        loss_fn = get_loss_instance(config['loss'].get('type', 'sigmoid_infonce'), config, device)
        params.extend(list(loss_fn.parameters()))
        print(f"Added {len(list(loss_fn.parameters()))} loss function parameters to optimizer")
    
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # Training loop - create config-specific directory
    config_name = Path(config_path).stem  # Extract filename without extension
    output_dir = Path(config['output_dir']) / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_r1 = 0.0  # R@1 metric, higher is better
    patience_counter = 0
    
    # Track metrics for plotting
    train_losses = []
    val_losses = []
    val_r1_scores = []
    
    print("Starting training...")
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_loss = train_epoch(image_head, text_head, train_loader, optimizer, device, config, pin_memory)
        
        # Validate: compute both loss (for monitoring) and R@1 (for early stopping)
        val_loss = validate(image_head, text_head, val_loader, device, config, pin_memory)
        val_metrics = compute_validation_metrics(image_head, text_head, val_loader, device, fast_mode=True)
        val_r1 = val_metrics['val_recall_1']
        
        # Store metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r1_scores.append(val_r1)
        
        # Log epoch results with both loss and R@1, plus optional temperature if learnable
        log_msg = f"Epoch {epoch+1}/{config['training']['num_epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R@1: {val_r1:.4f}"
        
        # Log learnable temperature if enabled
        if config['loss'].get('learnable_scale', False):
            from losses import get_loss_instance
            loss_fn = get_loss_instance(config['loss'].get('type', 'sigmoid_infonce'), config, device)
            temp = loss_fn.get_temperature()
            log_msg += f", Temp: {temp:.3f}"
        
        print(log_msg)
        
        # Save best model (higher R@1 is better)
        if val_r1 > best_val_r1:
            best_val_r1 = val_r1
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'config': config,
                'image_head_state': image_head.state_dict(),
                'text_head_state': text_head.state_dict(),
                'val_r1': val_r1
            }
            
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"New best model saved! Val R@1: {val_r1:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config['training']['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Training completed! Best val R@1: {best_val_r1:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    
    # Plot and save training curves
    print("Generating training curves...")
    plot_training_curves(train_losses, val_losses, val_r1_scores, output_dir)


if __name__ == "__main__":
    main()