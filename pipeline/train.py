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


class MLPProjectionHead(nn.Module):
    """MLP projection head."""
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)


class AttentionProjectionHead(nn.Module):
    """Attention-based projection head."""
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        # x: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        for attn, norm in zip(self.attention_layers, self.layer_norms):
            attn_out, _ = attn(x, x, x)
            x = norm(x + attn_out)
        
        # Remove sequence dimension and project
        x = x.squeeze(1)
        output = self.final_proj(x)
        return F.normalize(output, dim=-1)


class CrossModalProjectionHead(nn.Module):
    """Cross-modal projection head."""
    def __init__(self, image_dim, text_dim, output_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        
        # Project to common space
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention
        self.img_to_txt_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.txt_to_img_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Layer norms
        self.img_norm = nn.LayerNorm(hidden_dim)
        self.txt_norm = nn.LayerNorm(hidden_dim)
        
        # Final projections
        self.img_final = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))
        self.txt_final = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))
    
    def forward(self, image_emb, text_emb):
        # Project and add sequence dimension
        img_proj = self.image_proj(image_emb).unsqueeze(1)  # (B, 1, H)
        txt_proj = self.text_proj(text_emb).unsqueeze(1)    # (B, 1, H)
        
        # Cross-attention
        img_attn, _ = self.img_to_txt_attn(img_proj, txt_proj, txt_proj)
        txt_attn, _ = self.txt_to_img_attn(txt_proj, img_proj, img_proj)
        
        # Residual + norm
        img_out = self.img_norm(img_proj + img_attn).squeeze(1)
        txt_out = self.txt_norm(txt_proj + txt_attn).squeeze(1)
        
        # Final projection
        img_final = F.normalize(self.img_final(img_out), dim=-1)
        txt_final = F.normalize(self.txt_final(txt_out), dim=-1)
        
        return img_final, txt_final


def siglip_loss(image_features, text_features, temperature=0.05):
    """SigLIP loss function."""
    batch_size = image_features.shape[0]
    
    # Compute similarity matrix
    logits = torch.matmul(image_features, text_features.T) / temperature
    
    # Create labels (positive pairs are on diagonal)
    labels = torch.arange(batch_size, device=logits.device)
    
    # Compute cross-entropy loss for both directions
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2


def create_model(config, image_dim, text_dim):
    """Create projection heads based on config."""
    model_config = config['model']
    model_type = model_config['type']
    
    if model_type == 'mlp':
        image_head = MLPProjectionHead(
            image_dim, 
            model_config['output_dim'],
            model_config['hidden_dim'],
            model_config['dropout']
        )
        text_head = MLPProjectionHead(
            text_dim,
            model_config['output_dim'], 
            model_config['hidden_dim'],
            model_config['dropout']
        )
        return image_head, text_head, False
        
    elif model_type == 'attention':
        image_head = AttentionProjectionHead(
            image_dim,
            model_config['output_dim'],
            model_config['hidden_dim'],
            model_config['num_heads'],
            model_config['num_layers'],
            model_config['dropout']
        )
        text_head = AttentionProjectionHead(
            text_dim,
            model_config['output_dim'],
            model_config['hidden_dim'], 
            model_config['num_heads'],
            model_config['num_layers'],
            model_config['dropout']
        )
        return image_head, text_head, False
        
    elif model_type == 'cross_modal':
        cross_head = CrossModalProjectionHead(
            image_dim,
            text_dim,
            model_config['output_dim'],
            model_config['hidden_dim'],
            model_config['num_heads'],
            model_config['dropout']
        )
        return cross_head, None, True
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(image_head, text_head, train_loader, optimizer, device, config, is_cross_modal, pin_memory):
    """Train for one epoch."""
    if is_cross_modal:
        image_head.train()
    else:
        image_head.train()
        text_head.train()
    
    total_loss = 0
    for image_emb, text_emb in tqdm(train_loader, desc="Training"):
        if device.type == 'mps':
            # Ensure float32 for MPS compatibility
            image_emb = image_emb.to(device, dtype=torch.float32)
            text_emb = text_emb.to(device, dtype=torch.float32)
        else:
            image_emb, text_emb = image_emb.to(device, non_blocking=pin_memory), text_emb.to(device, non_blocking=pin_memory)
        
        # Forward pass
        if is_cross_modal:
            image_proj, text_proj = image_head(image_emb, text_emb)
        else:
            image_proj = image_head(image_emb)
            text_proj = text_head(text_emb)
        
        # Compute loss
        loss = siglip_loss(image_proj, text_proj, float(config['loss']['temperature']))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            image_head.parameters() if is_cross_modal else 
            list(image_head.parameters()) + list(text_head.parameters()),
            float(config['training']['max_grad_norm'])
        )
        optimizer.step()
        
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
        if device.type == 'mps':
            # Ensure float32 for MPS compatibility
            image_emb = image_emb.to(device, dtype=torch.float32)
            text_emb = text_emb.to(device, dtype=torch.float32)
        else:
            image_emb, text_emb = image_emb.to(device, non_blocking=pin_memory), text_emb.to(device, non_blocking=pin_memory)
        
        # Forward pass
        if is_cross_modal:
            image_proj, text_proj = image_head(image_emb, text_emb)
        else:
            image_proj = image_head(image_emb)
            text_proj = text_head(text_emb)
        
        # Compute loss
        loss = siglip_loss(image_proj, text_proj, float(config['loss']['temperature']))
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
    
    print("Starting training...")
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_loss = train_epoch(image_head, text_head, train_loader, optimizer, device, config, is_cross_modal, pin_memory)
        
        # Validate
        val_loss = validate(image_head, text_head, val_loader, device, config, is_cross_modal, pin_memory)
        
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
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


if __name__ == "__main__":
    main()