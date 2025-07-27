#!/usr/bin/env python3
"""
Model definitions for cross-modal projection heads.

This module contains all projection head architectures and loss functions
used in the cross-modal alignment pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# Loss Functions

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