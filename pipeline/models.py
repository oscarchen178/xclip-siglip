#!/usr/bin/env python3
"""
Model definitions for cross-modal projection heads.

This module contains all projection head architectures and loss functions
used in the cross-modal alignment pipeline.
"""

import math
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




# SOTA Projection Heads

class SigLIPProjectionHead(nn.Module):
    """SigLIP projection head - simple linear projection with layer norm."""
    def __init__(self, input_dim, output_dim, dropout=0.0):
        super().__init__()
        
        # SigLIP uses a simple architecture: LayerNorm -> Dropout -> Linear
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)  # No bias in SigLIP
        
        # Initialize projection layer
        nn.init.normal_(self.projection.weight, std=input_dim ** -0.5)
    
    def forward(self, x):
        # SigLIP: LayerNorm -> Dropout -> Linear -> L2 Normalize
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.projection(x)
        return F.normalize(x, dim=-1)

class CLIPProjectionHead(nn.Module):
    """CLIP-style projection head - simple MLP projection."""
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        # Simple projection (CLIP uses single linear layer)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)


class ALIGNProjectionHead(nn.Module):
    """ALIGN-style dual encoder with batch normalization and larger capacity."""
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.1, use_bn=True):
        super().__init__()
        
        layers = [
            nn.Linear(input_dim, hidden_dim),
        ]
        
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        else:
            layers.append(nn.LayerNorm(hidden_dim))
            
        layers.extend([
            nn.ReLU(inplace=True),  # ALIGN uses ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        ])
        
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim // 2))
        else:
            layers.append(nn.LayerNorm(hidden_dim // 2))
            
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        ])
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)




# Model Factory Function


def create_model(config, image_dim, text_dim):
    """Create projection heads based on config.
    
    Returns:
        tuple: (image_head, text_head) - Two separate projection heads
    """
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
        return image_head, text_head
        
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
        return image_head, text_head
    
    elif model_type == 'siglip':
        image_head = SigLIPProjectionHead(
            image_dim,
            model_config['output_dim'],
            model_config.get('dropout', 0.0)
        )
        text_head = SigLIPProjectionHead(
            text_dim,
            model_config['output_dim'],
            model_config.get('dropout', 0.0)
        )
        return image_head, text_head
    
    elif model_type == 'clip':
        image_head = CLIPProjectionHead(
            image_dim,
            model_config['output_dim'],
            model_config.get('hidden_dim', image_dim // 2),
            model_config.get('dropout', 0.1)
        )
        text_head = CLIPProjectionHead(
            text_dim,
            model_config['output_dim'],
            model_config.get('hidden_dim', text_dim // 2),
            model_config.get('dropout', 0.1)
        )
        return image_head, text_head
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: mlp, attention, siglip, clip")