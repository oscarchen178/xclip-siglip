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
    """CLIP-style projection head with learnable temperature scaling."""
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.1, learnable_temp=True):
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
        
        # Learnable temperature parameter (CLIP innovation)
        if learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * (1/0.07).log())  # Init to 1/0.07
        else:
            self.register_buffer('logit_scale', torch.tensor((1/0.07).log()))
    
    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)
    
    def get_temperature(self):
        """Get current temperature (for monitoring)."""
        return self.logit_scale.exp().item()


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


class BLIPProjectionHead(nn.Module):
    """BLIP-style projection head with momentum updates."""
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.1, momentum=0.999):
        super().__init__()
        
        # Main projection head
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Momentum projection head (for contrastive learning)
        self.momentum_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Initialize momentum projection with same weights
        for param_main, param_momentum in zip(self.projection.parameters(), 
                                            self.momentum_projection.parameters()):
            param_momentum.data.copy_(param_main.data)
            param_momentum.requires_grad = False  # No gradients for momentum network
        
        self.momentum = momentum
    
    def forward(self, x, use_momentum=False):
        if use_momentum:
            with torch.no_grad():
                return F.normalize(self.momentum_projection(x), dim=-1)
        else:
            return F.normalize(self.projection(x), dim=-1)
    
    @torch.no_grad()
    def update_momentum(self):
        """Update momentum projection head."""
        for param_main, param_momentum in zip(self.projection.parameters(), 
                                            self.momentum_projection.parameters()):
            param_momentum.data = param_momentum.data * self.momentum + param_main.data * (1. - self.momentum)


# Model Factory Function


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
        return image_head, text_head, False
    
    elif model_type == 'clip':
        image_head = CLIPProjectionHead(
            image_dim,
            model_config['output_dim'],
            model_config.get('hidden_dim', image_dim // 2),
            model_config.get('dropout', 0.1),
            model_config.get('learnable_temp', True)
        )
        text_head = CLIPProjectionHead(
            text_dim,
            model_config['output_dim'],
            model_config.get('hidden_dim', text_dim // 2),
            model_config.get('dropout', 0.1),
            model_config.get('learnable_temp', True)
        )
        return image_head, text_head, False
    
    elif model_type == 'align':
        image_head = ALIGNProjectionHead(
            image_dim,
            model_config['output_dim'],
            model_config['hidden_dim'],
            model_config.get('dropout', 0.1),
            model_config.get('use_bn', True)
        )
        text_head = ALIGNProjectionHead(
            text_dim,
            model_config['output_dim'],
            model_config['hidden_dim'],
            model_config.get('dropout', 0.1),
            model_config.get('use_bn', True)
        )
        return image_head, text_head, False
    
    elif model_type == 'blip':
        image_head = BLIPProjectionHead(
            image_dim,
            model_config['output_dim'],
            model_config['hidden_dim'],
            model_config.get('dropout', 0.1),
            model_config.get('momentum', 0.999)
        )
        text_head = BLIPProjectionHead(
            text_dim,
            model_config['output_dim'],
            model_config['hidden_dim'],
            model_config.get('dropout', 0.1),
            model_config.get('momentum', 0.999)
        )
        return image_head, text_head, False
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: mlp, attention, cross_modal, siglip, clip, align, blip")