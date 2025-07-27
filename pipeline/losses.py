#!/usr/bin/env python3
"""
Loss functions for cross-modal projection head training.

This module contains all loss function implementations for different
SOTA methods (SigLIP, CLIP, ALIGN, BLIP, etc.)
"""

import torch
import torch.nn.functional as F


def siglip_loss(image_features, text_features, temperature=0.05):
    """Original SigLIP loss function with sigmoid (not our current implementation)."""
    batch_size = image_features.shape[0]
    
    # Compute similarity matrix
    logits = torch.matmul(image_features, text_features.T) / temperature
    
    # Create labels: 1 for positive pairs (diagonal), -1 for negatives
    labels = torch.eye(batch_size, device=logits.device) * 2 - 1  # Diagonal=1, others=-1
    
    # SigLIP uses sigmoid loss instead of softmax
    # Loss = -log(sigmoid(labels * logits))
    sigmoid_loss = -F.logsigmoid(labels * logits)
    
    # Average over all pairs
    return sigmoid_loss.mean()


def siglip_contrastive_loss(image_features, text_features, temperature=0.05):
    """SigLIP-style contrastive loss (our current implementation for compatibility)."""
    batch_size = image_features.shape[0]
    
    # Compute similarity matrix
    logits = torch.matmul(image_features, text_features.T) / temperature
    
    # Create labels (positive pairs are on diagonal)
    labels = torch.arange(batch_size, device=logits.device)
    
    # Compute cross-entropy loss for both directions
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2


def clip_loss(image_features, text_features, logit_scale):
    """CLIP contrastive loss with learnable temperature."""
    batch_size = image_features.shape[0]
    
    # Apply learnable temperature scaling
    logits = torch.matmul(image_features, text_features.T) * logit_scale.exp()
    
    # Create labels (positive pairs are on diagonal)
    labels = torch.arange(batch_size, device=logits.device)
    
    # Compute cross-entropy loss for both directions
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2


def align_loss(image_features, text_features, temperature=0.05):
    """ALIGN dual encoder loss (similar to CLIP but with different scaling)."""
    batch_size = image_features.shape[0]
    
    # Compute similarity matrix with fixed temperature
    logits = torch.matmul(image_features, text_features.T) / temperature
    
    # Create labels
    labels = torch.arange(batch_size, device=logits.device)
    
    # ALIGN uses standard contrastive loss
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2


def blip_loss(image_features, text_features, momentum_image_features, momentum_text_features, 
              temperature=0.05, alpha=0.4):
    """BLIP momentum contrastive loss with larger effective batch size."""
    batch_size = image_features.shape[0]
    
    # Compute similarities with momentum features for larger effective batch
    sim_i2t_m = torch.matmul(image_features, momentum_text_features.T) / temperature
    sim_t2i_m = torch.matmul(text_features, momentum_image_features.T) / temperature
    
    # Standard contrastive loss
    sim_i2t = torch.matmul(image_features, text_features.T) / temperature
    sim_t2i = torch.matmul(text_features, image_features.T) / temperature
    
    # Labels for current batch
    labels = torch.arange(batch_size, device=image_features.device)
    
    # Main contrastive loss
    loss_i2t = F.cross_entropy(sim_i2t, labels)
    loss_t2i = F.cross_entropy(sim_t2i, labels)
    main_loss = (loss_i2t + loss_t2i) / 2
    
    # Momentum augmented loss (optional, controlled by alpha)
    if alpha > 0 and momentum_image_features is not None:
        # Create extended labels for momentum features
        momentum_batch_size = momentum_image_features.shape[0]
        extended_labels = torch.arange(momentum_batch_size, device=image_features.device)
        
        # Only use current batch indices as positive labels
        valid_indices = extended_labels < batch_size
        if valid_indices.sum() > 0:
            loss_i2t_m = F.cross_entropy(sim_i2t_m[:, valid_indices], labels)
            loss_t2i_m = F.cross_entropy(sim_t2i_m[:, valid_indices], labels)
            momentum_loss = (loss_i2t_m + loss_t2i_m) / 2
            
            # Combine main and momentum losses
            return (1 - alpha) * main_loss + alpha * momentum_loss
    
    return main_loss


def compute_loss(image_proj, text_proj, config, image_head=None, text_head=None, 
                momentum_image_proj=None, momentum_text_proj=None):
    """Compute loss based on configuration.
    
    Args:
        image_proj: Image projection features
        text_proj: Text projection features  
        config: Training configuration
        image_head: Image projection head (for CLIP temperature)
        text_head: Text projection head (for CLIP temperature)
        momentum_image_proj: Momentum image features (for BLIP)
        momentum_text_proj: Momentum text features (for BLIP)
    
    Returns:
        Computed loss value
    """
    loss_type = config['loss'].get('type', 'siglip')
    
    if loss_type == 'siglip':
        return siglip_loss(image_proj, text_proj, float(config['loss']['temperature']))
    
    elif loss_type == 'siglip_contrastive':
        return siglip_contrastive_loss(image_proj, text_proj, float(config['loss']['temperature']))
    
    elif loss_type == 'clip':
        # Use learnable temperature from model
        if hasattr(image_head, 'logit_scale'):
            logit_scale = image_head.logit_scale
        elif hasattr(text_head, 'logit_scale'):
            logit_scale = text_head.logit_scale
        else:
            # Fallback to fixed temperature
            temp = float(config['loss']['temperature'])
            logit_scale = torch.tensor(1.0 / temp).log()
        return clip_loss(image_proj, text_proj, logit_scale)
    
    elif loss_type == 'align':
        return align_loss(image_proj, text_proj, float(config['loss']['temperature']))
    
    elif loss_type == 'blip':
        alpha = float(config['loss'].get('alpha', 0.4))
        return blip_loss(image_proj, text_proj, momentum_image_proj, momentum_text_proj,
                        float(config['loss']['temperature']), alpha)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")