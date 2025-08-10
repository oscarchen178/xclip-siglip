#!/usr/bin/env python3
"""
Loss functions for cross-modal projection head training.

This module contains all loss function implementations for different
SOTA methods with proper InfoNCE variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SigmoidInfoNCELoss(nn.Module):
    """Sigmoid InfoNCE loss (SigLIP) - uses binary cross-entropy."""
    def __init__(self, tau: float = 0.07, learnable_scale: bool = False):
        super().__init__()
        self.learnable_scale = learnable_scale
        if not learnable_scale:
            self.tau = tau
        else:
            # Create learnable temperature parameter in the loss function
            self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1/tau))  # Init to 1/tau

    def forward(self, img_z, txt_z, img_ids):
        # img_ids is now mandatory for proper COCO multi-caption handling
        if img_ids is None:
            raise ValueError("img_ids is required for SigmoidInfoNCELoss - COCO has multiple captions per image")
        
        # Normalize features
        img_z = F.normalize(img_z, dim=-1)
        txt_z = F.normalize(txt_z, dim=-1)
        
        # Determine temperature to use
        if self.learnable_scale:
            tau = 1.0 / self.logit_scale.exp()  # Use loss function's own parameter
        else:
            tau = self.tau  # Use fixed temperature
        
        # Similarity matrix
        logits = img_z @ txt_z.t() / tau                 # B × B
        
        # Create multi-label target matrix using image IDs (vectorized)
        batch_size = img_z.size(0)
        img_ids_tensor = torch.as_tensor(img_ids, device=img_z.device, dtype=torch.long)
        
        # Broadcast comparison: [B, 1] == [1, B] -> [B, B] 
        target = (img_ids_tensor.unsqueeze(1) == img_ids_tensor.unsqueeze(0)).float()
        
        # Binary cross-entropy with logits (sigmoid InfoNCE)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        
        return loss, logits.detach(), target
    
    def get_temperature(self):
        """Get current temperature (for monitoring)."""
        if self.learnable_scale:
            return 1.0 / self.logit_scale.exp().item()
        else:
            return self.tau


class SoftmaxInfoNCELoss(nn.Module):
    """Softmax InfoNCE loss (CLIP) - uses cross-entropy."""
    def __init__(self, tau: float = 0.07, learnable_scale: bool = False):
        super().__init__()
        self.learnable_scale = learnable_scale
        if not learnable_scale:
            self.tau = tau
        else:
            # Create learnable temperature parameter in the loss function
            self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1/tau))  # Init to 1/tau

    def forward(self, img_z, txt_z, img_ids):
        # img_ids is now mandatory for proper COCO multi-caption handling
        if img_ids is None:
            raise ValueError("img_ids is required for SoftmaxInfoNCELoss - COCO has multiple captions per image")
        
        # Normalize features
        img_z = F.normalize(img_z, dim=-1)
        txt_z = F.normalize(txt_z, dim=-1)
        
        # Determine temperature to use
        if self.learnable_scale:
            tau = 1.0 / self.logit_scale.exp()  # Use loss function's own parameter
        else:
            tau = self.tau  # Use fixed temperature
        
        # Similarity matrix
        logits = img_z @ txt_z.t() / tau                  # B × B
        
        # For softmax InfoNCE with multi-caption data, we need to handle multiple positives
        # Create mask for positive pairs (same image_id) - vectorized
        batch_size = img_z.size(0)
        img_ids_tensor = torch.as_tensor(img_ids, device=img_z.device, dtype=torch.long)
        
        # Broadcast comparison: [B, 1] == [1, B] -> [B, B]
        pos_mask = (img_ids_tensor.unsqueeze(1) == img_ids_tensor.unsqueeze(0))
        
        # For each query, we need to handle multiple positives in softmax
        # Use multi-label soft targets instead of hard labels
        target_probs = pos_mask.float()
        # Normalize so each row sums to 1 (convert to probability distribution)
        row_sums = target_probs.sum(dim=1, keepdim=True)
        target_probs = target_probs / row_sums.clamp(min=1e-8)
        
        # KL divergence loss for I2T (image to text)
        log_probs_i2t = F.log_softmax(logits, dim=1)
        loss_i2t = -torch.sum(target_probs * log_probs_i2t) / batch_size
        
        # KL divergence loss for T2I (text to image) 
        target_probs_t2i = target_probs.t()
        row_sums_t2i = target_probs_t2i.sum(dim=1, keepdim=True)
        target_probs_t2i = target_probs_t2i / row_sums_t2i.clamp(min=1e-8)
        
        log_probs_t2i = F.log_softmax(logits.t(), dim=1)
        loss_t2i = -torch.sum(target_probs_t2i * log_probs_t2i) / batch_size
        
        loss = 0.5 * (loss_i2t + loss_t2i)
        
        return loss, logits.detach(), pos_mask
    
    def get_temperature(self):
        """Get current temperature (for monitoring)."""
        if self.learnable_scale:
            return 1.0 / self.logit_scale.exp().item()
        else:
            return self.tau


class EmbQueue:
    """FIFO queue that stores text embeddings and their image_ids."""
    def __init__(self, dim: int, size: int = 4096, device="cuda"):
        self.emb = torch.zeros(size, dim, device=device)
        self.ids = torch.full((size,), -1, device=device, dtype=torch.long)
        self.ptr = 0
        self.size = size

    @torch.no_grad()
    def enqueue(self, y, ids):
        n, Q = y.size(0), self.emb.size(0)
        if n > Q:                      # trim oversize batch
            y, ids = y[-Q:], ids[-Q:]
            n = Q
        end = (self.ptr + n) % Q
        if end < self.ptr:             # wrap-around
            first = Q - self.ptr
            self.emb = torch.cat([
                self.emb[:self.ptr],
                y[:first].detach(),
                y[first:].detach(),
                self.emb[end:]
            ], dim=0)[:Q]
            self.ids = torch.cat([
                self.ids[:self.ptr],
                ids[:first].detach(),
                ids[first:].detach(),
                self.ids[end:]
            ], dim=0)[:Q]
        else:
            self.emb = torch.cat([
                self.emb[:self.ptr],
                y.detach(),
                self.emb[end:]
            ], dim=0)[:Q]
            self.ids = torch.cat([
                self.ids[:self.ptr],
                ids.detach(),
                self.ids[end:]
            ], dim=0)[:Q]
        self.ptr = end
    
    def to(self, device):
        """Move queue to specified device."""
        self.emb = self.emb.to(device)
        self.ids = self.ids.to(device)
        return self


class QueueInfoNCELoss(nn.Module):
    """Queue-based InfoNCE loss with hard negatives and image ID awareness."""
    def __init__(self, queue_size: int = 4096, tau: float = 0.07, feature_dim: int = 512, learnable_scale: bool = False):
        super().__init__()
        self.learnable_scale = learnable_scale
        if not learnable_scale:
            self.tau = tau
        else:
            # Create learnable temperature parameter in the loss function
            self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1/tau))  # Init to 1/tau
        self.queue = None
        self.queue_size = queue_size
        self.feature_dim = feature_dim

    def _init_queue(self, device, feature_dim):
        """Initialize queue when first called."""
        self.queue = EmbQueue(dim=feature_dim, size=self.queue_size, device=device)
        self.feature_dim = feature_dim

    def forward(self, img_z, txt_z, img_ids):
        # Normalize features
        img_z = F.normalize(img_z, dim=-1)
        txt_z = F.normalize(txt_z, dim=-1)
        
        # Ensure img_ids is a tensor of dtype long on the same device
        img_ids = torch.as_tensor(img_ids, device=img_z.device, dtype=torch.long)
        
        # Determine temperature to use
        if self.learnable_scale:
            tau = 1.0 / self.logit_scale.exp()  # Use loss function's own parameter
        else:
            tau = self.tau  # Use fixed temperature
        
        # Initialize queue if needed or if feature dimension has changed
        if self.queue is None or self.queue.emb.size(1) != img_z.size(1):
            self._init_queue(img_z.device, img_z.size(1))
        
        # Move queue to correct device if needed
        if self.queue.emb.device != img_z.device:
            self.queue.to(img_z.device)

        # In-batch logits
        logits = img_z @ txt_z.t() / tau                  # B × B
        labels = torch.arange(img_z.size(0), device=img_z.device)

        # Extra negatives from queue
        if self.queue.emb.sum() != 0:  # Check if queue has been populated
            q_logits = img_z @ self.queue.emb.t() / tau   # B × Q
            
            # Mask out positives (same image_id) in queue
            mask = img_ids.unsqueeze(1).eq(self.queue.ids.unsqueeze(0))
            q_logits = q_logits.masked_fill(mask, float('-inf'))
            
            # Concatenate batch and queue logits
            logits = torch.cat([logits, q_logits], dim=1)

        # Cross-entropy loss (image to text)
        loss = F.cross_entropy(logits, labels)
        
        # Update queue with current batch
        self.queue.enqueue(txt_z.detach(), img_ids.detach())
        
        return loss, logits.detach(), labels
    
    def get_temperature(self):
        """Get current temperature (for monitoring)."""
        if self.learnable_scale:
            return 1.0 / self.logit_scale.exp().item()
        else:
            return self.tau


# Loss function management


# Global loss instances for stateful losses
_loss_instances = {}


def clear_loss_instances():
    """Clear cached loss instances. Useful for hyperparameter tuning."""
    global _loss_instances
    _loss_instances.clear()


def get_loss_instance(loss_type, config, device):
    """Get or create loss function instance."""
    learnable_scale = config['loss'].get('learnable_scale', False)
    key = f"{loss_type}_{device}_{learnable_scale}"
    
    if key not in _loss_instances:
        temperature = float(config['loss'].get('temperature', 0.07))
        
        if loss_type == 'sigmoid_infonce':
            _loss_instances[key] = SigmoidInfoNCELoss(tau=temperature, learnable_scale=learnable_scale)
        elif loss_type == 'softmax_infonce':
            _loss_instances[key] = SoftmaxInfoNCELoss(tau=temperature, learnable_scale=learnable_scale)
        elif loss_type == 'queue_infonce':
            queue_size = int(config['loss'].get('queue_size', 4096))
            feature_dim = int(config['loss'].get('feature_dim', 512))
            loss_fn = QueueInfoNCELoss(queue_size, temperature, feature_dim, learnable_scale=learnable_scale)
            _loss_instances[key] = loss_fn.to(device) if hasattr(loss_fn, 'to') else loss_fn
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    return _loss_instances[key]


def compute_loss(image_proj, text_proj, config, img_ids):
    """Compute loss based on configuration.
    
    Args:
        image_proj: Image projection features
        text_proj: Text projection features  
        config: Training configuration
        img_ids: Image IDs for queue-based losses
    
    Returns:
        Computed loss value
    """
    loss_type = config['loss'].get('type', 'sigmoid_infonce')
    assert img_ids is not None, "img_ids must be provided for loss computation"
    
    # InfoNCE losses - temperature is handled internally by loss functions
    if loss_type in ['sigmoid_infonce', 'softmax_infonce', 'queue_infonce']:
        loss_fn = get_loss_instance(loss_type, config, image_proj.device)
        loss, _, _ = loss_fn(image_proj, text_proj, img_ids)
        return loss
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Supported: sigmoid_infonce, softmax_infonce, queue_infonce")