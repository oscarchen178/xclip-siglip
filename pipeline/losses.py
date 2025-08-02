#!/usr/bin/env python3
"""
Loss functions for cross-modal projection head training.

This module contains all loss function implementations for different
SOTA methods with proper InfoNCE variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidInfoNCELoss(nn.Module):
    """Sigmoid InfoNCE loss (SigLIP) - uses binary cross-entropy."""
    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(self, img_z, txt_z, img_ids=None):
        # Normalize features
        img_z = F.normalize(img_z, dim=-1)
        txt_z = F.normalize(txt_z, dim=-1)
        
        # Similarity matrix
        logits = img_z @ txt_z.t() / self.tau                 # B × B
        labels = torch.arange(img_z.size(0), device=img_z.device)
        
        # Create one-hot labels for positive pairs
        target = F.one_hot(labels, num_classes=logits.size(1)).float()
        
        # Binary cross-entropy with logits (sigmoid InfoNCE)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        
        return loss, logits.detach(), labels


class SoftmaxInfoNCELoss(nn.Module):
    """Softmax InfoNCE loss (CLIP) - uses cross-entropy."""
    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(self, img_z, txt_z, img_ids=None):
        # Normalize features
        img_z = F.normalize(img_z, dim=-1)
        txt_z = F.normalize(txt_z, dim=-1)
        
        # Similarity matrix
        logits = img_z @ txt_z.t() / self.tau                  # B × B
        labels = torch.arange(img_z.size(0), device=img_z.device)
        
        # Cross-entropy loss for both directions (symmetric)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_i2t + loss_t2i)
        
        return loss, logits.detach(), labels


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
    def __init__(self, queue_size: int = 4096, tau: float = 0.07, feature_dim: int = 512):
        super().__init__()
        self.tau = tau
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
        
        # Ensure img_ids is on the same device
        if img_ids is not None:
            img_ids = img_ids.to(img_z.device)
        
        # Initialize queue if needed or if feature dimension has changed
        if self.queue is None or self.queue.emb.size(1) != img_z.size(1):
            self._init_queue(img_z.device, img_z.size(1))
        
        # Move queue to correct device if needed
        if self.queue.emb.device != img_z.device:
            self.queue.to(img_z.device)

        # In-batch logits
        logits = img_z @ txt_z.t() / self.tau                  # B × B
        labels = torch.arange(img_z.size(0), device=img_z.device)

        # Extra negatives from queue
        if self.queue.emb.sum() != 0:  # Check if queue has been populated
            q_logits = img_z @ self.queue.emb.t() / self.tau   # B × Q
            
            # Mask out positives (same image_id) in queue
            if img_ids is not None:
                mask = img_ids.unsqueeze(1).eq(self.queue.ids.unsqueeze(0))
                q_logits = q_logits.masked_fill(mask, float('-inf'))
            
            # Concatenate batch and queue logits
            logits = torch.cat([logits, q_logits], dim=1)

        # Cross-entropy loss (image to text)
        loss = F.cross_entropy(logits, labels)
        
        # Update queue with current batch
        if img_ids is not None:
            self.queue.enqueue(txt_z.detach(), img_ids.detach())
        else:
            # If no img_ids, use indices as fallback
            batch_ids = torch.arange(len(txt_z), device=txt_z.device)
            self.queue.enqueue(txt_z.detach(), batch_ids)
        
        return loss, logits.detach(), labels


# Loss function management


# Global loss instances for stateful losses
_loss_instances = {}


def clear_loss_instances():
    """Clear cached loss instances. Useful for hyperparameter tuning."""
    global _loss_instances
    _loss_instances.clear()


def get_loss_instance(loss_type, config, device):
    """Get or create loss function instance."""
    key = f"{loss_type}_{device}"
    
    if key not in _loss_instances:
        temperature = float(config['loss'].get('temperature', 0.07))
        
        if loss_type == 'sigmoid_infonce':
            _loss_instances[key] = SigmoidInfoNCELoss(tau=temperature)
        elif loss_type == 'softmax_infonce':
            _loss_instances[key] = SoftmaxInfoNCELoss(tau=temperature)
        elif loss_type == 'queue_infonce':
            queue_size = int(config['loss'].get('queue_size', 4096))
            feature_dim = int(config['loss'].get('feature_dim', 512))
            loss_fn = QueueInfoNCELoss(queue_size, temperature, feature_dim)
            _loss_instances[key] = loss_fn.to(device) if hasattr(loss_fn, 'to') else loss_fn
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    return _loss_instances[key]


def compute_loss(image_proj, text_proj, config, image_head=None, text_head=None, img_ids=None):
    """Compute loss based on configuration.
    
    Args:
        image_proj: Image projection features
        text_proj: Text projection features  
        config: Training configuration
        image_head: Image projection head (for CLIP temperature)
        text_head: Text projection head (for CLIP temperature)
        img_ids: Image IDs for queue-based losses
    
    Returns:
        Computed loss value
    """
    loss_type = config['loss'].get('type', 'sigmoid_infonce')
    
    # InfoNCE losses
    if loss_type in ['sigmoid_infonce', 'softmax_infonce', 'queue_infonce']:
        loss_fn = get_loss_instance(loss_type, config, image_proj.device)
        
        if loss_type == 'queue_infonce':
            if img_ids is None:
                img_ids = torch.arange(len(image_proj), device=image_proj.device)
            loss, _, _ = loss_fn(image_proj, text_proj, img_ids)
        else:
            loss, _, _ = loss_fn(image_proj, text_proj, img_ids)
        
        return loss
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Supported: sigmoid_infonce, softmax_infonce, queue_infonce")