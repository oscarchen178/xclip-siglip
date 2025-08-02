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


# Legacy compatibility functions
def siglip_loss(image_features, text_features, temperature=0.05):
    """Legacy SigLIP loss - redirects to SigmoidInfoNCELoss."""
    loss_fn = SigmoidInfoNCELoss(tau=temperature)
    loss, _, _ = loss_fn(image_features, text_features)
    return loss


def clip_loss(image_features, text_features, logit_scale):
    """CLIP loss with learnable temperature."""
    temperature = 1.0 / logit_scale.exp()
    loss_fn = SoftmaxInfoNCELoss(tau=temperature)
    loss, _, _ = loss_fn(image_features, text_features)
    return loss


def align_loss(image_features, text_features, temperature=0.05):
    """ALIGN loss - same as softmax InfoNCE."""
    loss_fn = SoftmaxInfoNCELoss(tau=temperature)
    loss, _, _ = loss_fn(image_features, text_features)
    return loss


def blip_loss(image_features, text_features, momentum_image_features, momentum_text_features, 
              temperature=0.05, alpha=0.4):
    """BLIP momentum contrastive loss."""
    batch_size = image_features.shape[0]
    
    # Normalize features
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
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
    
    # Legacy CLIP loss with learnable temperature
    elif loss_type == 'clip':
        if hasattr(image_head, 'logit_scale'):
            logit_scale = image_head.logit_scale
        elif hasattr(text_head, 'logit_scale'):
            logit_scale = text_head.logit_scale
        else:
            temp = float(config['loss']['temperature'])
            logit_scale = torch.tensor(1.0 / temp, device=image_proj.device).log()
        return clip_loss(image_proj, text_proj, logit_scale)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Supported: sigmoid_infonce, softmax_infonce, queue_infonce, clip")