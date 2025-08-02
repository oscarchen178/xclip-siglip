#!/usr/bin/env python3
"""Fast metrics for cross-modal retrieval with ASHA pruning support."""

import torch
import torch.nn.functional as F
from typing import List
from collections import defaultdict


def build_img_to_txt_mask(image_ids: List, num_images: int, num_texts: int, device: torch.device) -> torch.Tensor:
    """Build sparse mask for image->text positive pairs (handles COCO 1:5 mapping)."""
    img_to_txt = torch.zeros(num_images, num_texts, dtype=torch.bool, device=device)
    img_id_to_indices = defaultdict(list)
    
    for idx, img_id in enumerate(image_ids):
        img_id_to_indices[img_id].append(idx)
    
    for i, img_id in enumerate(image_ids):
        if i < num_images:  # Only for image indices
            txt_indices = img_id_to_indices[img_id]
            img_to_txt[i, txt_indices] = True
    
    return img_to_txt


@torch.no_grad()
def fast_r1(img_vec: torch.Tensor, txt_vec: torch.Tensor, img_to_txt: torch.Tensor) -> float:
    """Vectorized R@1 computation for ASHA pruning."""
    img_vec = F.normalize(img_vec, dim=-1)
    txt_vec = F.normalize(txt_vec, dim=-1)
    
    sims = img_vec @ txt_vec.T                    # N_img × N_txt
    
    # Image→text: find if top-1 text is positive
    top1_txt = sims.argmax(dim=1)                 # (N_img,)
    img_hits = img_to_txt[torch.arange(img_vec.size(0)), top1_txt].float()
    
    # Text→image: find if top-1 image is exact match
    top1_img = sims.argmax(dim=0)                 # (N_txt,)
    txt_hits = (top1_img == torch.arange(txt_vec.size(0), device=txt_vec.device)).float()
    
    return 0.5 * (img_hits.mean() + txt_hits.mean()).item()


@torch.no_grad()
def compute_recall_at_k(img_vec: torch.Tensor, txt_vec: torch.Tensor, 
                       image_ids: List, k_values: List[int]) -> dict:
    """Vectorized recall@k computation."""
    img_vec = F.normalize(img_vec, dim=-1)
    txt_vec = F.normalize(txt_vec, dim=-1)
    
    num_imgs, num_txts = img_vec.size(0), txt_vec.size(0)
    device = img_vec.device
    
    # Build target mask
    img_to_txt = build_img_to_txt_mask(image_ids, num_imgs, num_txts, device)
    
    sims = img_vec @ txt_vec.T  # N_img × N_txt
    
    metrics = {}
    max_k = max(k_values)
    
    # Image→text retrieval
    _, top_indices = torch.topk(sims, max_k, dim=1)  # (N_img, max_k)
    target_gathered = img_to_txt.gather(1, top_indices)  # (N_img, max_k)
    cumulative_hits = target_gathered.float().cumsum(dim=1) > 0  # (N_img, max_k)
    
    for k in k_values:
        i2t_recall = cumulative_hits[:, k-1].float().mean().item()
        metrics[f'i2t_recall@{k}'] = i2t_recall
    
    # Text→image retrieval (exact match only)
    _, top_indices = torch.topk(sims.T, max_k, dim=1)  # (N_txt, max_k)
    exact_match = top_indices == torch.arange(num_txts, device=device).unsqueeze(1)  # (N_txt, max_k)
    cumulative_hits = exact_match.float().cumsum(dim=1) > 0  # (N_txt, max_k)
    
    for k in k_values:
        t2i_recall = cumulative_hits[:, k-1].float().mean().item()
        metrics[f't2i_recall@{k}'] = t2i_recall
    
    # Mean recall
    all_recalls = [metrics[f'i2t_recall@{k}'] for k in k_values] + [metrics[f't2i_recall@{k}'] for k in k_values]
    metrics['mean_recall'] = sum(all_recalls) / len(all_recalls)
    
    return metrics


@torch.no_grad()
def compute_validation_metrics(image_head, text_head, val_loader, device, fast_mode=True):
    """Extract features and compute metrics for validation."""
    image_head.eval()
    text_head.eval()
    
    all_img_feats, all_txt_feats, all_img_ids = [], [], []
    
    for batch in val_loader:
        if len(batch) == 3:
            image_emb, text_emb, img_ids = batch
            all_img_ids.extend(img_ids)
        else:
            image_emb, text_emb = batch
            all_img_ids.extend(list(range(len(image_emb))))  # Fallback indices
        
        image_emb = image_emb.to(device, dtype=torch.float32)
        text_emb = text_emb.to(device, dtype=torch.float32)
        
        img_proj = image_head(image_emb)
        txt_proj = text_head(text_emb)
        
        all_img_feats.append(img_proj.cpu())
        all_txt_feats.append(txt_proj.cpu())
    
    img_feats = torch.cat(all_img_feats, dim=0)
    txt_feats = torch.cat(all_txt_feats, dim=0)
    
    # Sanity check for COCO structure
    if len(txt_feats) == len(img_feats) * 5:
        assert all(isinstance(id, (int, str)) for id in all_img_ids), "Expected image IDs"
    
    if fast_mode:
        # Build mask once for R@1
        img_to_txt = build_img_to_txt_mask(all_img_ids, len(img_feats), len(txt_feats), img_feats.device)
        r1 = fast_r1(img_feats, txt_feats, img_to_txt)
        return {'val_recall_1': r1}
    else:
        return compute_recall_at_k(img_feats, txt_feats, all_img_ids, [1, 5, 10])