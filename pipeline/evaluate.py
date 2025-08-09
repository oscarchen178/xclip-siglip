#!/usr/bin/env python3
"""
Simple evaluation script for cross-modal projection heads.
All configuration is done through config.yaml file.

Usage:
    python evaluate.py [config_file]
    python evaluate.py model_path [config_file]  # Legacy support
    
Examples:
    python evaluate.py configs/attention_small.yaml
    python evaluate.py config.yaml
    python evaluate.py results/attention_small/best_model.pt  # Legacy
"""

import sys
import torch
import yaml
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

from dataset import create_eval_dataset

# Suppress sklearn numerical warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')




def compute_retrieval_metrics(image_features, text_features, k_values, image_ids=None, batch_size=1000):
    """Compute retrieval metrics with proper COCO handling (1 image -> 5 captions)."""
    # Normalize features
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    num_samples = image_features.size(0)
    device = image_features.device
    
    if image_ids is None:
        raise ValueError("image_ids must be provided for proper COCO evaluation. "
                        "Index-based evaluation gives incorrect results for multi-caption datasets.")
    
    # Create mapping from image_id to all indices with that image_id
    from collections import defaultdict
    image_id_to_indices = defaultdict(list)
    for idx, img_id in enumerate(image_ids):
        image_id_to_indices[img_id].append(idx)
    
    # Image-to-text retrieval (corrected for COCO) - optimized batch processing
    i2t_ranks = []
    for start_idx in tqdm(range(0, num_samples, batch_size), desc="Computing I2T ranks"):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = range(start_idx, end_idx)
        
        # Compute similarities for batch of images against all texts
        batch_similarities = torch.matmul(image_features[start_idx:end_idx], text_features.T)
        
        # Vectorized rank computation for the batch
        sorted_indices = torch.argsort(batch_similarities, descending=True, dim=1)
        
        for i, batch_i in enumerate(batch_indices):
            # Find rank of ANY caption from the same image
            target_image_id = image_ids[batch_i]
            target_caption_indices = image_id_to_indices[target_image_id]
            
            # Vectorized rank finding
            target_mask = torch.zeros(num_samples, dtype=torch.bool, device=device)
            target_mask[target_caption_indices] = True
            
            # Find positions where target indices appear in sorted order
            ranks = torch.where(target_mask[sorted_indices[i]])[0]
            best_rank = (ranks[0] + 1).item() if len(ranks) > 0 else num_samples
            i2t_ranks.append(best_rank)
    
    # Text-to-image retrieval (find rank of the EXACT corresponding image) - optimized batch processing  
    t2i_ranks = []
    for start_idx in tqdm(range(0, num_samples, batch_size), desc="Computing T2I ranks"):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = range(start_idx, end_idx)
        
        # Compute similarities for batch of texts against all images
        batch_similarities = torch.matmul(image_features, text_features[start_idx:end_idx].T)
        
        # Vectorized rank computation for T2I
        sorted_indices = torch.argsort(batch_similarities, descending=True, dim=0)
        
        for j, batch_j in enumerate(batch_indices):
            # Find rank of ANY image with the same image_id as this text
            target_image_id = image_ids[batch_j]
            target_image_indices = image_id_to_indices[target_image_id]
            
            # Create mask for target images
            target_mask = torch.zeros(num_samples, dtype=torch.bool, device=device)
            target_mask[target_image_indices] = True
            
            # Find positions where target images appear in sorted order
            ranks = torch.where(target_mask[sorted_indices[:, j]])[0]
            best_rank = (ranks[0] + 1).item() if len(ranks) > 0 else num_samples
            t2i_ranks.append(best_rank)
    
    # Compute recall@k
    metrics = {}
    for k in k_values:
        i2t_recall_k = sum(1 for rank in i2t_ranks if rank <= k) / len(i2t_ranks)
        t2i_recall_k = sum(1 for rank in t2i_ranks if rank <= k) / len(t2i_ranks)
        metrics[f'i2t_recall@{k}'] = i2t_recall_k
        metrics[f't2i_recall@{k}'] = t2i_recall_k
    
    # Mean recall and median rank
    all_recalls = [metrics[f'i2t_recall@{k}'] for k in k_values] + [metrics[f't2i_recall@{k}'] for k in k_values]
    metrics['mean_recall'] = sum(all_recalls) / len(all_recalls)
    metrics['i2t_median_rank'] = np.median(i2t_ranks)
    metrics['t2i_median_rank'] = np.median(t2i_ranks)
    
    # Add diagnostic info
    unique_images = len(set(image_ids))
    total_samples = len(image_ids)
    metrics['unique_images'] = unique_images
    metrics['total_samples'] = total_samples
    metrics['avg_captions_per_image'] = total_samples / unique_images
    
    print(f"Evaluation stats: {total_samples} samples, {unique_images} unique images, {total_samples/unique_images:.1f} captions/image")
    
    return metrics, None


def create_visualizations(image_features, text_features, output_dir, config):
    """Create embedding visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Limit samples for visualization
    n_samples = min(config['evaluation']['visualization_samples'], len(image_features))
    if len(image_features) > n_samples:
        indices = torch.randperm(len(image_features))[:n_samples]
        image_features = image_features[indices]
        text_features = text_features[indices]
    
    print(f"Creating visualizations with {n_samples} samples...")
    
    # Combine embeddings
    all_embeddings = torch.cat([image_features, text_features], dim=0)
    labels = ['Image'] * len(image_features) + ['Text'] * len(text_features)
    
    # t-SNE visualization
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=config['evaluation']['tsne_perplexity'], random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings.numpy())
    
    plt.figure(figsize=(10, 8))
    for label in ['Image', 'Text']:
        mask = np.array(labels) == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=label, alpha=0.6, s=20)
    
    plt.title('t-SNE Visualization of Cross-Modal Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # PCA visualization
    print("Running PCA...")
    pca = PCA(n_components=min(50, all_embeddings.shape[1]))
    embeddings_pca = pca.fit_transform(all_embeddings.numpy())
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 2D PCA plot
    for label in ['Image', 'Text']:
        mask = np.array(labels) == label
        axes[0].scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], label=label, alpha=0.6, s=20)
    
    axes[0].set_title('PCA Visualization (First 2 Components)')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Explained variance plot
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, len(cumvar) + 1), cumvar, 'bo-', markersize=4)
    axes[1].axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% variance')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Cumulative Explained Variance Ratio')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Import model classes (same as in train.py)
    from models import create_model
    
    # Get dimensions from state dict
    img_state = checkpoint['image_head_state']
    txt_state = checkpoint['text_head_state']
    
    # Get dimensions from first layer - try different layer name patterns
    possible_keys = ['projection.0.weight', 'layer_norm.weight', 'projection.weight']
    
    img_first_key = None
    for key in possible_keys:
        if key in img_state:
            img_first_key = key
            break
    
    if img_first_key is None:
        img_first_key = next(iter(img_state.keys()))
    
    txt_first_key = None
    for key in possible_keys:
        if key in txt_state:
            txt_first_key = key
            break
    
    if txt_first_key is None:
        txt_first_key = next(iter(txt_state.keys()))
    
    # Get input dimensions from the first layer
    if 'weight' in img_first_key:
        image_dim = img_state[img_first_key].shape[1] if img_state[img_first_key].dim() > 1 else img_state[img_first_key].shape[0]
        text_dim = txt_state[txt_first_key].shape[1] if txt_state[txt_first_key].dim() > 1 else txt_state[txt_first_key].shape[0]
    else:
        # Fallback to default dimensions
        image_dim = 1408  # SigLIP default
        text_dim = 4096   # E5-Mistral default
    
    # Create model - now returns only (image_head, text_head)
    image_head, text_head = create_model(config, image_dim, text_dim)
    
    # Load weights
    image_head.load_state_dict(checkpoint['image_head_state'])
    text_head.load_state_dict(checkpoint['text_head_state'])
    
    # Move to device with consistent float32 dtype
    image_head = image_head.to(device, dtype=torch.float32)
    text_head = text_head.to(device, dtype=torch.float32)
    
    return image_head, text_head, config


@torch.no_grad()
def extract_features(image_head, text_head, data_loader, device, pin_memory=False):
    """Extract features from the model."""
    image_head.eval()
    text_head.eval()
    
    image_features = []
    text_features = []
    image_ids = []
    
    for batch in tqdm(data_loader, desc="Extracting features"):
        if len(batch) != 3:
            raise ValueError("Dataset must provide (image_emb, text_emb, img_ids). "
                           "Old format without image IDs is no longer supported.")
        
        image_emb, text_emb, img_ids = batch
        image_ids.extend([img_id.item() for img_id in img_ids])  # Convert to list of integers
        
        # Force float32 for all devices to ensure consistency
        image_emb = image_emb.to(device, dtype=torch.float32, non_blocking=pin_memory)
        text_emb = text_emb.to(device, dtype=torch.float32, non_blocking=pin_memory)
        
        # Simple projection head forward pass
        img_proj = image_head(image_emb)
        txt_proj = text_head(text_emb)
        
        image_features.append(img_proj.cpu())
        text_features.append(txt_proj.cpu())
    
    return torch.cat(image_features, dim=0), torch.cat(text_features, dim=0), image_ids


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py [config_file]")
        print("       python evaluate.py model_path [config_file]  # Legacy")
        sys.exit(1)
    
    first_arg = sys.argv[1]
    
    # Check if first argument is a model path (legacy) or config file
    if first_arg.endswith('.pt'):
        # Legacy mode: model_path [config_file]
        model_path = first_arg
        config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
        print(f"Legacy mode: Evaluating model: {model_path}")
    else:
        # New mode: config_file
        config_path = first_arg
        # Load config to determine model path
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config_name = Path(config_path).stem
        model_path = Path(config['output_dir']) / config_name / 'best_model.pt'
        print(f"Config mode: Using config {config_path}")
        print(f"Model path: {model_path}")
    
    # Load config (if not already loaded)
    if 'config' not in locals():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
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
    
    # Set MPS compatibility settings
    if device.type == 'mps':
        torch.backends.mps.fallback_enabled = True
    
    # Load model
    image_head, text_head, model_config = load_model_from_checkpoint(model_path, device)
    print(f"Model loaded successfully! Type: {model_config['model']['type']}")
    
    # Create evaluation dataset (test set only)
    if 'test_image_embeddings' not in config['data'] or 'test_text_embeddings' not in config['data']:
        print("Error: Test set not found in config!")
        print("Please run 'python split_train_data.py' first to create proper data splits.")
        sys.exit(1)
    
    print("Using test set for evaluation (~24K samples)")
    
    # Try to find metadata file for image IDs
    test_img_path = Path(config['data']['test_image_embeddings'])
    metadata_path = test_img_path.parent / "test_metadata.json"  # Look for test metadata
    if not metadata_path.exists():
        # Fallback: try original train2017 metadata (contains all image IDs)
        metadata_path = test_img_path.parent / "train2017_metadata.json"
    
    eval_dataset = create_eval_dataset(
        "../pretrain_encoded",
        config['data']['test_image_embeddings'],
        config['data']['test_text_embeddings'],
        "test_metadata.json"
    )
    
    # Create data loader with device-optimized settings
    pin_memory = device.type == 'cuda'  # Only use pin_memory for CUDA
    num_workers = 4 if device.type != 'mps' else 0  # MPS works better with num_workers=0
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Extract features
    print("Extracting features...")
    image_features, text_features, image_ids = extract_features(
        image_head, text_head, eval_loader, device, pin_memory
    )
    
    # Compute retrieval metrics
    print("Computing retrieval metrics...")
    metrics, similarities = compute_retrieval_metrics(
        image_features, text_features, config['evaluation']['top_k'], image_ids
    )
    
    # Print results
    print("\n" + "="*50)
    print("RETRIEVAL RESULTS")
    print("="*50)
    
    for k in config['evaluation']['top_k']:
        print(f"I2T Recall@{k}: {metrics[f'i2t_recall@{k}']:.3f}")
        print(f"T2I Recall@{k}: {metrics[f't2i_recall@{k}']:.3f}")
        print("-" * 30)
    
    print(f"Mean Recall: {metrics['mean_recall']:.3f}")
    print(f"I2T Median Rank: {metrics['i2t_median_rank']:.1f}")
    print(f"T2I Median Rank: {metrics['t2i_median_rank']:.1f}")
    
    # Save results
    output_dir = Path(model_path).parent / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "metrics.json", 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        json_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, np.floating):
                json_metrics[k] = float(v)
            else:
                json_metrics[k] = v
        json.dump(json_metrics, f, indent=2)
    
    # Save features if requested
    if config['evaluation']['save_features']:
        torch.save(image_features, output_dir / "image_features.pt")
        torch.save(text_features, output_dir / "text_features.pt")
        torch.save(similarities, output_dir / "similarities.pt")
        print(f"Features saved to {output_dir}")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(image_features, text_features, output_dir / "visualizations", config)
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()