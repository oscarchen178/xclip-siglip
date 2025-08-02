#!/usr/bin/env python3
"""Shared dataset classes for cross-modal training and evaluation."""

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Dataset for pre-encoded embeddings with optional image ID support."""
    
    def __init__(self, image_path, text_path, metadata_path=None, with_image_ids=True):
        """
        Args:
            image_path: Path to image embeddings (.pt file)
            text_path: Path to text embeddings (.pt file)  
            metadata_path: Optional path to metadata JSON with image_ids
            with_image_ids: If True, returns (img_emb, txt_emb, img_id). If False, returns (img_emb, txt_emb)
        """
        self.image_embeddings = torch.load(image_path, map_location='cpu').float()
        self.text_embeddings = torch.load(text_path, map_location='cpu').float()
        self.with_image_ids = with_image_ids
        
        # Validate dimensions match
        assert len(self.image_embeddings) == len(self.text_embeddings), \
            f"Mismatch: {len(self.image_embeddings)} images vs {len(self.text_embeddings)} texts"
        
        # Load image IDs if requested
        self.image_ids = None
        if with_image_ids:
            if metadata_path and Path(metadata_path).exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    self.image_ids = metadata.get('image_ids', None)
                    if self.image_ids and len(self.image_ids) != len(self.image_embeddings):
                        print(f"Warning: Image ID count mismatch, using indices instead")
                        self.image_ids = None
                except Exception as e:
                    print(f"Warning: Could not load metadata from {metadata_path}: {e}")
            
            # Fallback to indices if no valid image IDs
            if self.image_ids is None:
                self.image_ids = list(range(len(self.image_embeddings)))
        
    def __len__(self):
        return len(self.image_embeddings)
    
    def __getitem__(self, idx):
        if self.with_image_ids:
            return self.image_embeddings[idx], self.text_embeddings[idx], self.image_ids[idx]
        else:
            return self.image_embeddings[idx], self.text_embeddings[idx]


# Convenience functions for common use cases
def create_train_dataset(pretrain_dir="../pretrain_encoded",
                        image_path="train_image_embeddings.pt", 
                        text_path="train_text_embeddings.pt",
                        metadata_path="train_metadata.json"):
    """Create training dataset with image IDs for queue losses."""
    image_path = Path(pretrain_dir) / image_path
    text_path = Path(pretrain_dir) / text_path
    metadata_path = Path(pretrain_dir) / metadata_path
    return EmbeddingDataset(image_path, text_path, metadata_path, with_image_ids=True)


def create_eval_dataset(pretrain_dir, image_path, text_path, metadata_path=None):
    """Create evaluation dataset with image IDs for proper COCO metrics."""
    image_path = Path(pretrain_dir) / image_path
    text_path = Path(pretrain_dir) / text_path
    if metadata_path:
        metadata_path = Path(pretrain_dir) / metadata_path
    return EmbeddingDataset(image_path, text_path, metadata_path, with_image_ids=True)


def create_legacy_dataset(image_path, text_path):
    """Create legacy dataset without image IDs (for backward compatibility)."""
    return EmbeddingDataset(image_path, text_path, with_image_ids=False)