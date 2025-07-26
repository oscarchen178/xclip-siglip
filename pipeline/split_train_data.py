#!/usr/bin/env python3
"""
Split the train2017 embeddings into proper train/validation splits.
This ensures that val2017 can be used as a held-out test set.

Usage:
    python split_train_data.py
"""

import torch
import json
from pathlib import Path
import numpy as np

def split_train_data(test_ratio=0.2, seed=42):
    """Split train2017 data into train/test with specified ratio. Use val2017 as validation."""
    
    print("Loading train2017 embeddings...")
    
    # Load train2017 data (from parent directory)
    train_img_path = "../pretrain_encoded/train2017_image_embeddings.pt"
    train_txt_path = "../pretrain_encoded/train2017_text_embeddings.pt"
    
    if not Path(train_img_path).exists():
        print(f"Error: {train_img_path} not found!")
        return
    
    image_embeddings = torch.load(train_img_path, map_location='cpu')
    text_embeddings = torch.load(train_txt_path, map_location='cpu')
    
    total_samples = len(image_embeddings)
    print(f"Total train2017 samples: {total_samples}")
    
    # Create reproducible split
    np.random.seed(seed)
    indices = np.random.permutation(total_samples)
    
    test_size = int(total_samples * test_ratio)
    train_size = total_samples - test_size
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    print(f"Split: {train_size} train, {test_size} test")
    
    # Split embeddings
    train_images = image_embeddings[train_indices]
    train_texts = text_embeddings[train_indices]
    test_images = image_embeddings[test_indices]
    test_texts = text_embeddings[test_indices]
    
    
    # Save split data (back to pretrain_encoded directory)
    output_dir = Path("../pretrain_encoded")
    output_dir.mkdir(exist_ok=True)
    
    print("Saving split data...")
    
    # Save training split
    torch.save(train_images, output_dir / "train_image_embeddings.pt")
    torch.save(train_texts, output_dir / "train_text_embeddings.pt")
    
    # Save test split  
    torch.save(test_images, output_dir / "test_image_embeddings.pt")
    torch.save(test_texts, output_dir / "test_text_embeddings.pt")
    
    # Copy val2017 to shorter names
    import shutil
    print("Creating validation files with shorter names...")
    if (output_dir / "val2017_image_embeddings.pt").exists():
        shutil.copy2(output_dir / "val2017_image_embeddings.pt", output_dir / "val_image_embeddings.pt")
        shutil.copy2(output_dir / "val2017_text_embeddings.pt", output_dir / "val_text_embeddings.pt")
    
    print("✅ Data split complete!")
    print(f"Training: {len(train_images)} samples (~{len(train_images)/1000:.0f}K)")
    print(f"Test: {len(test_images)} samples (~{len(test_images)/1000:.0f}K)")
    print(f"Validation: Use existing val2017_* files (~5K samples)")
    print()
    print("Files created:")
    print("  train_image_embeddings.pt / train_text_embeddings.pt")
    print("  val_image_embeddings.pt / val_text_embeddings.pt") 
    print("  test_image_embeddings.pt / test_text_embeddings.pt")
    print()
    print("✅ Config files are already set up to use these files!")

if __name__ == "__main__":
    split_train_data()