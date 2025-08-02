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
    """Split train2017 data at IMAGE LEVEL into train/test to prevent data leakage."""
    
    print("Loading train2017 embeddings...")
    
    # Load train2017 data (from parent directory)
    train_img_path = "../pretrain_encoded/train2017_image_embeddings.pt"
    train_txt_path = "../pretrain_encoded/train2017_text_embeddings.pt"
    metadata_path = "../pretrain_encoded/train2017_metadata.json"
    
    if not Path(train_img_path).exists():
        print(f"Error: {train_img_path} not found!")
        return
    
    if not Path(metadata_path).exists():
        print(f"Error: {metadata_path} not found! Need image IDs for proper splitting.")
        return
    
    image_embeddings = torch.load(train_img_path, map_location='cpu')
    text_embeddings = torch.load(train_txt_path, map_location='cpu')
    
    # Load metadata for image IDs (REQUIRED for proper splitting)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if 'image_ids' not in metadata:
        print("Error: No image_ids in metadata! Cannot perform proper image-level split.")
        return
    
    image_ids = metadata['image_ids']
    total_samples = len(image_embeddings)
    print(f"Total train2017 samples: {total_samples}")
    
    # Group samples by unique image ID
    from collections import defaultdict
    image_to_indices = defaultdict(list)
    for idx, img_id in enumerate(image_ids):
        image_to_indices[img_id].append(idx)
    
    unique_images = list(image_to_indices.keys())
    print(f"Unique images: {len(unique_images)}")
    print(f"Average captions per image: {total_samples / len(unique_images):.1f}")
    
    # Split at IMAGE level (not caption level)
    np.random.seed(seed)
    unique_images_shuffled = np.random.permutation(unique_images)
    
    test_images_count = int(len(unique_images) * test_ratio)
    train_images_count = len(unique_images) - test_images_count
    
    train_image_ids_set = set(unique_images_shuffled[:train_images_count])
    test_image_ids_set = set(unique_images_shuffled[train_images_count:])
    
    print(f"Image-level split: {train_images_count} train images, {test_images_count} test images")
    
    # Collect all indices for train and test sets
    train_indices = []
    test_indices = []
    
    for idx, img_id in enumerate(image_ids):
        if img_id in train_image_ids_set:
            train_indices.append(idx)
        elif img_id in test_image_ids_set:
            test_indices.append(idx)
    
    print(f"Sample-level split: {len(train_indices)} train samples, {len(test_indices)} test samples")
    
    # Split embeddings
    train_images = image_embeddings[train_indices]
    train_texts = text_embeddings[train_indices]
    test_images = image_embeddings[test_indices]
    test_texts = text_embeddings[test_indices]
    
    # Create metadata for splits
    train_image_ids = [image_ids[i] for i in train_indices]
    test_image_ids = [image_ids[i] for i in test_indices]
    
    train_metadata = {
        'dataset_name': 'train',
        'total_pairs': len(train_images),
        'unique_images': len(train_image_ids_set),
        'image_ids': train_image_ids,
        'source': 'image-level split from train2017'
    }
    test_metadata = {
        'dataset_name': 'test', 
        'total_pairs': len(test_images),
        'unique_images': len(test_image_ids_set),
        'image_ids': test_image_ids,
        'source': 'image-level split from train2017'
    }
    
    # Verify no image leakage
    train_unique = set(train_image_ids)
    test_unique = set(test_image_ids)
    overlap = train_unique.intersection(test_unique)
    if overlap:
        print(f"ERROR: Data leakage detected! {len(overlap)} images appear in both train and test")
        return
    else:
        print("✅ No data leakage - train and test have completely separate images")
    
    
    # Save split data (back to pretrain_encoded directory)
    output_dir = Path("../pretrain_encoded")
    output_dir.mkdir(exist_ok=True)
    
    print("Saving split data...")
    
    # Save training split
    torch.save(train_images, output_dir / "train_image_embeddings.pt")
    torch.save(train_texts, output_dir / "train_text_embeddings.pt")
    with open(output_dir / "train_metadata.json", 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    # Create tuning subset from TRAIN split only (simple random sampling)
    TUNING_SUBSET_SIZE = 100000
    if len(train_images) > TUNING_SUBSET_SIZE:
        print(f"Creating tuning subset ({TUNING_SUBSET_SIZE:,} samples from train split)...")
        
        # Simple random sampling from train set (not image-level, just for tuning speed)
        np.random.seed(123)  # Fixed seed for reproducible tuning subset
        tuning_indices = np.random.choice(len(train_images), TUNING_SUBSET_SIZE, replace=False)
        
        tuning_images = train_images[tuning_indices]
        tuning_texts = train_texts[tuning_indices]
        tuning_image_ids = [train_image_ids[i] for i in tuning_indices]
        
        tuning_metadata = {
            'dataset_name': 'train_tuning',
            'total_pairs': len(tuning_images),
            'unique_images': len(set(tuning_image_ids)),
            'image_ids': tuning_image_ids,
            'source': f'random subset of {TUNING_SUBSET_SIZE:,} samples from train split for hyperparameter tuning'
        }
        
        # Save tuning subset
        torch.save(tuning_images, output_dir / "train_tuning_image_embeddings.pt")
        torch.save(tuning_texts, output_dir / "train_tuning_text_embeddings.pt") 
        with open(output_dir / "train_tuning_metadata.json", 'w') as f:
            json.dump(tuning_metadata, f, indent=2)
        
        print(f"✅ Tuning subset created: {TUNING_SUBSET_SIZE:,} samples from train split only")
    
    # Save test split  
    torch.save(test_images, output_dir / "test_image_embeddings.pt")
    torch.save(test_texts, output_dir / "test_text_embeddings.pt")
    with open(output_dir / "test_metadata.json", 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    # Copy val2017 to shorter names
    import shutil
    print("Creating validation files with shorter names...")
    if (output_dir / "val2017_image_embeddings.pt").exists():
        shutil.copy2(output_dir / "val2017_image_embeddings.pt", output_dir / "val_image_embeddings.pt")
        shutil.copy2(output_dir / "val2017_text_embeddings.pt", output_dir / "val_text_embeddings.pt")
    
    print("✅ Image-level data split complete!")
    print(f"Training: {len(train_images)} samples from {len(train_image_ids_set)} images (~{len(train_images)/1000:.0f}K)")
    print(f"Test: {len(test_images)} samples from {len(test_image_ids_set)} images (~{len(test_images)/1000:.0f}K)")
    print(f"Validation: Use existing val2017_* files (~5K samples)")
    print()
    print("Files created:")
    print("  train_image_embeddings.pt / train_text_embeddings.pt (full train set)")
    print("  train_tuning_image_embeddings.pt / train_tuning_text_embeddings.pt (100K subset for tuning)")
    print("  val_image_embeddings.pt / val_text_embeddings.pt") 
    print("  test_image_embeddings.pt / test_text_embeddings.pt")
    print("  train_metadata.json / test_metadata.json / train_tuning_metadata.json (with image IDs)")
    print()
    print("✅ No data leakage - completely separate images in train/test!")
    print("✅ Config files are already set up to use these files!")

if __name__ == "__main__":
    split_train_data()