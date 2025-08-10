#!/usr/bin/env python3
"""
Visualization script for cross-modal retrieval results.

Shows retrieval results:
- 5 captions for 1 image (Image-to-Text)
- 5 images for 1 caption (Text-to-Image)

Usage:
    python visualize_retrieval.py configs/siglip.yaml
    python visualize_retrieval.py configs/attention.yaml
"""

import sys
import torch
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import random
from PIL import Image
import textwrap

from dataset import create_eval_dataset
from evaluate import load_model_from_checkpoint


def get_image_path_from_id(image_id, coco_root="data"):
    """Convert COCO image ID to file path."""
    # Try different possible paths
    possible_paths = [
        Path(coco_root) / "train2017" / f"{image_id:012d}.jpg",  # Try train2017 first (test set comes from train)
        Path(coco_root) / "val2017" / f"{image_id:012d}.jpg",   # Then val2017
        Path("../data") / "train2017" / f"{image_id:012d}.jpg", # Try relative paths
        Path("../data") / "val2017" / f"{image_id:012d}.jpg",
        Path("data") / "train2017" / f"{image_id:012d}.jpg",    # Try from current dir
        Path("data") / "val2017" / f"{image_id:012d}.jpg"
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None


def get_top_retrievals(similarities, query_idx, image_ids, k=5, is_text_query=False, avoid_duplicates=True):
    """Get top-k retrievals with optional duplicate filtering for images."""
    if is_text_query:
        # Text-to-Image: similarities is (num_images, num_texts), we want column query_idx
        sims = similarities[:, query_idx]
        # Expand candidate pool adaptively to ensure we can find k unique images
        max_pool = min(sims.numel(), k * 50)
        top_indices = torch.topk(sims, max_pool).indices
        top_similarities = sims[top_indices]

        if avoid_duplicates:
            seen_image_ids = set()
            unique_results = []
            for idx, sim in zip(top_indices, top_similarities):
                img_id = image_ids[idx.item()]
                if img_id not in seen_image_ids:
                    seen_image_ids.add(img_id)
                    unique_results.append((idx.item(), sim.item()))
                    if len(unique_results) >= k:
                        break
            return unique_results[:k]
        else:
            return [(idx.item(), sim.item()) for idx, sim in zip(top_indices[:k], top_similarities[:k])]
    else:
        # Image-to-Text: similarities is (num_images, num_texts), we want row query_idx  
        sims = similarities[query_idx]
        top_indices = torch.topk(sims, k).indices
        top_similarities = sims[top_indices]
        return [(idx.item(), sim.item()) for idx, sim in zip(top_indices, top_similarities)]


def visualize_i2t(similarities, sample_idx, captions, image_ids, coco_root="../data", save_path=None):
    """Image→Text visualization: large query image + list of top-5 captions."""

    # Get query image info
    query_image_id = image_ids[sample_idx]
    query_caption = captions[sample_idx]
    query_image_path = get_image_path_from_id(query_image_id, coco_root)

    # Get top-5 retrievals
    i2t_results = get_top_retrievals(similarities, sample_idx, image_ids, k=5, is_text_query=False)
    t2i_results = get_top_retrievals(similarities, sample_idx, image_ids, k=5, is_text_query=True, avoid_duplicates=True)
    # Short query caption for console printout
    query_caption_short = (query_caption[:80] + '...') if len(query_caption) > 80 else query_caption

    # Figure layout similar to old caption retrieval scripts
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(17, 7))
    outer = gridspec.GridSpec(1, 1)

    # Top row: left image, right caption list
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1.2, 1.8], wspace=0.08)
    ax_img = fig.add_subplot(top[0, 0])
    if query_image_path and Path(query_image_path).exists():
        try:
            qimg = Image.open(query_image_path)
            ax_img.imshow(qimg)
            ax_img.set_title(f"Query Image\nID: {query_image_id}", fontsize=14, fontweight='bold')
        except Exception:
            ax_img.text(0.5, 0.5, f'Image not found\nID: {query_image_id}', ha='center', va='center')
            ax_img.set_title('Query Image', fontsize=14, fontweight='bold')
    else:
        ax_img.text(0.5, 0.5, f'Image not found\nID: {query_image_id}', ha='center', va='center')
        ax_img.set_title('Query Image', fontsize=14, fontweight='bold')
    ax_img.axis('off')

    ax_txt = fig.add_subplot(top[0, 1])
    ax_txt.axis('off')
    # Build monospaced caption list
    lines = ["Top 5 Captions (Image→Text):\n"]
    for rank, (idx, sim) in enumerate(i2t_results[:5], start=1):
        cap = captions[idx]
        lines.append(f"{rank}. {cap}")
        lines.append(f"   Score: {sim:.3f}")
        lines.append("")
    ax_txt.text(0.03, 0.98, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=18, va='top', fontfamily='monospace', wrap=True)

    plt.suptitle(f'Image→Text Retrieval (Sample {sample_idx})', fontsize=20, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    # Print concise results
    print(f"\nImage-to-Text (Query ID: {query_image_id}):")
    for i, (idx, sim) in enumerate(i2t_results[:3]):
        caption_short = (captions[idx][:80] + '...') if len(captions[idx]) > 80 else captions[idx]
        print(f"  {i+1}. {sim:.3f}: {caption_short}")
    
    print()


def visualize_t2i(similarities, sample_idx, captions, image_ids, coco_root="../data", save_path=None):
    """Text→Image visualization: large query caption + grid of top-5 images."""
    # Query data
    query_image_id = image_ids[sample_idx]
    query_caption = captions[sample_idx]

    # Get top-5 retrievals (text→image)
    t2i_results = get_top_retrievals(similarities, sample_idx, image_ids, k=5, is_text_query=True, avoid_duplicates=True)

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(24, 5.5))
    # Make layout compact: smaller caption row and zero vertical spacing to reduce the gap
    outer = gridspec.GridSpec(2, 1, height_ratios=[0.12, 1], hspace=0.0)

    # Query caption
    ax_qcap = fig.add_subplot(outer[0, 0])
    ax_qcap.axis('off')
    ax_qcap.text(0.02, 0.95, "Query Caption (Text→Image):", fontsize=20, fontweight='bold', va='top')
    # Place caption block with more space below the title
    ax_qcap.text(0.02, 0.40, textwrap.fill(query_caption, width=90), fontsize=18, va='top')

    # Compute width ratios from image aspect ratios (same height, variable widths)
    img_infos = []
    for idx, sim in t2i_results[:5]:
        ret_image_id = image_ids[idx]
        ret_path = get_image_path_from_id(ret_image_id, coco_root)
        aspect = 1.0
        if ret_path and Path(ret_path).exists():
            try:
                with Image.open(ret_path) as im_tmp:
                    w, h = im_tmp.size
                    aspect = (w / max(h, 1)) if h else 1.0
            except Exception:
                aspect = 1.0
        img_infos.append((idx, sim, ret_path, ret_image_id, aspect))
    width_ratios = [max(0.6, min(2.5, info[4])) for info in img_infos]

    # Images grid with per-cell sub-panels; same height row, variable column widths
    bot = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[1], wspace=0.04, width_ratios=width_ratios)
    for i, (idx, sim, ret_path, ret_image_id, _aspect) in enumerate(img_infos):
        # Allocate minimal gap between image and text, fit image snugly
        cell = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=bot[0, i], height_ratios=[0.82, 0.18], hspace=0.0)
        ax_img = fig.add_subplot(cell[0, 0])
        ax_txt = fig.add_subplot(cell[1, 0])

        if ret_path and Path(ret_path).exists():
            try:
                img = Image.open(ret_path)
                ax_img.imshow(img)
                ax_img.set_aspect('equal', adjustable='box')
                ax_img.set_anchor('S')  # anchor at bottom so image touches caption
                ax_img.margins(0)
            except Exception:
                ax_img.text(0.5, 0.5, f'Could not load\nID: {ret_image_id}', ha='center', va='center')
        else:
            ax_img.text(0.5, 0.5, f'Image not found\nID: {ret_image_id}', ha='center', va='center')
        ax_img.axis('off')

        # Caption and meta under image with manual line wrapping
        ax_txt.axis('off')
        cap_wrapped = textwrap.fill(captions[idx], width=45)
        txt = f"#{i+1}  score {sim:.3f}\nID: {ret_image_id}\n{cap_wrapped}"
        ax_txt.text(0.02, 0.98, txt, va='top', ha='left', fontsize=11)

    plt.suptitle(f'Text→Image Retrieval (Sample {sample_idx})', fontsize=20, fontweight='bold', y=0.985)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize cross-modal retrieval results (top-5)")
    parser.add_argument("config", help="Path to config file")
    # Removed single-output arg; we will always save two files (retrieve_cap.png, retrieve_img.png)
    parser.add_argument("--idx", type=int, default=None, help="Specific sample index to visualize (default: random)")
    
    args = parser.parse_args()
    
    # Fixed parameters  
    coco_root = "../data"  # From pipeline directory, data is one level up
    seed = 42
    
    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get device first
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Determine model path using same logic as train.py
    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        temp_config = yaml.safe_load(f)
    
    config_name = config_path.stem
    output_dir = Path(temp_config.get('output_dir', 'results')) / config_name
    model_path = output_dir / 'best_model.pt'
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"Loading model from: {model_path}")
    
    # Load model using evaluate.py function
    try:
        image_head, text_head, config = load_model_from_checkpoint(model_path, device)
        print(f"✓ Model loaded: {config['model']['type']}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"✓ Using device: {device}")
    
    # Load VAL dataset to avoid train+test metadata mismatches
    base_dir = config.get('base_data_dir', '../pretrain_encoded')
    val_img = config['data'].get('val_image_embeddings', 'val_image_embeddings.pt')
    val_txt = config['data'].get('val_text_embeddings', 'val_text_embeddings.pt')
    # Prefer explicit val metadata if present; otherwise use COCO's val2017 metadata
    val_meta = config['data'].get('val_metadata', 'val2017_metadata.json')
    val_dataset = create_eval_dataset(
        base_dir,
        val_img,
        val_txt,
        val_meta
    )
    
    print(f"✓ Loaded val dataset: {len(val_dataset)} samples")
    
    # Load image IDs from metadata
    metadata_path = Path(base_dir) / val_meta
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    image_ids = metadata['image_ids']
    
    # Load captions aligned with embeddings. Prefer metadata to preserve order
    if 'captions' in metadata:
        captions = metadata['captions']
        print(f"✓ Loaded {len(captions)} captions from metadata")
    else:
        coco_annotations_path = "../data/annotations/captions_val2017.json"
        print(f"✓ Loading COCO captions from: {coco_annotations_path}")
        with open(coco_annotations_path, 'r') as f:
            coco_data = json.load(f)
        captions = [ann['caption'] for ann in coco_data.get('annotations', [])][:len(val_dataset)]
    
    # Compute features
    dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    all_image_features = []
    all_text_features = []
    
    print("✓ Computing features...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            img_emb, txt_emb = batch[:2]  # Handle with or without image_ids
            
            img_emb = img_emb.to(device, dtype=torch.float32)
            txt_emb = txt_emb.to(device, dtype=torch.float32)
            
            img_feat = image_head(img_emb)
            txt_feat = text_head(txt_emb)
            
            all_image_features.append(img_feat.cpu())
            all_text_features.append(txt_feat.cpu())
    
    image_features = torch.cat(all_image_features, dim=0)
    text_features = torch.cat(all_text_features, dim=0)
    
    # Compute similarities (normalize and compute cosine similarity)
    print("✓ Computing similarities...")
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    similarities = torch.matmul(image_features, text_features.t())  # (num_images, num_texts)
    
    # Select random sample
    sample_idx = args.idx if args.idx is not None else random.randint(0, len(val_dataset) - 1)
    print(f"✓ Selected sample {sample_idx} for visualization")
    
    # Save two separate figures: I2T and T2I, to the same folder as before
    out_dir = Path("../final_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    i2t_path = out_dir / "retrieve_cap.png"
    t2i_path = out_dir / "retrieve_img.png"
    visualize_i2t(similarities, sample_idx, captions, image_ids, coco_root, save_path=str(i2t_path))
    visualize_t2i(similarities, sample_idx, captions, image_ids, coco_root, save_path=str(t2i_path))


if __name__ == "__main__":
    main()