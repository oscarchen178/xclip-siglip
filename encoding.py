"""Encode MS-COCO images + captions with:
  Image encoder:  google/siglip2-giant-opt-patch16-256
  Text  encoder:  intfloat/e5-mistral-7b-instruct

Usage:
  python encoding.py <image_folder> <caption_file> [output_name]

Arguments:
  image_folder    Path to folder containing COCO images (e.g., train2017/ or val2017/)
  caption_file    Path to COCO caption annotations JSON file
  output_name     Optional output name prefix (default: uses folder name)

Examples:
  python encoding.py data/val2017 data/annotations/captions_val2017.json
  python encoding.py data/train2017 data/annotations/captions_train2017.json train

Outputs:
  outputs/{name}_image_embeddings.pt   (tensor [N, D_img])
  outputs/{name}_text_embeddings.pt    (tensor [N, D_txt])
  outputs/{name}_metadata.json         (image/caption mappings)

Notes:
  * Sequential model loading to fit 4090 24GB VRAM
  * Downloads models to pretrained_encoders/ for reuse
  * Processes ALL images and captions in the dataset
"""
from __future__ import annotations
import json, random, math, os, time, sys
from pathlib import Path
from typing import List, Dict, Tuple
import gc

import torch
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
)

# ----------------------------- Configuration -------------------------------- #
OUTPUT_DIR = Path("outputs")
MODEL_CACHE_DIR = Path("pretrained_encoders")

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
IMAGE_MODEL_NAME = "google/siglip2-giant-opt-patch16-256"
TEXT_MODEL_NAME  = "intfloat/e5-mistral-7b-instruct"
IMAGE_BATCH_SIZE = 32
TEXT_BATCH_SIZE  = 16

# Using bfloat16 for speed/memory efficiency with large datasets
TORCH_DTYPE_PREF = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# Alternative for best quality (slower, more memory):
# TORCH_DTYPE_PREF = torch.float32

# ----------------------------- Utilities ------------------------------------ #
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clear_gpu_memory():
    """Clear GPU memory between models"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def load_coco_captions(captions_path: Path) -> Tuple[Dict[int, List[str]], Dict[int, dict]]:
    """Load all COCO captions and image metadata"""
    with open(captions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    id_to_captions: Dict[int, List[str]] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        id_to_captions.setdefault(img_id, []).append(ann["caption"].strip())
    
    id_to_imgmeta = {img['id']: img for img in data["images"]}
    return id_to_captions, id_to_imgmeta

def get_all_image_caption_pairs(id_to_caps: Dict[int, List[str]], id_to_imgmeta: Dict[int, dict], images_dir: Path) -> Tuple[List[Path], List[str], List[int], List[int]]:
    """Get ALL image-caption pairs from the dataset"""
    image_paths = []
    captions = []
    image_ids = []
    caption_indices = []
    
    for img_id, img_meta in tqdm(id_to_imgmeta.items(), desc="Preparing data"):
        img_filename = img_meta['file_name']
        img_path = images_dir / img_filename
        
        if not img_path.exists():
            print(f"Warning: Missing image {img_path}")
            continue
            
        caps = id_to_caps.get(img_id, [])
        for cap_idx, caption in enumerate(caps):
            image_paths.append(img_path)
            captions.append(caption)
            image_ids.append(img_id)
            caption_indices.append(cap_idx)
    
    return image_paths, captions, image_ids, caption_indices

def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

# ----------------------------- Encoding Functions -------------------------- #
@torch.no_grad()
def encode_all_images(image_paths: List[Path], batch_size: int) -> torch.Tensor:
    """Encode all images using SigLIP"""
    print(f"Loading SigLIP image encoder...")
    print_gpu_memory()
    
    # Load model with local caching
    image_processor = AutoProcessor.from_pretrained(
        IMAGE_MODEL_NAME, 
        cache_dir=MODEL_CACHE_DIR / "siglip2-giant"
    )
    image_model = AutoModel.from_pretrained(
        IMAGE_MODEL_NAME,
        torch_dtype=TORCH_DTYPE_PREF,
        device_map="auto",
        cache_dir=MODEL_CACHE_DIR / "siglip2-giant"
    )
    image_model.eval()
    
    # Debug info
    print(f"Image model device: {image_model.device}")
    print(f"Image model dtype: {next(image_model.parameters()).dtype}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
    print_gpu_memory()
    
    # Get unique image paths to avoid duplicate encoding
    unique_paths = list(dict.fromkeys(image_paths))  # Preserves order, removes duplicates
    path_to_idx = {path: idx for idx, path in enumerate(unique_paths)}
    
    print(f"Encoding {len(unique_paths)} unique images...")
    
    embs = []
    for i in tqdm(range(0, len(unique_paths), batch_size), desc="Encoding images", 
                  unit="batch", total=(len(unique_paths) + batch_size - 1) // batch_size):
        batch_paths = unique_paths[i:i+batch_size]
        images = [load_image(p) for p in batch_paths]
        
        inputs = image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(image_model.device, non_blocking=True) for k, v in inputs.items()}
        
        feats = image_model.get_image_features(**inputs)  # shape [B, D]
        feats = torch.nn.functional.normalize(feats, dim=-1)
        embs.append(feats.cpu())
        
        # Memory cleanup for large batches
        del inputs, feats
        
    unique_embeddings = torch.cat(embs, dim=0)
    
    # Create full embeddings array matching original image_paths order
    full_embeddings = torch.zeros(len(image_paths), unique_embeddings.shape[1])
    for i, path in enumerate(image_paths):
        unique_idx = path_to_idx[path]
        full_embeddings[i] = unique_embeddings[unique_idx]
    
    # Cleanup
    del image_model, image_processor, embs, unique_embeddings
    clear_gpu_memory()
    print("Image encoding complete, model unloaded")
    print_gpu_memory()
    
    return full_embeddings

@torch.no_grad()
def encode_all_texts(texts: List[str], batch_size: int) -> torch.Tensor:
    """Encode all captions using E5-Mistral"""
    print(f"Loading E5-Mistral text encoder...")
    print_gpu_memory()
    
    # Load model with local caching
    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR / "e5-mistral-7b"
    )
    text_model = AutoModel.from_pretrained(
        TEXT_MODEL_NAME,
        torch_dtype=TORCH_DTYPE_PREF,
        device_map="auto",
        cache_dir=MODEL_CACHE_DIR / "e5-mistral-7b"
    )
    text_model.eval()
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(text_model, 'gradient_checkpointing_enable'):
        text_model.gradient_checkpointing_enable()
    
    # Debug info
    print(f"Text model device: {text_model.device}")
    print(f"Text model dtype: {next(text_model.parameters()).dtype}")
    print_gpu_memory()
    print(f"Encoding {len(texts)} captions...")
    
    # E5 instruct models expect a task prefix
    proc_texts = [f"passage: {t}" for t in texts]
    
    embs = []
    for i in tqdm(range(0, len(proc_texts), batch_size), desc="Encoding captions",
                  unit="batch", total=(len(proc_texts) + batch_size - 1) // batch_size):
        batch_txt = proc_texts[i:i+batch_size]
        
        inputs = tokenizer(batch_txt, padding=True, truncation=True, 
                          return_tensors="pt", max_length=128)  # Reduced for single sentences
        inputs = {k: v.to(text_model.device, non_blocking=True) for k, v in inputs.items()}
        
        outputs = text_model(**inputs)
        
        # E5 uses last hidden state + attention mask mean pooling
        hidden = outputs.last_hidden_state  # [B, L, H]
        mask = inputs['attention_mask'].unsqueeze(-1)  # [B, L, 1]
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        sentence_emb = summed / counts
        sentence_emb = torch.nn.functional.normalize(sentence_emb, dim=-1)
        
        embs.append(sentence_emb.cpu())

        # Memory cleanup
        del inputs, outputs, hidden, mask, summed, counts, sentence_emb
    
    full_embeddings = torch.cat(embs, dim=0)
    
    # Cleanup
    del text_model, tokenizer, embs
    clear_gpu_memory()
    print("Text encoding complete, model unloaded")
    print_gpu_memory()
    
    return full_embeddings

# ----------------------------- Execution ------------------------------------ #
def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python encoding.py <image_folder> <caption_file> [output_name]")
        print("\nExamples:")
        print("  python encoding.py data/val2017 data/annotations/captions_val2017.json")
        print("  python encoding.py data/train2017 data/annotations/captions_train2017.json train")
        sys.exit(1)
    
    images_dir = Path(sys.argv[1])
    captions_file = Path(sys.argv[2])
    
    # Determine output name prefix
    if len(sys.argv) >= 4:
        output_name = sys.argv[3]
    else:
        output_name = images_dir.name  # Use folder name (e.g., "val2017", "train2017")
    
    set_seed(SEED)
    
    # Verify paths
    if not captions_file.exists():
        raise FileNotFoundError(f"Cannot find captions file at {captions_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Cannot find images directory at {images_dir}")

    print(f"Processing dataset: {output_name}")
    print(f"Images directory: {images_dir}")
    print(f"Captions file: {captions_file}")
    
    print("Loading COCO dataset...")
    id_to_caps, id_to_imgmeta = load_coco_captions(captions_file)
    
    print(f"Found {len(id_to_imgmeta)} images with {sum(len(caps) for caps in id_to_caps.values())} total captions")
    
    # Get ALL image-caption pairs
    image_paths, captions, image_ids, caption_indices = get_all_image_caption_pairs(id_to_caps, id_to_imgmeta, images_dir)
    
    print(f"Total pairs to encode: {len(image_paths)} (images: {len(set(image_paths))}, captions: {len(captions)})")
    
    # Sequential encoding to fit in 4090 memory
    print("\n" + "="*60)
    print("PHASE 1: ENCODING IMAGES")
    print("="*60)
    image_embeddings = encode_all_images(image_paths, IMAGE_BATCH_SIZE)
    
    # Save image embeddings immediately after encoding
    print("Saving image embeddings...")
    torch.save(image_embeddings, OUTPUT_DIR / f"{output_name}_image_embeddings.pt")
    print(f"Saved: {output_name}_image_embeddings.pt")
    
    print("\n" + "="*60)
    print("PHASE 2: ENCODING CAPTIONS") 
    print("="*60)
    text_embeddings = encode_all_texts(captions, TEXT_BATCH_SIZE)

    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    print("Final embeddings shapes:")
    print(f"  Images: {image_embeddings.shape}")
    print(f"  Texts:  {text_embeddings.shape}")

    # Save text embeddings (image embeddings already saved)
    print("Saving text embeddings...")
    torch.save(text_embeddings, OUTPUT_DIR / f"{output_name}_text_embeddings.pt")
    print(f"Saved: {output_name}_text_embeddings.pt")
    
    # Save comprehensive metadata
    metadata = {
        "dataset_name": output_name,
        "total_pairs": len(image_paths),
        "unique_images": len(set(image_paths)),
        "total_captions": len(captions),
        "image_paths": [str(p) for p in image_paths],
        "captions": captions,
        "image_ids": image_ids,
        "caption_indices": caption_indices,
        "image_model": IMAGE_MODEL_NAME,
        "text_model": TEXT_MODEL_NAME,
        "image_embedding_shape": list(image_embeddings.shape),
        "text_embedding_shape": list(text_embeddings.shape),
        "seed": SEED,
        "images_directory": str(images_dir),
        "captions_file": str(captions_file),
    }
    
    with open(OUTPUT_DIR / f"{output_name}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved embeddings and metadata to {OUTPUT_DIR}/ with prefix '{output_name}'")
    print("Files created:")
    print(f"  - {output_name}_image_embeddings.pt")
    print(f"  - {output_name}_text_embeddings.pt") 
    print(f"  - {output_name}_metadata.json")
    print("Done!")

if __name__ == "__main__":
    main()