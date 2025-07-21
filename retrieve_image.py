"""Text-to-Image Retrieval using trained projection heads.

Usage:
  python retrieve_image.py "<text_query>" <k>
  
Examples:
  python retrieve_image.py "a cat sitting on a chair" 5
  python retrieve_image.py "people playing soccer" 3
"""
import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

# ======================== CONFIGURATION ========================
DATASET_NAME = "val2017"            # Which dataset to search in
MODEL_PATH = "models/best"          # Path prefix for projection models
HIDDEN_DIM = 1024                   # Must match training config
OUTPUT_DIM = 1024                   # Must match training config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===============================================================

class ProjectionHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
            nn.LayerNorm(OUTPUT_DIM)
        )
    
    def forward(self, x):
        return self.mlp(x)

def load_models_and_data():
    """Load trained projection models and dataset embeddings"""
    print("Loading models and data...")
    
    # Load dataset embeddings and metadata
    outputs_dir = Path("outputs")
    image_embeddings = torch.load(outputs_dir / f"{DATASET_NAME}_image_embeddings.pt").float()
    text_embeddings = torch.load(outputs_dir / f"{DATASET_NAME}_text_embeddings.pt").float()
    
    with open(outputs_dir / f"{DATASET_NAME}_metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Get embedding dimensions
    image_dim = image_embeddings.shape[1]
    text_dim = text_embeddings.shape[1]
    
    # Load projection models
    image_proj = ProjectionHead(image_dim).to(DEVICE)
    text_proj = ProjectionHead(text_dim).to(DEVICE)
    
    image_proj.load_state_dict(torch.load(f"{MODEL_PATH}_image_projection.pt", map_location=DEVICE))
    text_proj.load_state_dict(torch.load(f"{MODEL_PATH}_text_projection.pt", map_location=DEVICE))
    
    image_proj.eval()
    text_proj.eval()
    
    # Project all image embeddings to search space
    with torch.no_grad():
        image_embeddings = image_embeddings.to(DEVICE)
        projected_images = image_proj(image_embeddings)
        projected_images = F.normalize(projected_images, dim=-1)
    
    print(f"Loaded {len(metadata['image_paths'])} images for search")
    
    return text_proj, projected_images, metadata

def encode_text_query(text_query, text_proj):
    """Encode text query using E5-Mistral encoder and projection"""
    print(f"Encoding text query: '{text_query}'")
    
    # Load E5-Mistral tokenizer and model
    model_cache_dir = Path("pretrained_encoders")
    text_model_name = "intfloat/e5-mistral-7b-instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(
        text_model_name,
        cache_dir=model_cache_dir / "e5-mistral-7b"
    )
    model = AutoModel.from_pretrained(
        text_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=model_cache_dir / "e5-mistral-7b"
    )
    model.eval()
    
    # Process text with E5 prefix
    processed_text = f"passage: {text_query}"
    
    with torch.no_grad():
        # Tokenize
        inputs = tokenizer(processed_text, padding=True, truncation=True, 
                          return_tensors="pt", max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get text features
        outputs = model(**inputs)
        
        # E5 uses last hidden state + attention mask mean pooling
        hidden = outputs.last_hidden_state  # [1, L, H]
        mask = inputs['attention_mask'].unsqueeze(-1)  # [1, L, 1]
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        text_features = summed / counts
        text_features = F.normalize(text_features, dim=-1)
        
        # Project to search space
        text_features = text_features.float().to(DEVICE)
        projected_text = text_proj(text_features)
        projected_text = F.normalize(projected_text, dim=-1)
    
    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return projected_text

def retrieve_images(query_embedding, projected_images, metadata, k=5):
    """Retrieve top-k unique images using cosine similarity"""
    with torch.no_grad():
        # Compute cosine similarities
        similarities = torch.matmul(query_embedding, projected_images.T).squeeze()
        
        # Get all indices sorted by similarity (descending)
        sorted_indices = torch.argsort(similarities, descending=True).cpu().numpy()
        sorted_scores = similarities[sorted_indices].cpu().numpy()
    
    # Get unique images (avoid duplicates from multiple captions per image)
    seen_image_ids = set()
    results = []
    
    for idx, score in zip(sorted_indices, sorted_scores):
        if len(results) >= k:
            break
            
        image_id = metadata['image_ids'][idx]
        
        # Skip if we've already seen this image
        if image_id in seen_image_ids:
            continue
            
        seen_image_ids.add(image_id)
        
        image_path = Path(metadata['image_paths'][idx])
        caption = metadata['captions'][idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
            results.append({
                'image': image,
                'image_path': str(image_path),
                'score': float(score),
                'image_id': image_id,
                'caption': caption,
                'index': int(idx)
            })
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            continue
    
    return results

def display_results(query_text, results, k):
    """Display text query and top-k retrieved images"""
    n_results = len(results)
    cols = min(3, n_results)  # Max 3 columns
    rows = (n_results + cols - 1) // cols
    
    fig_height = max(4, rows * 4)
    plt.figure(figsize=(15, fig_height + 2))
    
    # Add title with query
    plt.suptitle(f"Top {k} Images for Query: '{query_text}'", 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Display images
    for i, result in enumerate(results):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(result['image'])
        plt.axis('off')
        
        # Create title with image info
        title = f"#{i+1} (Score: {result['score']:.3f})\n"
        title += f"ID: {result['image_id']}\n"
        title += f"Caption: {result['caption'][:50]}..."  # Truncate long captions
        
        plt.title(title, fontsize=12, pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for suptitle
    
    # Save plot
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / "image_retrieval_results.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print('  python retrieve_image.py "<text_query>" <k>')
        print("\nExamples:")
        print('  python retrieve_image.py "a cat sitting on a chair" 5')
        print('  python retrieve_image.py "people playing soccer" 3')
        sys.exit(1)
    
    text_query = sys.argv[1]
    k = int(sys.argv[2])
    
    # Load models and data
    text_proj, projected_images, metadata = load_models_and_data()
    
    # Encode text query
    query_embedding = encode_text_query(text_query, text_proj)
    
    # Retrieve images
    print(f"Searching for top {k} images...")
    results = retrieve_images(query_embedding, projected_images, metadata, k)
    
    if not results:
        print("No images could be loaded. Check image paths in metadata.")
        sys.exit(1)
    
    # Display results
    print(f"\nTop {len(results)} images for query: '{text_query}'")
    print("="*60)
    for i, result in enumerate(results):
        print(f"{i+1}. Image ID: {result['image_id']}")
        print(f"   Similarity: {result['score']:.3f}")
        print(f"   Caption: {result['caption']}")
        print(f"   Path: {result['image_path']}")
        print()
    
    # Show visualization
    display_results(text_query, results, k)
    print(f"Visualization saved to plots/image_retrieval_results.png")

if __name__ == "__main__":
    main()