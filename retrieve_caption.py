"""Image-to-Caption Retrieval using trained projection heads.

Usage:
  python retrieve_caption.py <image_id> <k>           # Use existing COCO image by ID
  python retrieve_caption.py <image_path> <k>         # Use new image file
  
Examples:
  python retrieve_caption.py 139 5                    # Top 5 captions for COCO image ID 139
  python retrieve_caption.py my_image.jpg 3           # Top 3 captions for new image
"""
import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModel

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
    
    # Project all text embeddings to search space
    with torch.no_grad():
        text_embeddings = text_embeddings.to(DEVICE)
        projected_text = text_proj(text_embeddings)
        projected_text = F.normalize(projected_text, dim=-1)
    
    print(f"Loaded {len(metadata['captions'])} captions for search")
    
    return image_proj, projected_text, metadata, image_embeddings

def encode_new_image(image_path, image_proj):
    """Encode a new image using the original SigLIP encoder and projection"""
    print(f"Encoding new image: {image_path}")
    
    # Load SigLIP processor and model
    model_cache_dir = Path("pretrained_encoders")
    image_model_name = "google/siglip2-giant-opt-patch16-256"
    
    processor = AutoProcessor.from_pretrained(
        image_model_name, 
        cache_dir=model_cache_dir / "siglip2-giant"
    )
    model = AutoModel.from_pretrained(
        image_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=model_cache_dir / "siglip2-giant"
    )
    model.eval()
    
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get image features
        image_features = model.get_image_features(**inputs)
        image_features = F.normalize(image_features, dim=-1)
        
        # Project to search space
        image_features = image_features.float().to(DEVICE)
        projected_image = image_proj(image_features)
        projected_image = F.normalize(projected_image, dim=-1)
    
    # Clean up
    del model, processor
    torch.cuda.empty_cache()
    
    return projected_image, image

def find_image_by_id(image_id, metadata, image_embeddings, image_proj):
    """Find image embedding by COCO image ID"""
    try:
        # Find the index of this image_id in metadata
        image_indices = [i for i, img_id in enumerate(metadata['image_ids']) if img_id == image_id]
        if not image_indices:
            print(f"Error: Image ID {image_id} not found in dataset")
            return None, None
        
        # Use first occurrence (any caption for this image)
        idx = image_indices[0]
        image_path = Path(metadata['image_paths'][idx])
        
        # Get pre-computed embedding and project it
        with torch.no_grad():
            image_emb = image_embeddings[idx:idx+1].to(DEVICE)
            projected_image = image_proj(image_emb)
            projected_image = F.normalize(projected_image, dim=-1)
        
        # Load image for display
        image = Image.open(image_path).convert("RGB")
        
        return projected_image, image
    
    except Exception as e:
        print(f"Error loading image ID {image_id}: {e}")
        return None, None

def retrieve_captions(query_embedding, projected_text, metadata, k=5):
    """Retrieve top-k captions using cosine similarity"""
    with torch.no_grad():
        # Compute cosine similarities
        similarities = torch.matmul(query_embedding, projected_text.T).squeeze()
        
        # Get top-k indices
        top_k_indices = torch.topk(similarities, k).indices.cpu().numpy()
        top_k_scores = torch.topk(similarities, k).values.cpu().numpy()
    
    # Get corresponding captions
    results = []
    for idx, score in zip(top_k_indices, top_k_scores):
        caption = metadata['captions'][idx]
        image_id = metadata['image_ids'][idx]
        results.append({
            'caption': caption,
            'score': float(score),
            'image_id': image_id,
            'index': int(idx)
        })
    
    return results

def display_results(image, results, query_info):
    """Display query image and top-k captions"""
    plt.figure(figsize=(12, 8))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Query Image\n{query_info}", fontsize=12)
    
    # Display captions
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    caption_text = "Top Retrieved Captions:\n\n"
    for i, result in enumerate(results):
        caption_text += f"{i+1}. {result['caption']}\n"
        caption_text += f"   Score: {result['score']:.3f}\n"
        caption_text += f"   Image ID: {result['image_id']}\n\n"
    
    plt.text(0.05, 0.95, caption_text, transform=plt.gca().transAxes, 
             fontsize=14, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / "retrieval_results.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python retrieve_caption.py <image_id> <k>     # Use COCO image ID")
        print("  python retrieve_caption.py <image_path> <k>   # Use new image file")
        print("\nExamples:")
        print("  python retrieve_caption.py 139 5")
        print("  python retrieve_caption.py my_image.jpg 3")
        sys.exit(1)
    
    query_input = sys.argv[1]
    k = int(sys.argv[2])
    
    # Load models and data
    image_proj, projected_text, metadata, image_embeddings = load_models_and_data()
    
    # Determine if input is image ID or file path
    try:
        # Try to parse as image ID
        image_id = int(query_input)
        query_embedding, image = find_image_by_id(image_id, metadata, image_embeddings, image_proj)
        query_info = f"COCO Image ID: {image_id}"
    except ValueError:
        # Treat as file path
        image_path = Path(query_input)
        if not image_path.exists():
            print(f"Error: Image file {image_path} not found")
            sys.exit(1)
        query_embedding, image = encode_new_image(image_path, image_proj)
        query_info = f"File: {image_path.name}"
    
    if query_embedding is None:
        sys.exit(1)
    
    # Retrieve captions
    print(f"Searching for top {k} captions...")
    results = retrieve_captions(query_embedding, projected_text, metadata, k)
    
    # Display results
    print(f"\nTop {k} captions for {query_info}:")
    print("="*60)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['caption']}")
        print(f"   Similarity: {result['score']:.3f}")
        print(f"   Source Image ID: {result['image_id']}")
        print()
    
    # Show visualization
    display_results(image, results, query_info)
    print(f"Visualization saved to plots/retrieval_results.png")

if __name__ == "__main__":
    main()