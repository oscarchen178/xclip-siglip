"""Train projection heads for image-text alignment using SigLIP loss.

Usage:
  python train_projection.py

Loads train2017 (split 70/30 for train/test) and val2017 (for validation)
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ======================== CONFIGURATION ========================
TRAIN_SPLIT = 0.7           # 70% train, 30% test from train2017
HIDDEN_DIM = 512            # MLP hidden dimension
OUTPUT_DIM = 256            # Final projection dimension
BATCH_SIZE = 512            # Training batch size
LEARNING_RATE = 1e-3        # AdamW learning rate
WEIGHT_DECAY = 0.01         # L2 regularization
NUM_EPOCHS = 20             # Training epochs
TEMPERATURE = 0.07          # SigLIP temperature
NUM_WORKERS = 4             # DataLoader workers
# ===============================================================

class ProjectionHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
            nn.LayerNorm(OUTPUT_DIM)
        )
    
    def forward(self, x):
        return self.mlp(x)

class EmbeddingDataset(Dataset):
    def __init__(self, image_embeddings, text_embeddings):
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings
    
    def __len__(self):
        return len(self.image_embeddings)
    
    def __getitem__(self, idx):
        return self.image_embeddings[idx], self.text_embeddings[idx]

def siglip_loss(image_features, text_features):
    """SigLIP loss function"""
    batch_size = image_features.shape[0]
    
    # Normalize features
    image_features = nn.functional.normalize(image_features, dim=-1)
    text_features = nn.functional.normalize(text_features, dim=-1)
    
    # Compute similarity matrix
    logits = torch.matmul(image_features, text_features.T) / TEMPERATURE
    
    # Create labels (positive pairs are on diagonal)
    labels = torch.arange(batch_size, device=logits.device)
    
    # Compute cross-entropy loss for both directions
    loss_i2t = nn.functional.cross_entropy(logits, labels)
    loss_t2i = nn.functional.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2

def train_epoch(image_proj, text_proj, dataloader, optimizer, device):
    image_proj.train()
    text_proj.train()
    total_loss = 0
    
    for image_emb, text_emb in tqdm(dataloader, desc="Training"):
        image_emb, text_emb = image_emb.to(device), text_emb.to(device)
        
        # Project embeddings
        image_proj_emb = image_proj(image_emb)
        text_proj_emb = text_proj(text_emb)
        
        # Compute loss
        loss = siglip_loss(image_proj_emb, text_proj_emb)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(image_proj, text_proj, dataloader, device):
    image_proj.eval()
    text_proj.eval()
    total_loss = 0
    
    for image_emb, text_emb in dataloader:
        image_emb, text_emb = image_emb.to(device), text_emb.to(device)
        
        # Project embeddings
        image_proj_emb = image_proj(image_emb)
        text_proj_emb = text_proj(text_emb)
        
        # Compute loss
        loss = siglip_loss(image_proj_emb, text_proj_emb)
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outputs_dir = Path("outputs")
    
    # Load train2017 embeddings
    print("Loading train2017 embeddings...")
    train_image_emb = torch.load(outputs_dir / "train2017_image_embeddings.pt").float()
    train_text_emb = torch.load(outputs_dir / "train2017_text_embeddings.pt").float()
    
    # Load val2017 embeddings  
    print("Loading val2017 embeddings...")
    val_image_emb = torch.load(outputs_dir / "val2017_image_embeddings.pt").float()
    val_text_emb = torch.load(outputs_dir / "val2017_text_embeddings.pt").float()
    
    print(f"Train embeddings: {train_image_emb.shape[0]} pairs")
    print(f"Val embeddings: {val_image_emb.shape[0]} pairs")
    
    # Get dimensions
    image_dim = train_image_emb.shape[1]
    text_dim = train_text_emb.shape[1]
    
    # Create train dataset and split
    train_dataset = EmbeddingDataset(train_image_emb, train_text_emb)
    train_size = int(TRAIN_SPLIT * len(train_dataset))
    test_size = len(train_dataset) - train_size
    
    train_subset, test_subset = random_split(train_dataset, [train_size, test_size])
    
    print(f"Split train2017: {train_size} train, {test_size} test")
    
    # Create validation dataset
    val_dataset = EmbeddingDataset(val_image_emb, val_text_emb)
    
    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Create models
    image_proj = ProjectionHead(image_dim).to(device)
    text_proj = ProjectionHead(text_dim).to(device)
    
    # Setup optimizer
    optimizer = optim.AdamW(
        list(image_proj.parameters()) + list(text_proj.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    
    # Training loop
    print(f"Training for {NUM_EPOCHS} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train_epoch(image_proj, text_proj, train_loader, optimizer, device)
        
        # Validate
        val_loss = validate(image_proj, text_proj, val_loader, device)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            torch.save(image_proj.state_dict(), models_dir / "best_image_projection.pt")
            torch.save(text_proj.state_dict(), models_dir / "best_text_projection.pt")
            print(f"  ✅ New best model saved (val_loss: {val_loss:.4f})")
    
    # Final test evaluation
    test_loss = validate(image_proj, text_proj, test_loader, device)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    
    # Save final models and config
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    torch.save(image_proj.state_dict(), models_dir / "final_image_projection.pt")
    torch.save(text_proj.state_dict(), models_dir / "final_text_projection.pt")
    
    print(f"Training complete! Models saved to {models_dir}/")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()