# Configuration Files

This directory contains YAML configuration files for different training setups.

## Available Configurations

### 🔥 **InfoNCE Configurations**

| Config | Description | Loss Function | Model | Best For |
|--------|-------------|---------------|-------|----------|
| **`sigmoid_infonce.yaml`** | SigLIP-style binary cross-entropy | Sigmoid InfoNCE | SigLIP Head | Balanced performance |
| **`softmax_infonce.yaml`** | CLIP-style cross-entropy | Softmax InfoNCE | CLIP Head | Strong baselines |
| **`queue_infonce.yaml`** | Hard negatives with queue | Queue InfoNCE | Attention Head | Advanced training |
| **`clip_learnable.yaml`** | CLIP with learnable temperature | CLIP loss | CLIP Head | Temperature optimization |

### 📜 **Legacy Configurations**

| Config | Description | Status |
|--------|-------------|--------|
| **`siglip.yaml`** | Original SigLIP config (updated) | ✅ Compatible |
| **`clip.yaml`** | Original CLIP config | ✅ Compatible |

### 📊 **Model Types**

- **`siglip`**: Simple projection (LayerNorm → Dropout → Linear)
- **`clip`**: MLP projection with optional learnable temperature
- **`attention`**: Multi-head self-attention based projection

### 🎯 **Loss Functions**

- **`sigmoid_infonce`**: Binary cross-entropy (SigLIP-style)
- **`softmax_infonce`**: Cross-entropy (CLIP-style) 
- **`queue_infonce`**: Hard negatives with memory queue
- **`clip`**: CLIP loss with learnable temperature

## Quick Start

```bash
# Train with different configurations
python train.py configs/sigmoid_infonce.yaml
python train.py configs/softmax_infonce.yaml
python train.py configs/queue_infonce.yaml

# Hyperparameter tuning (finds best config automatically)
python tune_hyperparams.py

# Train with best config from tuning
python train.py optuna_results/best_config.yaml
```

## Configuration Structure

```yaml
experiment_name: "your_experiment"
device: "auto"  # auto, cuda, mps, cpu
seed: 42

# Data files
data:
  train_image_embeddings: "train_image_embeddings.pt"
  train_text_embeddings: "train_text_embeddings.pt"
  # ... etc

# Model architecture  
model:
  type: "siglip"  # siglip, clip, attention
  output_dim: 512
  dropout: 0.1
  # ... model-specific params

# Training settings
training:
  batch_size: 1024
  learning_rate: 1e-4
  weight_decay: 0.01
  num_epochs: 50

# Loss function
loss:
  type: "sigmoid_infonce"
  temperature: 0.07

# Evaluation
evaluation:
  top_k: [1, 5, 10, 50]
  save_features: true
```

## Recommended Configurations

1. **For beginners**: Start with `softmax_infonce.yaml`
2. **For SigLIP research**: Use `sigmoid_infonce.yaml`  
3. **For advanced training**: Try `queue_infonce.yaml`
4. **For hyperparameter tuning**: Run `tune_hyperparams.py`

## Legacy Configurations

Old configurations in this directory may use deprecated loss names:
- `"siglip"` → now `"sigmoid_infonce"`
- `"infonce"` → now `"softmax_infonce"`

The system maintains backward compatibility, but new configurations use the updated names for clarity.