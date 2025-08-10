# Training Configuration Files

This directory contains YAML configuration files for training different projection head models.

## Available Configurations

| Config | Model Type | Loss Function | Description |
|--------|------------|---------------|-------------|
| **`siglip.yaml`** | SigLIP | Sigmoid InfoNCE | Authentic SigLIP with low temperature (0.01) |
| **`clip.yaml`** | CLIP | Softmax InfoNCE | CLIP with learnable temperature scaling |
| **`attention.yaml`** | Attention | Softmax InfoNCE | Multi-head self-attention projection |
| **`mlp.yaml`** | MLP | Softmax InfoNCE | Simple MLP baseline |
| **`optuna_best.yaml`** | Attention | Softmax InfoNCE | Best configuration from tuning |
| **`optuna_best_v2.yaml`** | Attention | Softmax InfoNCE | Alternative best configuration |

## Model Architecture Details

- **SigLIP**: LayerNorm → Dropout → Linear (simple & effective)
- **CLIP**: Linear → GELU → Dropout → Linear (MLP with learnable temp)
- **Attention**: Multi-head self-attention → Linear projection
- **MLP**: Linear → GELU → Dropout → Linear → LayerNorm (baseline)

## Configuration Features

- **`base_data_dir`** parameter for flexible data paths
- **Complete metadata paths** for COCO image ID handling
- **Learnable temperature** support in loss functions
- **Model-specific parameters** for each architecture

## Quick Start

```bash
# Train different architectures
python train.py configs/siglip.yaml      # SigLIP model
python train.py configs/clip.yaml        # CLIP model  
python train.py configs/attention.yaml   # Attention model
python train.py configs/mlp.yaml         # MLP baseline

# Use optimized configurations
python train.py configs/optuna_best.yaml
```

## Configuration Structure

**Complete config template:**
```yaml
experiment_name: "my_experiment"
device: "auto"
seed: 42
base_data_dir: "../pretrain_encoded"

data:
  train_image_embeddings: "train_image_embeddings.pt"
  train_text_embeddings: "train_text_embeddings.pt"
  train_metadata: "train_metadata.json"
  val_image_embeddings: "val_image_embeddings.pt"
  val_text_embeddings: "val_text_embeddings.pt"
  val_metadata: "val2017_metadata.json"
  test_image_embeddings: "test_image_embeddings.pt"
  test_text_embeddings: "test_text_embeddings.pt"
  test_metadata: "test_metadata.json"

model:
  type: "siglip"           # siglip, clip, attention, mlp
  output_dim: 512
  dropout: 0.0
  # Model-specific parameters:
  # hidden_dim: 1024       # for clip, attention, mlp
  # num_heads: 8           # for attention only
  # num_layers: 2          # for attention only

training:
  batch_size: 512
  learning_rate: 1e-4
  weight_decay: 0.01
  num_epochs: 50
  patience: 5
  max_grad_norm: 1.0

loss:
  type: "sigmoid_infonce"  # sigmoid_infonce, softmax_infonce, queue_infonce
  temperature: 0.07
  learnable_scale: false   # true for learnable temperature

evaluation:
  top_k: [1, 5, 10, 50]
  visualization_samples: 5000
  save_features: true
```

## Hyperparameter Tuning

For automated hyperparameter optimization, see `../optuna_configs/`:
- **`default.yaml`**: Multi-model search (all 4 architectures, 150 trials)
- **`siglip.yaml`**: SigLIP-specific (lower temps, sigmoid loss only)
- **`clip.yaml`**: CLIP-specific (learnable temperature, softmax loss)
- **`attention.yaml`**: Attention-specific (head/layer tuning)
- **`mlp.yaml`**: MLP baseline optimization

```bash
# Run hyperparameter tuning
python tune_hyperparams.py optuna_configs/default.yaml
python tune_hyperparams.py optuna_configs/siglip.yaml
```

## Model-Specific Recommendations

**SigLIP** (`siglip.yaml`):
- Low temperature (0.01)
- Minimal dropout (0.0)
- Sigmoid InfoNCE loss

**CLIP** (`clip.yaml`): 
- Learnable temperature
- Standard dropout (0.1)
- Softmax InfoNCE loss

**Attention** (`attention.yaml`):
- Multi-head attention (8 heads)
- Multiple layers (1-3)
- Softmax InfoNCE loss

**MLP** (`mlp.yaml`):
- Simple baseline architecture
- Hidden layer configuration
- Standard regularization
