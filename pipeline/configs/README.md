# Training Configuration Files

This directory contains YAML configuration files for training specific projection head models.

## Available Model Configurations

| Config | Model Type | Loss Function | Description |
|--------|------------|---------------|-------------|
| **`siglip.yaml`** | SigLIP | Sigmoid InfoNCE | Authentic SigLIP with low temperature |
| **`clip.yaml`** | CLIP | CLIP/Softmax InfoNCE | CLIP with learnable temperature |
| **`attention.yaml`** | Attention | Softmax InfoNCE | Multi-head self-attention projection |
| **`mlp.yaml`** | MLP | Softmax InfoNCE | Simple MLP baseline |
| **`optuna_best.yaml`** | Best | Variable | Best configuration from hyperparameter tuning |

## Model Architecture Details

- **`siglip`**: Simple projection (LayerNorm → Dropout → Linear) with sigmoid loss
- **`clip`**: MLP projection with learnable temperature scaling
- **`attention`**: Multi-head self-attention based projection
- **`mlp`**: Basic MLP baseline (Linear → GELU → Dropout → Linear)

## Quick Start

```bash
# Train with different model configurations
python train.py configs/siglip.yaml
python train.py configs/clip.yaml
python train.py configs/attention.yaml
python train.py configs/mlp.yaml

# Hyperparameter tuning for specific models
python tune_hyperparams.py optuna_configs/siglip.yaml
python tune_hyperparams.py optuna_configs/clip.yaml

# Train with best config from tuning
python train.py configs/optuna_best.yaml
```

## Hyperparameter Tuning Configurations

For hyperparameter tuning configurations, see the `../optuna_configs/` directory:
- `../optuna_configs/default.yaml` - Multi-model search (all architectures)
- `../optuna_configs/siglip.yaml` - SigLIP-focused optimization
- `../optuna_configs/clip.yaml` - CLIP-focused optimization  
- `../optuna_configs/attention.yaml` - Attention architecture tuning
- `../optuna_configs/mlp.yaml` - MLP baseline optimization

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