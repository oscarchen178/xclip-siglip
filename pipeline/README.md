# Cross-Modal Projection Head Pipeline

Train and evaluate projection heads for aligning SigLIP image embeddings with E5-Mistral text embeddings using COCO dataset.

## Quick Setup

**Required Directory Structure:**
```
xclip-siglip/
├── pretrain_encoded/          # Put pre-encoded embeddings here
│   ├── train2017_image_embeddings.pt
│   ├── train2017_text_embeddings.pt  
│   ├── train2017_metadata.json
│   ├── val2017_image_embeddings.pt
│   ├── val2017_text_embeddings.pt
│   └── val2017_metadata.json
└── pipeline/                  # This directory
```

**Essential Steps:**
1. **Get pre-encoded embeddings** (from `../encoding.py`)
2. **Split data**: `python split_train_data.py`
3. **Train**: `python train.py configs/siglip.yaml`
4. **Evaluate**: `python evaluate.py configs/siglip.yaml`

**Optional: Hyperparameter Tuning:**
```bash
python tune_hyperparams.py optuna_configs/default.yaml
```

## Model Architectures

**Available projection heads:**

1. **SigLIP** (`type: "siglip"`) - Simple layer norm + linear projection
```yaml
model:
  type: "siglip"
  output_dim: 512
  dropout: 0.0
```

2. **CLIP** (`type: "clip"`) - MLP with learnable temperature
```yaml
model:
  type: "clip"
  output_dim: 512
  hidden_dim: 1024
  dropout: 0.1
```

3. **Attention** (`type: "attention"`) - Multi-head self-attention
```yaml
model:
  type: "attention"
  output_dim: 512
  hidden_dim: 1024
  num_heads: 8
  num_layers: 2
  dropout: 0.1
```

4. **MLP** (`type: "mlp"`) - Simple multilayer perceptron baseline
```yaml
model:
  type: "mlp"
  output_dim: 512
  hidden_dim: 1024
  dropout: 0.1
```

## Loss Functions

**Available loss types:**
- **`sigmoid_infonce`**: SigLIP-style binary cross-entropy (multi-label)
- **`softmax_infonce`**: CLIP-style cross-entropy with multiple positives
- **`queue_infonce`**: Hard negative mining with embedding queue

## Configuration System

**Main configs** (`configs/`):
- `siglip.yaml` - SigLIP model with sigmoid loss
- `clip.yaml` - CLIP model with softmax loss + learnable temperature
- `attention.yaml` - Attention model configuration
- `mlp.yaml` - MLP baseline configuration

**Key config parameters:**
```yaml
base_data_dir: "../pretrain_encoded"  # Data directory
data:
  train_image_embeddings: "train_image_embeddings.pt"
  train_text_embeddings: "train_text_embeddings.pt"
  train_metadata: "train_metadata.json"
  val_metadata: "val2017_metadata.json"
```

## Hyperparameter Tuning

**Optuna configs** (`optuna_configs/`):
- `default.yaml` - Multi-model search (all architectures)
- `siglip.yaml` - SigLIP-specific optimization
- `clip.yaml` - CLIP-specific optimization
- `attention.yaml` - Attention architecture tuning
- `mlp.yaml` - MLP baseline optimization

**Usage:**
```bash
# Run hyperparameter tuning
python tune_hyperparams.py optuna_configs/default.yaml

# Model-specific tuning
python tune_hyperparams.py optuna_configs/siglip.yaml
python tune_hyperparams.py optuna_configs/attention.yaml
```

**Features:**
- Fast tuning with 100K sample subset
- ASHA pruning for early trial termination
- Persistent SQLite database storage
- Parallel execution support

## Data Splits

After running `split_train_data.py`:
- **Training** (~473K samples): Full dataset for final training
- **Training Subset** (~100K samples): Fast hyperparameter tuning
- **Validation** (~5K samples): Early stopping during training
- **Test** (~24K samples): Final evaluation

**Image-level splitting** prevents data leakage - no images appear in both train and test.

## Key Features

- **4 projection head architectures** with model-specific parameters
- **3 InfoNCE loss variants** with multi-caption COCO handling
- **Configurable data paths** via `base_data_dir`
- **Hyperparameter optimization** with Optuna + ASHA pruning
- **Image-level data splitting** to prevent leakage

## Results & Evaluation

**Evaluation metrics:**
- **I2T Recall@K**: Image-to-text retrieval (COCO-aware)
- **T2I Recall@K**: Text-to-image retrieval
- **Visualizations**: t-SNE and PCA plots

**Output files:**
- `results/{config_name}/best_model.pt` - Trained model
- `results/{config_name}/training_curves.png` - Loss curves
- `results/{config_name}/evaluation_results/` - Metrics and plots

## Quick Start Examples

```bash
# Basic training
python train.py configs/siglip.yaml
python evaluate.py configs/siglip.yaml

# Compare all architectures
for config in configs/*.yaml; do
    python train.py $config
    python evaluate.py $config
done

# Hyperparameter tuning + training
python tune_hyperparams.py optuna_configs/default.yaml
python train.py results/best_config.yaml  # Use optimized config
```

## Expected Performance

- **Training time**: ~30 minutes per model on modern GPU
- **Tuning time**: ~1-2 hours for 100 trials
- **R@1 scores**: 30-60% on COCO test set