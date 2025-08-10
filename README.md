# SigLIP-E5 Cross-Modal Retrieval

Cross-modal image-text retrieval using SigLIP2 and E5-Mistral encoders on COCO dataset.

## Quick Start

### 1. Data Setup
```
data/
├── train2017/              # Training images
├── val2017/                # Validation images  
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

### 2. Install & Encode
```bash
pip install -r requirements.txt

# Encode datasets
python encoding.py data/val2017 data/annotations/captions_val2017.json
python encoding.py data/train2017 data/annotations/captions_train2017.json
```

### 3. Train Pipeline
```bash
cd pipeline
python split_train_data.py                    # Split data
python train.py configs/siglip.yaml           # Train model
python evaluate.py configs/siglip.yaml        # Evaluate
```

## Architecture

- **Image**: SigLIP2-Giant (1536D) → Projection Head → 512D
- **Text**: E5-Mistral-7B (4096D) → Projection Head → 512D  
- **Loss**: InfoNCE variants with proper COCO image ID matching
- **Expected**: 30-60% R@1 on COCO test set

## Key Features

- **4 projection heads**: SigLIP, CLIP, Attention, MLP
- **3 loss functions**: Sigmoid InfoNCE, Softmax InfoNCE, Queue InfoNCE
- **COCO-aware evaluation** with proper 1:5 image-to-caption mapping
- **Configurable data paths** via `base_data_dir`
- **Hyperparameter tuning** with Optuna + ASHA pruning

## Advanced Usage

See `pipeline/README.md` for:
- Hyperparameter tuning configurations
- Multiple model architectures and loss functions
- Training and evaluation workflows

### Output Structure
Files are saved to `pretrain_encoded/` directory:
- `{dataset}_image_embeddings.pt` - Image embeddings tensor
- `{dataset}_text_embeddings.pt` - Text embeddings tensor
- `{dataset}_metadata.json` - COCO image IDs and mappings

Example files:
- `pretrain_encoded/val2017_image_embeddings.pt`
- `pretrain_encoded/train2017_text_embeddings.pt`
- `pretrain_encoded/val2017_metadata.json`

