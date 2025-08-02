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
- **Loss**: InfoNCE (Sigmoid/Softmax/Queue variants)
- **Expected**: 30-60% R@1 on COCO test set

## Advanced Usage

See `pipeline/README.md` for:
- Hyperparameter tuning with Optuna
- Multiple model architectures (SigLIP, CLIP, Attention, MLP)
- Configuration management
Files are saved to `outputs/` directory:
- `{dataset}_image_embeddings.pt` - Image embeddings tensor
- `{dataset}_text_embeddings.pt` - Text embeddings tensor  
- `{dataset}_metadata.json` - Dataset metadata and mappings

Example output files:
- `outputs/val2017_image_embeddings.pt`
- `outputs/train2017_text_embeddings.pt`

