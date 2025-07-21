# xclip-siglip


## Encoders

*Image encoder:* siglip2-giant-opt-patch16-256
*Text encoder:* E5-mistral-7b-instruct

## Encode the COCO dataset

### Data Setup
Place your COCO dataset in the following structure:
```
data/
├── train2017/              # Training images
├── val2017/                # Validation images  
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

### Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Encode validation set
python encoding.py data/val2017 data/annotations/captions_val2017.json

# Encode training set
python encoding.py data/train2017 data/annotations/captions_train2017.json
```

### Outputs
Files are saved to `outputs/` directory:
- `{dataset}_image_embeddings.pt` - Image embeddings tensor
- `{dataset}_text_embeddings.pt` - Text embeddings tensor  
- `{dataset}_metadata.json` - Dataset metadata and mappings

Example output files:
- `outputs/val2017_image_embeddings.pt`
- `outputs/train2017_text_embeddings.pt`

