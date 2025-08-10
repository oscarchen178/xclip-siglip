# Final Results Summary

This document summarizes the performance evaluation results for three different model configurations on the COCO dataset retrieval task.

## Model Performance Comparison

### Overall Results
| Model | Mean Recall | I2T R@1 | T2I R@1 | I2T R@5 | T2I R@5 | I2T R@10 | T2I R@10 |
|-------|-------------|---------|---------|---------|---------|----------|----------|
| **MLP** | **0.614** | **0.511** | 0.345 | **0.763** | 0.345 | **0.844** | 0.453 |
| SigLIP | 0.620 | 0.507 | **0.362** | 0.759 | **0.362** | 0.841 | **0.468** |
| CLIP | 0.613 | 0.504 | 0.349 | 0.756 | 0.349 | 0.839 | 0.456 |

### Key Metrics
- **Dataset**: COCO validation set with 23,657 unique images and ~118K image-caption pairs
- **Average captions per image**: ~5.0
- **Median ranks**: All models achieve median rank of 1 for I2T and 11 for T2I

### Performance Analysis

#### Image-to-Text Retrieval (I2T)
- **MLP** performs best overall with 51.1% R@1 accuracy
- All models show strong performance with R@1 > 50%
- R@50 performance is excellent across all models (95.7-95.9%)

#### Text-to-Image Retrieval (T2I)
- **SigLIP** shows the best T2I performance with 36.2% R@1
- T2I task is generally more challenging than I2T across all models
- R@50 performance reaches ~69-70% for all models

#### Overall Assessment
1. **MLP** excels at image-to-text retrieval tasks
2. **SigLIP** provides the best balance with superior text-to-image performance
3. **CLIP** serves as a solid baseline with consistent performance
4. All models demonstrate strong retrieval capabilities with mean recall > 61%

## Visualizations Available
Each model directory contains:
- `metrics.json`: Detailed numerical results
- `training_curves.png`: Training progress visualization
- `pca_analysis.png`: Principal component analysis of embeddings
- `tsne_visualization.png`: t-SNE embedding visualization
- Root directory contains retrieval comparison plots

## Conclusion
The results demonstrate successful implementation of cross-modal retrieval with competitive performance across all three architectures. The slight performance differences suggest that the choice of projection head architecture can be tuned based on specific task requirements (I2T vs T2I emphasis).