# Hyperparameter Tuning Guide

This guide covers the automated hyperparameter tuning system built with Optuna for finding optimal configurations across different projection head architectures and training settings.

## Overview

The hyperparameter tuning system automatically explores combinations of:
- **Model architectures**: SigLIP, CLIP, Attention, MLP projection heads
- **Model parameters**: Output dimensions, dropout, architecture-specific settings
- **Loss functions**: Sigmoid InfoNCE, Softmax InfoNCE, Queue InfoNCE
- **Training hyperparameters**: Learning rate, batch size, weight decay, temperature

## Quick Start

```bash
# 1. Ensure data is split (creates tuning subset)
python split_train_data.py

# 2. Run hyperparameter tuning
python tune_hyperparams.py optuna_configs/default.yaml

# 3. Use best configuration for final training
python train.py optuna_results/best_config.yaml
```

## Configuration Files (`optuna_configs/`)

### Study Settings

```yaml
study_name: "xclip_siglip_tuning"     # Unique study identifier
n_trials: 100                        # Number of trials to run
n_jobs: 3                            # Parallel trials (adjust for hardware)
storage: "sqlite:///optuna_study.db" # Persistent storage
timeout: null                        # Max time in seconds (null = no limit)
```

### Search Space Configuration

#### Model Architecture Parameters

```yaml
search_space:
  # Primary architecture choice
  head_type:
    type: categorical
    choices: ['siglip', 'clip', 'attention', 'mlp']
  
  # Output embedding dimension
  output_dim:
    type: categorical
    choices: [256, 512, 768, 1024]
  
  # Regularization
  dropout:
    type: float
    low: 0.0
    high: 0.3
```

#### Architecture-Specific Parameters

```yaml
  # Hidden layer size (for clip/attention/mlp)
  hidden_dim:
    type: categorical
    choices: [512, 1024, 2048]
  
  # CLIP-specific: learnable temperature
  learnable_temp:
    type: categorical
    choices: [true, false]
  
  # Attention-specific: number of attention heads
  num_heads:
    type: categorical
    choices: [4, 8, 16]
  
  # Attention-specific: number of transformer layers
  num_layers:
    type: categorical
    choices: [1, 2, 3]
```

#### Loss Function Parameters

```yaml
  # Loss function type
  loss_type:
    type: categorical
    choices: ['sigmoid_infonce', 'softmax_infonce', 'queue_infonce']
  
  # Temperature scaling
  temperature:
    type: float
    low: 0.01
    high: 0.2
    log: true  # Log-scale sampling
  
  # Queue size (for queue_infonce only)
  queue_size:
    type: categorical
    choices: [2048, 4096, 8192]
```

#### Training Hyperparameters

```yaml
  # Batch size
  batch_size:
    type: categorical
    choices: [2048, 4096, 8192]
  
  # Learning rate (log-scale)
  learning_rate:
    type: float
    low: 1.0e-5
    high: 1.0e-2
    log: true
  
  # L2 regularization (log-scale)
  weight_decay:
    type: float
    low: 1.0e-6
    high: 1.0e-1
    log: true
```

## Key Features

### Fast Tuning with Subset
- Uses **100K sample subset** instead of full 473K dataset
- **~5-10x faster** trials while preserving relative performance rankings
- Tuning subset created automatically by `split_train_data.py`

### Parallel Execution
- Multiple trials run simultaneously (`n_jobs` parameter)
- Automatic GPU memory management and process isolation
- Clean error handling and recovery

### Persistent Results
- All trials saved to SQLite database
- Can interrupt and resume tuning at any time
- Results preserved across runs

### YAML-Based Configuration
- Easy to modify search spaces without touching code
- Version control friendly
- Multiple studies can use different configurations

## Advanced Usage

### Multiple Studies

Create different studies for focused exploration:

```yaml
# optuna_config_focused.yaml
study_name: "attention_focused_tuning"
search_space:
  head_type:
    choices: ['attention']  # Focus on best architecture
  output_dim:
    choices: [768, 1024, 1536]  # Explore higher dimensions
  num_heads:
    choices: [8, 12, 16, 20]  # Fine-tune attention heads
```

### Resuming Studies

```bash
# Automatically resumes from database
python tune_hyperparams.py

# Or with different config targeting same study
python tune_hyperparams.py --config optuna_config_resume.yaml
```

### Analyzing Results

```python
import optuna

# Load study
study = optuna.load_study("xclip_siglip_tuning", storage="sqlite:///optuna_study.db")

# Best results
print(f"Best R@1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Trial history
print(f"Total trials: {len(study.trials)}")
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"Completed trials: {len(completed)}")

# Parameter importance
importance = optuna.importance.get_param_importances(study)
print("Parameter importance:", importance)
```

### Visualization

```python
import optuna.visualization as vis

# Optimization history
fig1 = vis.plot_optimization_history(study)
fig1.show()

# Parameter importance
fig2 = vis.plot_param_importances(study)
fig2.show()

# Parameter relationships
fig3 = vis.plot_parallel_coordinate(study)
fig3.show()
```

## Performance Guidelines

### Hardware Requirements

| Setup | n_jobs | RAM | VRAM | Expected Time (100 trials) |
|-------|--------|-----|------|----------------------------|
| RTX 4090 + 64GB RAM | 3 | ~33GB | ~24GB | ~45 minutes |
| RTX 4090 + 32GB RAM | 2 | ~22GB | ~16GB | ~60 minutes |
| RTX 3080 + 16GB RAM | 1 | ~11GB | ~8GB | ~120 minutes |

### Optimization Tips

1. **Start with broad search**: Use default ranges first
2. **Focus promising areas**: Create new studies with narrower ranges around best results
3. **Monitor resource usage**: Adjust `n_jobs` based on available RAM
4. **Use early stopping**: ASHA pruner automatically stops unpromising trials

## Expected Results

### Typical Performance
- **Tuning R@1**: 10-15% on 100K subset with 8 epochs
- **Final R@1**: 25-40% on full dataset with 50 epochs
- **Best architectures**: Usually Attention > CLIP > SigLIP > MLP
- **Best losses**: Softmax InfoNCE typically outperforms others

### Common Findings
- **Higher output dimensions** (1024) often work better
- **Low dropout** (0.0-0.1) typically optimal
- **Large batch sizes** (4096-8192) improve stability
- **Attention heads**: 16 heads often optimal for attention architecture
- **Learning rates**: ~1e-4 commonly optimal

## Troubleshooting

### Common Issues

**CategoricalDistribution Error**
```
ValueError: CategoricalDistribution does not support dynamic value space
```
**Solution**: Parameter ranges changed between studies. Use new study name or revert to original ranges.

**OOM/Process Killed**
```
RuntimeError: CUDA out of memory
# or system process killed
```
**Solution**: Reduce `n_jobs` in config file.

**Slow Trials (>5 minutes each)**
```
Trial taking very long...
```
**Solution**: Ensure using tuning subset files (`train_tuning_*.pt`). Run `python split_train_data.py` if missing.

**DataLoader Worker Errors**
```
RuntimeError: DataLoader worker exited unexpectedly
```
**Solution**: Reduce `n_jobs` or `batch_size` in config.

### Debugging

Enable verbose logging:
```python
import logging
logging.getLogger("optuna").setLevel(logging.DEBUG)
```

Check database manually:
```bash
sqlite3 optuna_study.db
.tables
SELECT COUNT(*) FROM trials;
```

## Integration with Training Pipeline

### Generated Configuration

The best hyperparameters are automatically saved to `best_hyperparams_config.yaml`:

```yaml
# Best Hyperparameter Configuration from Optuna Tuning
# Trial 74: R@1 = 11.45% (Best result from 100 trials)

model:
  type: attention
  output_dim: 1024
  hidden_dim: 512
  dropout: 0.004132
  num_heads: 16
  num_layers: 1

loss:
  type: softmax_infonce
  temperature: 0.03817

training:
  batch_size: 4096
  learning_rate: 0.0001106
  weight_decay: 0.000005939
  num_epochs: 50
  # ... full training configuration
```

### Final Training

```bash
# Use optimized hyperparameters for final training
python train.py best_hyperparams_config.yaml

# Expected performance improvement
# Tuning: 11.45% R@1 (100K subset, 8 epochs)
# Final: 25-40% R@1 (473K dataset, 50 epochs)
```

## Best Practices

1. **Always run split_train_data.py first** to create tuning subset
2. **Start with default search space** before customizing
3. **Monitor memory usage** and adjust `n_jobs` accordingly
4. **Use meaningful study names** for different experiments
5. **Save important configurations** as separate YAML files
6. **Check failed trials** for patterns indicating systematic issues
7. **Use focused studies** to fine-tune promising configurations

## File Structure After Tuning

```
pipeline/
├── optuna_study.db                    # Study database
├── optuna_config.yaml                 # Tuning configuration
├── best_hyperparams_config.yaml       # Best configuration (auto-generated)
├── checkpoints/                       # Trial checkpoints
│   ├── 00074/                        # Best trial checkpoints
│   │   ├── best_model.pt
│   │   └── cfg.yaml
│   └── ...
└── optuna_results/                    # Study results (if configured)
    ├── best_results.json
    └── study_summary.json
```

This hyperparameter tuning system provides a robust, scalable approach to finding optimal configurations across the complex space of projection head architectures and training hyperparameters.