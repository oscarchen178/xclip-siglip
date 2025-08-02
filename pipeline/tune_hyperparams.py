#!/usr/bin/env python3
"""
Hyperparameter tuning with Optuna + ASHA for cross-modal projection heads.

Configuration via optuna_config.yaml:
  study_name: "cross_modal_tuning"
  n_trials: 100
  timeout: 3600  # seconds
  storage: null  # or "sqlite:///optuna.db"
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import optuna
import torch
import torch.nn as nn
import yaml
from optuna.pruners import SuccessiveHalvingPruner
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import SigLIPProjectionHead, CLIPProjectionHead, AttentionProjectionHead, MLPProjectionHead
from losses import compute_loss, clear_loss_instances
from metrics import compute_validation_metrics
from dataset import create_train_dataset, create_eval_dataset


# Configure logging
logging.basicConfig(level=logging.INFO)
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))




def create_projection_heads(trial, image_dim: int, text_dim: int, device: torch.device):
    """Create projection heads based on trial suggestions."""
    # Suggest architecture
    head_type = trial.suggest_categorical('head_type', ['siglip', 'clip', 'attention', 'mlp'])
    output_dim = trial.suggest_categorical('output_dim', [256, 512, 768, 1024])
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    
    if head_type == 'siglip':
        image_head = SigLIPProjectionHead(image_dim, output_dim, dropout)
        text_head = SigLIPProjectionHead(text_dim, output_dim, dropout)
        
    elif head_type == 'clip':
        hidden_dim = trial.suggest_categorical('hidden_dim', [512, 1024, 2048])
        learnable_temp = trial.suggest_categorical('learnable_temp', [True, False])
        
        image_head = CLIPProjectionHead(image_dim, output_dim, hidden_dim, dropout, learnable_temp)
        text_head = CLIPProjectionHead(text_dim, output_dim, hidden_dim, dropout, learnable_temp)
        
    elif head_type == 'attention':
        hidden_dim = trial.suggest_categorical('hidden_dim', [512, 1024, 2048])
        num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
        num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
        
        image_head = AttentionProjectionHead(image_dim, output_dim, hidden_dim, num_heads, num_layers, dropout)
        text_head = AttentionProjectionHead(text_dim, output_dim, hidden_dim, num_heads, num_layers, dropout)
        
    elif head_type == 'mlp':
        hidden_dim = trial.suggest_categorical('hidden_dim', [512, 1024, 2048])
        
        image_head = MLPProjectionHead(image_dim, output_dim, hidden_dim, dropout)
        text_head = MLPProjectionHead(text_dim, output_dim, hidden_dim, dropout)
    
    # Move to device  
    image_head = image_head.to(device)
    text_head = text_head.to(device)
    
    return image_head, text_head


def create_config(trial, search_space) -> Dict[str, Any]:
    """Create training config based on trial suggestions from search space config."""
    # Model configuration
    head_type = trial.suggest_categorical('head_type', search_space['head_type']['choices'])
    output_dim = trial.suggest_categorical('output_dim', search_space['output_dim']['choices'])
    dropout = trial.suggest_float('dropout', search_space['dropout']['low'], search_space['dropout']['high'])
    
    # Loss configuration
    loss_type = trial.suggest_categorical('loss_type', search_space['loss_type']['choices'])
    temperature = trial.suggest_float('temperature', 
                                    search_space['temperature']['low'], 
                                    search_space['temperature']['high'], 
                                    log=search_space['temperature']['log'])
    
    # Training hyperparameters
    batch_size = trial.suggest_categorical('batch_size', search_space['batch_size']['choices'])
    learning_rate = trial.suggest_float('learning_rate', 
                                      search_space['learning_rate']['low'], 
                                      search_space['learning_rate']['high'], 
                                      log=search_space['learning_rate']['log'])
    weight_decay = trial.suggest_float('weight_decay', 
                                     search_space['weight_decay']['low'], 
                                     search_space['weight_decay']['high'], 
                                     log=search_space['weight_decay']['log'])
    
    config = {
        'model': {
            'type': head_type,
            'output_dim': output_dim,
            'dropout': dropout
        },
        'loss': {
            'type': loss_type,
            'temperature': temperature
        },
        'training': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'max_grad_norm': 1.0
        }
    }
    
    # Add model-specific params
    if head_type in ['clip', 'attention', 'mlp']:
        config['model']['hidden_dim'] = trial.suggest_categorical('hidden_dim', search_space['hidden_dim']['choices'])
    
    if head_type == 'clip':
        config['model']['learnable_temp'] = trial.suggest_categorical('learnable_temp', search_space['learnable_temp']['choices'])
    
    if head_type == 'attention':
        config['model']['num_heads'] = trial.suggest_categorical('num_heads', search_space['num_heads']['choices'])
        config['model']['num_layers'] = trial.suggest_categorical('num_layers', search_space['num_layers']['choices'])
    
    if loss_type == 'queue_infonce':
        config['loss']['queue_size'] = trial.suggest_categorical('queue_size', search_space['queue_size']['choices'])
        config['loss']['feature_dim'] = config['model']['output_dim']
    
    return config


def train_epoch_simple(image_head, text_head, train_loader, optimizer, config, device):
    """Simplified training for one epoch."""
    image_head.train()
    text_head.train()
    
    total_loss = 0
    for image_emb, text_emb, img_ids in train_loader:
        image_emb = image_emb.to(device)
        text_emb = text_emb.to(device)
        
        # Forward pass
        img_proj = image_head(image_emb)
        txt_proj = text_head(text_emb)
        
        # Compute loss
        loss = compute_loss(img_proj, txt_proj, config, image_head, text_head, img_ids=img_ids)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(image_head.parameters()) + list(text_head.parameters()),
            config['training']['max_grad_norm']
        )
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def create_objective_function(tuning_config):
    """Create objective function with access to tuning config."""
    def objective(trial):
        """Optuna objective function."""
        # Clear cached loss instances to ensure fresh state for each trial
        clear_loss_instances()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Create datasets (use tuning subset for faster hyperparameter search)
        train_dataset = create_train_dataset("../pretrain_encoded", "train_tuning_image_embeddings.pt", "train_tuning_text_embeddings.pt", "train_tuning_metadata.json")
        val_dataset = create_eval_dataset("../pretrain_encoded", "val_image_embeddings.pt", "val_text_embeddings.pt", "val_metadata.json")
        
        # Get dimensions
        image_dim = train_dataset.image_embeddings.shape[1]
        text_dim = train_dataset.text_embeddings.shape[1]
        
        # Create config and models
        config = create_config(trial, tuning_config['search_space'])
        image_head, text_head = create_projection_heads(trial, image_dim, text_dim, device)
        
        # Create data loaders
        batch_size = config['training']['batch_size']
        num_workers = 4 if device.type != 'mps' else 0
        pin_memory = device.type == 'cuda'
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers, pin_memory=pin_memory)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            list(image_head.parameters()) + list(text_head.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Training loop with early reporting for ASHA
        best_val_r1 = 0.0
        patience_counter = 0
        max_patience = 2  # Reduced patience for faster pruning
        
        for epoch in range(8):  # Reduced epochs for faster tuning
            # Train
            _ = train_epoch_simple(image_head, text_head, train_loader, optimizer, config, device)
            
            # Validate (fast R@1 for ASHA)
            val_metrics = compute_validation_metrics(image_head, text_head, val_loader, device, fast_mode=True)
            val_r1 = val_metrics['val_recall_1']
            
            # Report intermediate value for pruning
            trial.report(val_r1, epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping
            if val_r1 > best_val_r1:
                best_val_r1 = val_r1
                patience_counter = 0
                # --- save model & cfg ---
                ckpt_dir = Path("checkpoints") / f"{trial.number:05d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": image_head.state_dict(),
                            "text_model": text_head.state_dict()},
                           ckpt_dir / "best_model.pt")
                yaml.safe_dump(config, open(ckpt_dir / "cfg.yaml", "w"))
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                break
    
        return best_val_r1
    
    return objective


def main():
    # Load config
    config_path = "optuna_config.yaml"
    
    # Default config if file doesn't exist (basic version - should use YAML file)
    default_config = {
        'study_name': 'cross_modal_tuning',
        'n_trials': 100,
        'timeout': None,
        'storage': None,
        'n_jobs': 1,
        'search_space': {
            'head_type': {'choices': ['siglip', 'clip', 'attention', 'mlp']},
            'output_dim': {'choices': [256, 512, 768, 1024]},
            'dropout': {'low': 0.0, 'high': 0.3},
            'loss_type': {'choices': ['sigmoid_infonce', 'softmax_infonce', 'queue_infonce']},
            'temperature': {'low': 0.01, 'high': 0.2, 'log': True},
            'batch_size': {'choices': [2048, 4096, 8192]},
            'learning_rate': {'low': 1e-5, 'high': 1e-2, 'log': True},
            'weight_decay': {'low': 1e-6, 'high': 1e-1, 'log': True},
            'hidden_dim': {'choices': [512, 1024, 2048]},
            'learnable_temp': {'choices': [True, False]},
            'num_heads': {'choices': [4, 8, 16]},
            'num_layers': {'choices': [1, 2, 3]},
            'queue_size': {'choices': [2048, 4096, 8192]}
        }
    }
    
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            tuning_config = yaml.safe_load(f)
    else:
        tuning_config = default_config
        # Save default config
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, indent=2)
        print(f"Created default config: {config_path}")
    
    # Create output directory
    output_dir = Path("optuna_results")
    output_dir.mkdir(exist_ok=True)
    
    # Configure pruner (ASHA)
    pruner = SuccessiveHalvingPruner(
        min_resource=2,      # Minimum epochs before pruning (reduced)
        reduction_factor=4,  # More aggressive pruning for speed
        min_early_stopping_rate=1  # Less exploration for speed
    )
    
    # Create or load study
    storage_url = tuning_config['storage'] or f"sqlite:///{output_dir}/optuna_study.db"
    
    study = optuna.create_study(
        study_name=tuning_config['study_name'],
        direction='maximize',
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True
    )
    
    print(f"Starting hyperparameter tuning:")
    print(f"  Study: {tuning_config['study_name']}")
    print(f"  Trials: {tuning_config['n_trials']}")
    print(f"  Parallel jobs: {tuning_config.get('n_jobs', 1)}")
    print(f"  Storage: {storage_url}")
    device_name = 'CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'
    print(f"  Device: {device_name}")
    
    # Run optimization with progress bar
    start_time = time.time()
    
    # Create progress bar with better formatting for parallel execution
    progress_bar = tqdm(
        total=tuning_config['n_trials'], 
        desc="Hyperparameter Tuning", 
        unit="trial",
        position=0,
        leave=True,
        dynamic_ncols=True
    )
    
    def update_progress(study, trial):
        progress_bar.update(1)
        progress_bar.set_postfix({
            'Best': f"{study.best_value:.4f}" if study.best_value else "N/A",
            'Trial': trial.number + 1,
            'Jobs': tuning_config.get('n_jobs', 1)
        })
        
        # Print trial completion info above progress bar
        if trial.state.name == 'COMPLETE':
            tqdm.write(f"Trial {trial.number + 1} completed: value={trial.value:.4f}, params={trial.params}")
        elif trial.state.name == 'FAIL':
            tqdm.write(f"Trial {trial.number + 1} failed: {trial.state.name}")
        elif trial.state.name == 'PRUNED':
            tqdm.write(f"Trial {trial.number + 1} pruned at step {trial.last_step}")
    
    try:
        objective_fn = create_objective_function(tuning_config)
        study.optimize(objective_fn, n_trials=tuning_config['n_trials'], timeout=tuning_config['timeout'], 
                      n_jobs=tuning_config.get('n_jobs', 1), callbacks=[update_progress])
    except KeyboardInterrupt:
        print("Optimization interrupted by user")
    finally:
        progress_bar.close()
    
    elapsed_time = time.time() - start_time
    
    # Save results
    print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (R@1): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    # Create training config from best trial 
    best_config = create_config(study.best_trial)
    
    # Add metadata
    best_config['experiment_name'] = f"best_{tuning_config['study_name']}"
    best_config['output_dir'] = "results"
    best_config['device'] = "auto"
    best_config['seed'] = 42
    
    # Add data paths
    best_config['data'] = {
        'train_image_embeddings': "train_image_embeddings.pt",
        'train_text_embeddings': "train_text_embeddings.pt",
        'val_image_embeddings': "val_image_embeddings.pt", 
        'val_text_embeddings': "val_text_embeddings.pt",
        'test_image_embeddings': "test_image_embeddings.pt",
        'test_text_embeddings': "test_text_embeddings.pt"
    }
    
    # Add full training settings
    best_config['training'].update({
        'num_epochs': 50,
        'patience': 5
    })
    
    # Add evaluation settings
    best_config['evaluation'] = {
        'top_k': [1, 5, 10, 50],
        'visualization_samples': 5000,
        'tsne_perplexity': 30,
        'save_features': True
    }
    
    # Save results
    best_results = {
        'best_trial_number': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'elapsed_time': elapsed_time,
        'tuning_config': tuning_config
    }
    
    with open(output_dir / "best_results.json", "w") as f:
        json.dump(best_results, f, indent=2)
    
    with open(output_dir / "best_config.yaml", "w") as f:
        yaml.dump(best_config, f, indent=2)
    
    print(f"\nBest configuration saved to:")
    print(f"  {output_dir}/best_results.json")
    print(f"  {output_dir}/best_config.yaml")
    
    # Print study statistics
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    print(f"\nStudy statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")
    
    print(f"\nTo train with best config:")
    print(f"  cd pipeline && python train.py {output_dir}/best_config.yaml")


if __name__ == "__main__":
    main()