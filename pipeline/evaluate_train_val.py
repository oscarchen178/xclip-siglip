#!/usr/bin/env python3
"""
Evaluation script for tuning subsets using functions from evaluate.py.

This script evaluates trained models on:
1. Validation set (val2017 - ~25K samples) 
2. Training tuning subset (~100K samples from train split)

Usage:
    python evaluate_tuning.py [config_file]
    python evaluate_tuning.py model_path [config_file]  # Legacy support
    
Examples:
    python evaluate_tuning.py configs/optuna_best_v2.yaml
    python evaluate_tuning.py results/optuna_best_v2/best_model.pt
"""

import sys
import torch
import yaml
import json
from pathlib import Path
from torch.utils.data import DataLoader

# Import functions from existing evaluate.py
from evaluate import (
    load_model_from_checkpoint,
    extract_features, 
    compute_retrieval_metrics,
    create_visualizations
)
from dataset import create_eval_dataset


def evaluate_on_dataset(image_head, text_head, dataset_name, config, device):
    """Evaluate model on a specific dataset."""
    
    print(f"\n{'='*60}")
    print(f"EVALUATING ON {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Map dataset names to file paths
    dataset_configs = {
        'validation': {
            'image_embeddings': 'val_image_embeddings.pt',
            'text_embeddings': 'val_text_embeddings.pt', 
            'metadata': 'val2017_metadata.json'
        },
        'train_tuning': {
            'image_embeddings': 'train_tuning_image_embeddings.pt',
            'text_embeddings': 'train_tuning_text_embeddings.pt',
            'metadata': 'train_tuning_metadata.json'
        }
    }
    
    if dataset_name not in dataset_configs:
        print(f"Error: Unknown dataset '{dataset_name}'")
        return None
    
    dataset_config = dataset_configs[dataset_name]
    
    # Create evaluation dataset
    base_dir = config.get('base_data_dir', '../pretrain_encoded')  # Fallback for legacy configs
    eval_dataset = create_eval_dataset(
        base_dir,
        dataset_config['image_embeddings'],
        dataset_config['text_embeddings'], 
        dataset_config['metadata']
    )
    
    # Create data loader with device-optimized settings
    pin_memory = device.type == 'cuda'
    num_workers = 4 if device.type != 'mps' else 0
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Dataset samples: {len(eval_dataset)}")
    
    # Extract features
    print("Extracting features...")
    image_features, text_features, image_ids = extract_features(
        image_head, text_head, eval_loader, device, pin_memory
    )
    
    # Compute retrieval metrics
    print("Computing retrieval metrics...")
    metrics, _ = compute_retrieval_metrics(
        image_features, text_features, config['evaluation']['top_k'], image_ids
    )
    
    # Print results
    print(f"\nRETRIEVAL RESULTS - {dataset_name.upper()}")
    print("-" * 50)
    
    for k in config['evaluation']['top_k']:
        print(f"I2T Recall@{k}: {metrics[f'i2t_recall@{k}']:.4f}")
        print(f"T2I Recall@{k}: {metrics[f't2i_recall@{k}']:.4f}")
        print("-" * 30)
    
    print(f"Mean Recall: {metrics['mean_recall']:.4f}")
    print(f"I2T Median Rank: {metrics['i2t_median_rank']:.1f}")
    print(f"T2I Median Rank: {metrics['t2i_median_rank']:.1f}")
    
    return {
        'metrics': metrics,
        'features': (image_features, text_features),
        'sample_count': len(eval_dataset)
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_tuning.py [config_file]")
        print("       python evaluate_tuning.py model_path [config_file]  # Legacy")
        sys.exit(1)
    
    first_arg = sys.argv[1]
    
    # Check if first argument is a model path (legacy) or config file
    if first_arg.endswith('.pt'):
        # Legacy mode: model_path [config_file]
        model_path = first_arg
        config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
        print(f"Legacy mode: Evaluating model: {model_path}")
    else:
        # New mode: config_file
        config_path = first_arg
        # Load config to determine model path
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config_name = Path(config_path).stem
        model_path = Path(config['output_dir']) / config_name / 'best_model.pt'
        print(f"Config mode: Using config {config_path}")
        print(f"Model path: {model_path}")
    
    # Load config (if not already loaded)
    if 'config' not in locals():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Setup device
    if config['device'] == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using device: {device} ({torch.cuda.get_device_name()})") 
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
            print(f"Using device: {device} (Apple Silicon GPU)")
        else:
            device = torch.device('cpu')
            print(f"Using device: {device}")
    else:
        device = torch.device(config['device'])
        print(f"Using device: {device}")
    
    # Set MPS compatibility settings
    if device.type == 'mps':
        torch.backends.mps.fallback_enabled = True
    
    # Load model
    image_head, text_head, model_config = load_model_from_checkpoint(model_path, device)
    print(f"Model loaded successfully! Type: {model_config['model']['type']}")
    
    # Evaluate on both datasets
    results = {}
    datasets_to_evaluate = ['validation', 'train_tuning']
    
    for dataset_name in datasets_to_evaluate:
        try:
            result = evaluate_on_dataset(image_head, text_head, dataset_name, config, device)
            if result:
                results[dataset_name] = result
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            continue
    
    # Save combined results
    output_dir = Path(model_path).parent / "tuning_evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics for each dataset
    combined_metrics = {}
    for dataset_name, result in results.items():
        combined_metrics[dataset_name] = result['metrics']
        
        # Save individual dataset metrics
        with open(output_dir / f"{dataset_name}_metrics.json", 'w') as f:
            json.dump(result['metrics'], f, indent=2, default=float)
        
        # Save features if requested
        if config['evaluation']['save_features']:
            image_features, text_features = result['features']
            torch.save(image_features, output_dir / f"{dataset_name}_image_features.pt")
            torch.save(text_features, output_dir / f"{dataset_name}_text_features.pt")
    
    # Save combined metrics
    with open(output_dir / "combined_metrics.json", 'w') as f:
        json.dump(combined_metrics, f, indent=2, default=float)
    
    # Print summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} {'Samples':<10} {'I2T R@1':<10} {'T2I R@1':<10} {'Mean R@1':<10}")
    print("-" * 80)
    
    for dataset_name, result in results.items():
        metrics = result['metrics']
        i2t_r1 = metrics['i2t_recall@1'] 
        t2i_r1 = metrics['t2i_recall@1']
        mean_r1 = (i2t_r1 + t2i_r1) / 2
        sample_count = result['sample_count']
        
        print(f"{dataset_name:<15} {sample_count:<10} {i2t_r1:<10.4f} {t2i_r1:<10.4f} {mean_r1:<10.4f}")
    
    # Create visualizations for validation set (smaller, faster)
    if 'validation' in results and config['evaluation'].get('save_features', False):
        print(f"\nCreating visualizations for validation set...")
        val_features = results['validation']['features']
        create_visualizations(
            val_features[0], val_features[1], 
            output_dir / "validation_visualizations", 
            config
        )
    
    print(f"\nTuning evaluation complete! Results saved to {output_dir}")
    print(f"Use these metrics to compare different model configurations.")


if __name__ == "__main__":
    main()