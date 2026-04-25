import scanpy as sc 
import anndata
import nicheformer
print(nicheformer.__file__)
import sys
sys.path.insert(0, "/maiziezhou_lab2/yuling/nicheformer/src")

import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import anndata as ad
from typing import Optional, Dict, Any
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from nicheformer.models._nicheformer import Nicheformer, CosineWarmupScheduler
from nicheformer.models._nicheformer_fine_tune import NicheformerFineTune
from nicheformer.data.dataset import NicheformerDataset

config = {
    'data_path': '/maiziezhou_lab2/yuling/nicheformer/data/HumanLymph/xenium_human_lymph_ready_to_tokenize.h5ad',  # Path to your AnnData file
    'technology_mean_path': '/maiziezhou_lab2/yuling/nicheformer/data/model_means/xenium_mean_script.npy',  # Path to technology mean file
    'checkpoint_path': '/maiziezhou_lab2/yuling/nicheformer/data/nicheformer.ckpt',  # Path to pre-trained model
    'output_path': '/maiziezhou_lab2/yuling/nicheformer/data/HumanLymph/predictions_fineTune.h5ad',  # Where to save results
    'output_dir': '/maiziezhou_lab2/yuling/nicheformer/data/HumanLymph/directory',  # Directory for checkpoints
    
    # Training parameters
    'batch_size': 32,
    'max_seq_len': 1500,
    'aux_tokens': 30,
    'chunk_size': 1000,
    'num_workers': 4,
    'precision': 32,
    'max_epochs': 100,
    'lr': 1e-4,
    'warmup': 10,
    'gradient_clip_val': 1.0,
    'accumulate_grad_batches': 10,
    
    # Model parameters
    'supervised_task': 'niche_classification',  # classification task
    'extract_layers': [11],  # Which layers to extract features from
    'function_layers': "mean",  # Architecture of prediction head
    'dim_prediction': 1,  # keep 1 for classification reshape
    'n_classes': 1,  # will be updated after encoding labels
    'freeze': False,  # Set to False for partial fine-tuning (True = linear probing)
    'freeze_layers': [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Which layers to freeze (0-indexed). Only used if freeze=False
    'backbone_lr': 1e-5,  # Learning rate for backbone (unfrozen layers). Only used if freeze=False
    'head_lr': 1e-4,  # Learning rate for classification head
    'reinit_layers': False,
    'extractor': False,
    'regress_distribution': False,
    'pool': 'mean',
    'predict_density': False,
    'ignore_zeros': False,
    'organ': 'brain',
    'label': 'original_clusters'  # Use obsm vector as target
}
# Set random seed for reproducibility
pl.seed_everything(42)

# Load data
adata = ad.read_h5ad(config['data_path'])
adata.obs_names = adata.obs_names.astype(int).astype(str)

label_col = config['label']
if label_col not in adata.obs:
    raise KeyError(f"obs['{label_col}'] not found")
adata.obs[label_col] = adata.obs[label_col].astype('category').cat.codes.astype(int)

config['n_classes'] = int(adata.obs[label_col].nunique())
for col in ['modality', 'assay', 'specie']:
    if col not in adata.obs:
        adata.obs[col] = 0
    adata.obs[col] = adata.obs[col].astype(int)

technology_mean = np.load(config['technology_mean_path'])

# Create datasets
common_obs = [label_col, 'modality', 'assay', 'specie']  # needed by model

train_dataset = NicheformerDataset(
    adata=adata,
    technology_mean=technology_mean,
    split='train',
    max_seq_len=1500,
    aux_tokens=config.get('aux_tokens', 30),
    chunk_size=config.get('chunk_size', 1000),
    metadata_fields = {
        'obs': common_obs,
        #'obsm': [label_col],
    }
)

val_dataset = NicheformerDataset(
    adata=adata,
    technology_mean=technology_mean,
    split='val',
    max_seq_len=1500,
    aux_tokens=config.get('aux_tokens', 30),
    chunk_size=config.get('chunk_size', 1000),
    metadata_fields = {
        'obs': common_obs,
        #'obsm': [label_col],
    }
)

test_dataset = NicheformerDataset(
    adata=adata,
    technology_mean=technology_mean,
    split='test',
    max_seq_len=1500,
    aux_tokens=config.get('aux_tokens', 30),
    chunk_size=config.get('chunk_size', 1000),
    metadata_fields = {
        'obs': common_obs,
        #'obsm': [label_col],
    }
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config.get('num_workers', 4),
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config.get('num_workers', 4),
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config.get('num_workers', 4),
    pin_memory=True
)
# Load pre-trained model
model = Nicheformer.load_from_checkpoint(checkpoint_path=config['checkpoint_path'], strict=False,weights_only=False)

# Create fine-tuning model
fine_tune_model = NicheformerFineTune(
    backbone=model,
    supervised_task=config['supervised_task'],
    extract_layers=config['extract_layers'],
    function_layers=config['function_layers'],
    lr=config['head_lr'],  # Use head_lr for classification head
    warmup=config['warmup'],
    max_epochs=config['max_epochs'],
    dim_prediction=config['dim_prediction'],
    n_classes=config['n_classes'],
    # baseline=config['baseline'],
    freeze=config['freeze'],
    reinit_layers=config['reinit_layers'],
    extractor=config['extractor'],
    regress_distribution=config['regress_distribution'],
    pool=config['pool'],
    predict_density=config['predict_density'],
    ignore_zeros=config['ignore_zeros'],
    organ=config.get('organ', 'unknown'),
    label=config['label'],
    without_context=True
)

# Partial fine-tuning: Freeze specific layers if freeze=False
if not config['freeze'] and 'freeze_layers' in config:
    freeze_layers = config['freeze_layers']
    print(f"Partial fine-tuning: Freezing layers {freeze_layers}")
    
    # Freeze embeddings and positional encoding
    for param in fine_tune_model.backbone.embeddings.parameters():
        param.requires_grad = False
    if hasattr(fine_tune_model.backbone, 'positional_embedding'):
        for param in fine_tune_model.backbone.positional_embedding.parameters():
            param.requires_grad = False
    
    # Freeze specified encoder layers
    for layer_idx in freeze_layers:
        if 0 <= layer_idx < len(fine_tune_model.backbone.encoder.layers):
            for param in fine_tune_model.backbone.encoder.layers[layer_idx].parameters():
                param.requires_grad = False
            print(f"  Frozen layer {layer_idx}")
        else:
            print(f"  Warning: Layer {layer_idx} out of range (0-{len(fine_tune_model.backbone.encoder.layers)-1})")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in fine_tune_model.parameters())
    trainable_params = sum(p.numel() for p in fine_tune_model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Override configure_optimizers to use different learning rates
    original_configure_optimizers = fine_tune_model.configure_optimizers
    
    def configure_optimizers_with_different_lrs():
        """Configure optimizer with different learning rates for backbone and head."""
        # Separate parameters into backbone (unfrozen) and head
        backbone_params = []
        head_params = []
        
        # Get backbone parameters (only unfrozen ones)
        for name, param in fine_tune_model.backbone.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
        
        # Get head parameters
        for param in fine_tune_model.linear_head.parameters():
            if param.requires_grad:
                head_params.append(param)
        
        # Optional: Add extractor parameters if exists
        if hasattr(fine_tune_model, 'pooler_head'):
            for param in fine_tune_model.pooler_head.parameters():
                if param.requires_grad:
                    head_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = []
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': config.get('backbone_lr', config['head_lr'] * 0.1),
                'weight_decay': 0.001
            })
            print(f"Backbone parameters: {sum(p.numel() for p in backbone_params):,} with LR={config.get('backbone_lr', config['head_lr'] * 0.1)}")
        
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': config['head_lr'],
                'weight_decay': 0.001
            })
            print(f"Head parameters: {sum(p.numel() for p in head_params):,} with LR={config['head_lr']}")
        
        optimizer = optim.AdamW(param_groups)
        
        # Create learning rate scheduler
        lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=config['warmup'],
            max_epochs=config['max_epochs']
        )
        
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
    
    # Replace the configure_optimizers method
    fine_tune_model.configure_optimizers = configure_optimizers_with_different_lrs
else:
    if config['freeze']:
        print("Linear probing: All backbone parameters are frozen")
    else:
        print("Full fine-tuning: All parameters are trainable")

# Configure trainer
trainer = pl.Trainer(
    max_epochs=config['max_epochs'],
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    default_root_dir=config['output_dir'],
    precision=config.get('precision', 32),
    gradient_clip_val=config.get('gradient_clip_val', 1.0),
    accumulate_grad_batches=config.get('accumulate_grad_batches', 10),
)
# Train the model
print("Training the model...")
trainer.fit(
    model=fine_tune_model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader
)

# Test the model
print("Testing the model...")
test_results = trainer.test(
    model=fine_tune_model,
    dataloaders=test_loader
)
# ============================
# Extract partial fine-tuned latent & predictions
# ============================

preds = trainer.predict(
    fine_tune_model,
    dataloaders=test_loader
)

# clustering embedding (partial fine-tuned)
embeddings = torch.cat(
    [p["embedding"] for p in preds],
    dim=0
).cpu().numpy()

# prediction logits
logits = torch.cat(
    [p["prediction"] for p in preds],
    dim=0
).cpu().numpy()

pred_class = logits.argmax(axis=1)
test_mask = adata.obs.nicheformer_split == 'test'
assert embeddings.shape[0] == test_mask.sum()
prediction_key = f"predictions_{config.get('label', 'X_niche_1')}"
adata.obsm["X_nicheformer"] = np.zeros(
    (adata.n_obs, embeddings.shape[1]), dtype=np.float32
)
adata.obsm["X_nicheformer"][test_mask] = embeddings

adata.obs.loc[test_mask, f"{prediction_key}_class"] = pred_class
adata.obsm[f"{prediction_key}_logits"] = np.zeros(
    (adata.n_obs, logits.shape[1]), dtype=np.float32
)
adata.obsm[f"{prediction_key}_logits"][test_mask] = logits

# Save updated AnnData
adata.write_h5ad(config['output_path'])

print(f"Results saved to {config['output_path']}")