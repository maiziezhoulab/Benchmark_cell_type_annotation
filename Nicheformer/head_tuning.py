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
from torch.utils.data import DataLoader
import anndata as ad
from typing import Optional, Dict, Any
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from nicheformer.models._nicheformer import Nicheformer
from nicheformer.models._nicheformer_fine_tune import NicheformerFineTune
from nicheformer.data.dataset import NicheformerDataset

config = {
    'data_path': '/maiziezhou_lab2/yuling/nicheformer/data/HumanLymph/xenium_human_lymph_ready_to_tokenize.h5ad',  # Path to your AnnData file
    'technology_mean_path': '/maiziezhou_lab2/yuling/nicheformer/data/model_means/xenium_mean_script.npy',  # Path to technology mean file
    'checkpoint_path': '/maiziezhou_lab2/yuling/nicheformer/data/nicheformer.ckpt',  # Path to pre-trained model
    'output_path': '/maiziezhou_lab2/yuling/nicheformer/data/HumanLymph/predictions.h5ad',  # Where to save results
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
    'freeze': True,  # Whether to freeze backbone
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
    lr=config['lr'],
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
fine_tune_model.eval()
preds = trainer.predict(
    fine_tune_model,
    dataloaders=test_loader
)

embeddings = torch.cat(
    [p["embedding"] for p in preds],
    dim=0
).cpu().numpy()

cls_predictions = torch.cat(
    [p["prediction"] for p in preds],
    dim=0
).cpu().numpy()
predictions = cls_predictions

test_mask = adata.obs.nicheformer_split == 'test'
assert embeddings.shape[0] == test_mask.sum()

 
prediction_key = f"predictions_{config.get('label', 'X_niche_1')}"
adata.obsm["X_nicheformer"] = np.zeros((adata.n_obs, embeddings.shape[1]))
adata.obsm["X_nicheformer"][test_mask] = embeddings
'''
if 'classification' in config['supervised_task']:
    # For classification tasks
    # predictions[0] is class indices (1D), predictions[1] is probabilities (2D)
    adata.obs.loc[test_mask, f"{prediction_key}_class"] = predictions[0]
    # Store probability distributions in obsm (multi-dimensional)
    adata.obsm[f"{prediction_key}_probs"] = np.zeros((len(adata), predictions[1].shape[1]))
    #adata.obsm[f"{prediction_key}_probs"][test_mask] = predictions[1]
    probs = predictions[1]
    if probs.ndim == 3 and probs.shape[-1] == 1:
        probs = probs.squeeze(-1)        # or probs = probs.squeeze(-1)
    adata.obsm[f"{prediction_key}_probs"][test_mask] = probs
else:
    # For regression tasks
    # Check if predictions is 1D or 2D
    if predictions.ndim == 1:
        # 1D predictions can go in obs
        adata.obs.loc[test_mask, prediction_key] = predictions
    else:
        # Multi-dimensional predictions go in obsm
        adata.obsm[prediction_key] = np.zeros((len(adata), predictions.shape[1]))
        adata.obsm[prediction_key][test_mask] = predictions

# Store test metrics
for metric_name, value in test_results[0].items():
    adata.uns[f"{prediction_key}_metrics_{metric_name}"] = value
'''
# class index
pred_class = cls_predictions.argmax(axis=1)

adata.obs.loc[test_mask, f"{prediction_key}_class"] = pred_class

# store raw logits / probabilities
adata.obsm[f"{prediction_key}_logits"] = np.zeros(
    (adata.n_obs, cls_predictions.shape[1])
)
adata.obsm[f"{prediction_key}_logits"][test_mask] = cls_predictions

# Save updated AnnData
adata.write_h5ad(config['output_path'])
print(f"Results saved to {config['output_path']}")
