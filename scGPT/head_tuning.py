# %%
import copy
import gc
import json
from datetime import timedelta
'''
export NCCL_TIMEOUT=7200  
export NCCL_ASYNC_ERROR_HANDLING=1
'''
import os
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")   # 
os.environ.setdefault("NCCL_IB_DISABLE", "1")
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ.setdefault("NCCL_SOCKET_IFNAME", "enp6s0")
# python -m torch.distributed.run --standalone --nproc_per_node=2 /maiziezhou_lab2/yuling/scGPT/linear_probing.py
os.environ['NCCL_TIMEOUT'] = '7200'
from pathlib import Path
import sys
import time
import random
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import shutil
import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
import anndata
from scgpt.tokenizer.gene_tokenizer import GeneVocab
import pickle
#sys.path.append("../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
import pandas as pd
import seaborn as sns 
sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import torch.distributed as dist

def _finalize_ddp():
    import torch
    
    if not (dist.is_available() and dist.is_initialized()):
        return
    
    try:
        rank = dist.get_rank()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        try:
            world = dist.get_world_size()
            grp = dist.new_group(ranks=list(range(world)), backend="gloo")
            dist.barrier(group=grp)
            print(f"[Rank {rank}] Gloo barrier passed")
        except Exception as e:
            print(f"[Rank {rank}] Gloo barrier failed: {e}, trying default barrier")
            try:
                dist.barrier()
            except Exception as e2:
                print(f"[Rank {rank}] Default barrier also failed: {e2}")
        dist.destroy_process_group()
        print(f"[Rank {rank}] Process group destroyed")
        
    except Exception as e:
        print(f"[Rank {rank}] Error in _finalize_ddp: {e}")
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

torch.cuda.synchronize()   # record time 
start_time = time.perf_counter()
GPU_IDS = [1, 2]  

hyperparameter_defaults = dict(
    seed=0,
    dataset_name = "HumanLymph",
    do_train=True,
    load_model = "/maiziezhou_lab2/yuling/scGPT_model/scGPT_human",
    mask_ratio = 0.0,
    epochs= 10,
    n_bins = 51,
    MVC=False, # Masked value prediction for cell embedding
    ecs_thres=0.0, # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    lr=1e-4,
    batch_size=32,
    layer_size=128,
    nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead=4,  # number of heads in nn.MultiheadAttention
    dropout=0.2,  # dropout probability
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    fast_transformer = True,
    pre_norm=False,
    amp= False,  # Automatic Mixed Precision
    include_zero_gene = False,
    freeze = True, #freeze
    DSBN = False,  # Domain-spec batchnorm
)
'''
run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
'''

from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank < len(GPU_IDS):
        actual_gpu_id = GPU_IDS[local_rank]
        torch.cuda.set_device(actual_gpu_id)
    else:
        torch.cuda.set_device(local_rank)
    
    dist.init_process_group(backend="nccl",timeout=timedelta(seconds=7200))
    return rank, world_size, local_rank

use_ddp = torch.cuda.device_count() > 1 and int(os.environ.get("WORLD_SIZE", "1")) > 1
if use_ddp:
    rank, world_size, local_rank = setup_ddp()
else:
    rank, world_size, local_rank = 0, 1, 0

def get_device(local_rank):
    if use_ddp and local_rank < len(GPU_IDS):
        actual_gpu_id = GPU_IDS[local_rank]
        return torch.device(f"cuda:{actual_gpu_id}" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

is_main = (rank == 0)
device = get_device(local_rank)
print('is_main', is_main)
if not is_main:
    os.environ["WANDB_MODE"] = "disabled"  # silence side ranks
# --- after your wandb.init / is_main block ---
# If rank0: take values from wandb.config; else: use your defaults
def _safe_num_workers(batch_size: int) -> int:
        # try Linux affinity first
    try:
        n_cpus = len(os.sched_getaffinity(0))
    except Exception:
        # fallback for non-Linux / containers without affinity
        n_cpus = os.cpu_count() or 1
    # DataLoader rule of thumb
    nw = max(0, min(n_cpus, max(1, batch_size // 2)))
    return nw
# Convert dict -> object with attribute access
 
run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)  
 
#config = wandb.config if is_main else hyperparameter_defaults  # side ranks still need config
config = wandb.config
set_seed(1)
# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
print(config)
mask_ratio = 0.0

mask_value = "auto"  # for masked values, now it should always be auto

include_zero_gene = config.include_zero_gene  # if True, include zero genes among hvgs in the training
max_seq_len = 3001
n_bins = config.n_bins

# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"

# settings for training
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = config.MVC  # Masked value prediction for cell embedding
ECS = config.ecs_thres > 0  # Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight

explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

per_seq_batch_sample = False

# settings for optimizer
lr = config.lr  # TODO: test learning rate ratio between two tasks
lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
batch_size = config.batch_size
eval_batch_size = config.batch_size
epochs = config.epochs
schedule_interval = 1

# settings for the model
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"  # "linear" or "flash"
embsize = config.layer_size  # embedding dimension
d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
nhead = config.nhead  # number of heads in nn.MultiheadAttention
dropout = config.dropout  # dropout probability

# logging
log_interval = 100  # iterations
save_eval_interval = config.save_eval_interval  # epochs
do_eval_scib_metrics = True
assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if DAB > 1 else False
dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

# load and pre-process data 
slice_ids = ["2", "3", "4", "5", "6", "7", "9", "11", "17", "18", "19", "23", "24", "25", "26", "28", "33", "34", "36"]
def load_HMlymphNode(root_dir = '/maiziezhou_lab/Datasets/ST_datasets/humanMetastaticLymphNode/GSE251926_metastatic_lymph_node_3d.h5ad', section_id =  "1"):
    adataT = sc.read_h5ad(root_dir)
    section_id = int(section_id)  # Convert section_id to integer
    slice1 = adataT[adataT.obs['n_section'] == section_id]
    if 'gene_name' not in slice1.var.columns:
        slice1.var['gene_name'] = slice1.var_names
    slice1.obs['original_clusters'] = slice1.obs['annotation']
    return slice1
if dataset_name == "HumanLymph":
    slices=[]
    for index,i in enumerate(slice_ids[4:5]):
        ad=load_HMlymphNode(section_id=i)
        ad.obs['Z'] = index
        if 'gene_name' not in ad.var.columns:
            ad.var['gene_name'] = ad.var_names
        ad.obs['batch'] = index
        ad.layers["counts"] = ad.X
        slices.append(ad)
    adata = anndata.concat(slices, label = 'batch')
    ori_batch_col = "batch"
    if 'gene_name' not in adata.var.columns:
        adata.var['gene_name'] = adata.var_names
    adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
    adata.obs["celltype"] = adata.obs["original_clusters"].astype("category")
    adata.var.set_index(adata.var["gene_name"], inplace=True)
    #adata.var = adata.var.set_index("gene_symbols")
    data_is_raw = False
    #####------------
    slices_test =[]
    for index,i in enumerate(slice_ids[10:11]):
        ad = load_HMlymphNode(section_id=i)
        ad.obs['Z']=index
       
        if 'gene_name' not in ad.var.columns:
            ad.var['gene_name'] = ad.var_names
        ad.obs['batch']=index
        ad.layers["counts"]= ad.X
        slices_test.append(ad)
    #adata = scvi.data.pbmc_dataset()  # 11990 × 3346
    adata_test = anndata.concat(slices_test, label = 'batch')
    if 'gene_name' not in adata_test.var.columns:
        adata_test.var['gene_name'] = adata_test.var_names
    adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"
    adata_test.obs["celltype"] = adata_test.obs["original_clusters"].astype("category")
    adata_test.var.set_index(adata_test.var["gene_name"], inplace=True)
    adata_test_raw = adata_test.copy()
adata = adata.concatenate(adata_test, batch_key="str_batch")
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
celltypes = adata.obs["celltype"].unique()
num_types = len(np.unique(celltype_id_labels))
id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
adata.obs["celltype_id"] = celltype_id_labels
adata.var["gene_name"] = adata.var.index.tolist()
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    #genes_vocab = pd.Index(vocab.genes) 
    shutil.copy(vocab_file, save_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
    ##### set up the preprocessor
preprocessor = Preprocessor(
    use_key = "X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts= 0,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)

adata_test = adata[adata.obs["str_batch"] == "1"]
adata = adata[adata.obs["str_batch"] == "0"]

preprocessor(adata, batch_key=None)
preprocessor(adata_test, batch_key=None)
input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
}[input_style]
all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()
adata.obs['celltype'] = adata.obs['celltype'].astype('category')
#category_str2int(adata, from_col='celltype', to_col='celltype_id')  # 0,1,2,...
try:
    category_str2int(adata, "celltype", "celltype_id")
except TypeError:
    try:
        category_str2int(adata, "celltype")
        if "celltype_id" not in adata.obs.columns:
            adata.obs["celltype_id"] = adata.obs["celltype"].cat.codes.astype(int)
    except Exception:
        adata.obs["celltype_id"] = adata.obs["celltype"].cat.codes.astype(int)

celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
celltypes_labels = np.array(celltypes_labels)

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

(
    train_data,
    valid_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)
if config.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)
tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=include_zero_gene,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=include_zero_gene,
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)
def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}"
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if num_workers == 0:
        num_workers = _safe_num_workers(batch_size)
    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader
######
# train function 
######
def train(model: nn.Module, loader: DataLoader, epoch: int) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    (
        total_loss,
        total_mse,
        total_cls,
        total_cce,
        total_mvc,
        total_ecs,
        total_dab,
        total_adv_E,
        total_adv_D,
        total_zero_log_prob,
        total_mvc_zero_log_prob,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_error = 0.0
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
                #generative_training=False
                
            )
            masked_positions = input_values.eq(mask_value)  # the postions to predict
            loss = 0.0
            metrics_to_log = {}
            if MLM:
                loss_mse = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                loss = loss + loss_mse
                metrics_to_log = {"train/mse": loss_mse.item()}
            if explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            if CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})

                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)
            if CCE:
                loss_cce = 10 * output_dict["loss_cce"]
                loss = loss + loss_cce
                metrics_to_log.update({"train/cce": loss_cce.item()})
            if MVC:
                loss_mvc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_mvc
                metrics_to_log.update({"train/mvc": loss_mvc.item()})
            if MVC and explicit_zero_prob:
                loss_mvc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_mvc_zero_log_prob
                metrics_to_log.update({"train/mvc_nzlp": loss_mvc_zero_log_prob.item()})
            if ECS:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            if DAB:
                # try weighting and separate optimizer
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()
        
        # Clear cache periodically to help with memory management
        if batch % 10 == 0:
            torch.cuda.empty_cache()

        if ADV:
            # rerun the model for adversarial training
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
                #generative_training=False
            )

            # TRAINING DISCRIMINATOR
            loss_adv_D = criterion_adv(
                discriminator(output_dict["cell_emb"].detach()), batch_labels
            )
            if epoch > adv_D_delay_epochs:
                discriminator.zero_grad()
                loss_adv_D.backward()
                optimizer_D.step()

            # TRAINING ENCODER
            loss_adv_E = -criterion_adv(
                discriminator(output_dict["cell_emb"]), batch_labels
            )
            # NOTE: the loss is negative here because we want to maximize
            # the cross_entropy_loss, in other words, disguise against the discriminator
            if epoch > adv_E_delay_epochs:
                model.zero_grad()
                discriminator.zero_grad()
                loss_adv_E.backward()
                optimizer_E.step()
        if is_main:
            wandb.log(metrics_to_log)

        total_loss += loss.item()
        total_mse += loss_mse.item() if MLM else 0.0
        total_cls += loss_cls.item() if CLS else 0.0
        total_cce += loss_cce.item() if CCE else 0.0
        total_mvc += loss_mvc.item() if MVC else 0.0
        total_ecs += loss_ecs.item() if ECS else 0.0
        total_dab += loss_dab.item() if DAB else 0.0
        total_adv_E += loss_adv_E.item() if ADV else 0.0
        total_adv_D += loss_adv_D.item() if ADV else 0.0
        total_zero_log_prob += loss_zero_log_prob.item() if explicit_zero_prob else 0.0
        total_mvc_zero_log_prob += (
            loss_mvc_zero_log_prob.item() if MVC and explicit_zero_prob else 0.0
        )
        total_error += error_rate
        if is_main and batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_cls = total_cls / log_interval if CLS else 0.0
            cur_cce = total_cce / log_interval if CCE else 0.0
            cur_mvc = total_mvc / log_interval if MVC else 0.0
            cur_ecs = total_ecs / log_interval if ECS else 0.0
            cur_dab = total_dab / log_interval if DAB else 0.0
            cur_adv_E = total_adv_E / log_interval if ADV else 0.0
            cur_adv_D = total_adv_D / log_interval if ADV else 0.0
            cur_zero_log_prob = (
                total_zero_log_prob / log_interval if explicit_zero_prob else 0.0
            )
            cur_mvc_zero_log_prob = (
                total_mvc_zero_log_prob / log_interval
                if MVC and explicit_zero_prob
                else 0.0
            )
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | "
                + (f"mse {cur_mse:5.2f} | mre {cur_error:5.2f} |" if MLM else "")
                + (f"cls {cur_cls:5.2f} | " if CLS else "")
                + (f"err {cur_error:5.2f} | " if CLS else "")
                + (f"cce {cur_cce:5.2f} |" if CCE else "")
                + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
                + (f"ecs {cur_ecs:5.2f} |" if ECS else "")
                + (f"dab {cur_dab:5.2f} |" if DAB else "")
                + (f"adv_E {cur_adv_E:5.2f} |" if ADV else "")
                + (f"adv_D {cur_adv_D:5.2f} |" if ADV else "")
                + (f"nzlp {cur_zero_log_prob:5.2f} |" if explicit_zero_prob else "")
                + (
                    f"mvc_nzlp {cur_mvc_zero_log_prob:5.2f} |"
                    if MVC and explicit_zero_prob
                    else ""
                )
            )
            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_cce = 0
            total_mvc = 0
            total_ecs = 0
            total_dab = 0
            total_adv_E = 0
            total_adv_D = 0
            total_zero_log_prob = 0
            total_mvc_zero_log_prob = 0
            total_error = 0
            start_time = time.time()

#######
# load the pretrained model
################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=num_types if CLS else 1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=config.pre_norm,
)
if config.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
model.to(device)
#------------------------------------------------------------------------------------------
# Head-tuning mode: freeze backbone, train only classification head
HEAD_TUNING_KEYS = ("cls_decoder", "classifier", "classification_head", "head")
for name, para in model.named_parameters():
    if config.freeze:
        para.requires_grad = any(k in name for k in HEAD_TUNING_KEYS)
    else:
        para.requires_grad = True

post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
if is_main:
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.info(f"Trainable parameter tensors: {len(trainable_names)}")
    logger.info(f"Trainable parameter count: {post_freeze_param_count}")
    logger.info("Trainable module names:")
    for n in trainable_names:
        logger.info(f"  - {n}")

trainable_params = [p for p in model.parameters() if p.requires_grad]
if len(trainable_params) == 0:
    raise RuntimeError(
        "No trainable parameters found. Adjust HEAD_TUNING_KEYS or set freeze=False."
    )

if use_ddp:
    # Set find_unused_parameters=True to handle frozen parameters that don't receive gradients
    actual_gpu_id = GPU_IDS[local_rank] if local_rank < len(GPU_IDS) else local_rank
    model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id, find_unused_parameters=True)


if ADV:
    discriminator = AdversarialDiscriminator(
        d_model=embsize,
        n_cls=num_batch_types,
    ).to(device)
criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    trainable_params, lr=lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, schedule_interval, gamma=config.schedule_ratio
)
if DAB_separate_optim:
    optimizer_dab = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_dab = torch.optim.lr_scheduler.StepLR(
        optimizer_dab, schedule_interval, gamma=config.schedule_ratio
    )
if ADV:
    criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
    optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
    scheduler_E = torch.optim.lr_scheduler.StepLR(
        optimizer_E, schedule_interval, gamma=config.schedule_ratio
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
    scheduler_D = torch.optim.lr_scheduler.StepLR(
        optimizer_D, schedule_interval, gamma=config.schedule_ratio
    )

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    return_raw: bool = False,
    epoch: int = 0,
) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    predictions = []
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                    #generative_training = False,
                    return_attn=True,
                )
                output_values = output_dict["cls_output"]
                loss = criterion_cls(output_values, celltype_labels)
                
                if DAB:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)
    if is_main:
        wandb.log(
            {
                "valid/mse": total_loss / total_num,
                "valid/err": total_error / total_num,
                "valid/dab": total_dab / total_num,
                "valid/sum_mse_dab": (total_loss + dab_weight * total_dab) / total_num,
                "epoch": epoch,
            },
        )

    if return_raw:
        return np.concatenate(predictions, axis=0)

    return total_loss / total_num, total_error / total_num
#------------------------- new added 
def extract_embeddings(model: nn.Module,
                       adata_in: AnnData,
                       tag: str,
                       save_dir: Path,
                       batch_size: int = 256) -> np.ndarray:
    model.eval()
    X = adata_in.layers[input_layer_key].A if issparse(adata_in.layers[input_layer_key]) else adata_in.layers[input_layer_key]

    tokenized = tokenize_and_pad_batch(
        X, gene_ids, max_len=max_seq_len, vocab=vocab,
        pad_token=pad_token, pad_value=pad_value,
        append_cls=True, include_zero_gene=include_zero_gene,
    )
    def _get_obs_int(adata, key, default=0):
        if key in adata.obs:
            return adata.obs[key].to_numpy().astype(int)
        return np.zeros(adata.n_obs, dtype=int)

    dataset = SeqDataset({
        "gene_ids": tokenized["genes"],
        "values":   tokenized["values"],
        "target_values": tokenized["values"],
        "batch_labels": torch.from_numpy(_get_obs_int(adata_in, "batch_id")).long(),
        "celltype_labels": torch.from_numpy(_get_obs_int(adata_in, "celltype_id")).long(),
    })
    '''
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=min(len(os.sched_getaffinity(0)), batch_size//2), pin_memory=True)
    '''
    num_workers = _safe_num_workers(batch_size)
    pin_mem = torch.cuda.is_available()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    embs = []
    with torch.no_grad():
        for batch in loader:
            input_gene_ids = batch["gene_ids"].to(device)
            input_values   = batch["values"].to(device)
            batch_labels   = batch["batch_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            out = model(
                input_gene_ids, input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                CLS=True, CCE=False, MVC=False, ECS=False,  
                do_sample=False,
            )
            embs.append(out["cell_emb"].detach().cpu().numpy())

    embs = np.concatenate(embs, axis=0)
    obsm_key = f"X_scgpt_{tag}"
    adata_in.obsm[obsm_key] = embs
    np.save(save_dir / f"emb_{tag}.npy", embs)
    pd.DataFrame(embs, index=adata_in.obs_names).to_csv(save_dir / f"emb_{tag}.csv")
    def is_main_process():
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    def safe_write_h5ad(adata, path, retries=5, base_sleep=0.3):
        for t in range(retries):
            try:
                adata.write_h5ad(path, compression="gzip")
                return
            except BlockingIOError:
                time.sleep(base_sleep * (2 ** t) + random.random() * 0.2)
        adata.write_h5ad(path, compression="gzip")

    ad_copy = adata_in.copy()
    save_path = save_dir / f"adata_with_{tag}_emb.h5ad"
    if is_main_process():
        safe_write_h5ad(ad_copy, save_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    

    return embs

######-------- fine tuning
best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
define_wandb_metrcis()

try:
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        if is_main:
            print(f"DDP: world_size={world_size}")
            print(f"rank={rank}, local_rank={local_rank}, device={device}, cuda_current={torch.cuda.current_device()}")

        train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
        '''
      
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=batch_size,
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
        )
        valid_loader = prepare_dataloader(
            valid_data_pt,
            batch_size=eval_batch_size,
            shuffle=False,
            intra_domain_shuffle=False,
            drop_last=False,
        )
        '''
        train_dataset = SeqDataset(train_data_pt)
        valid_dataset = SeqDataset(valid_data_pt)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if use_ddp else None
        valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False,drop_last=True ) if use_ddp else None

        global_bs = config.batch_size
        bs_per_gpu = max(1, global_bs // world_size)
        num_workers = _safe_num_workers(bs_per_gpu)
        train_loader = DataLoader(train_dataset, batch_size=bs_per_gpu, sampler=train_sampler,
                                shuffle= False, pin_memory=True,
                                num_workers= num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=bs_per_gpu, sampler=valid_sampler,
                                shuffle = False, pin_memory=True,
                                num_workers= num_workers)
        if use_ddp:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch)
        
        if config.do_train:
            train(
                model,
                loader=train_loader,
                epoch=epoch,
            )
        if use_ddp:
            dist.barrier()
        
        val_loss, val_err = evaluate(
            model,
            loader=valid_loader,
            epoch=epoch,
        )
        elapsed = time.time() - epoch_start_time
        if is_main:
            logger.info("-" * 89)
            logger.info(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
            )
            logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            if is_main:
                logger.info(f"Best model with score {best_val_loss:5.4f}")

        scheduler.step()
        if DAB_separate_optim:
            scheduler_dab.step()
        if ADV:
            scheduler_D.step()
            scheduler_E.step()

except KeyboardInterrupt:
    print(f"[Rank {rank}] Training interrupted by user")
    if use_ddp:
        _finalize_ddp()
    sys.exit(0)
    
except torch.cuda.OutOfMemoryError as e:
    print(f"[Rank {rank}] CUDA Out of Memory error: {e}")
    print(f"[Rank {rank}] GPU {local_rank} memory stats:")
    if torch.cuda.is_available():
        print(f"  Allocated: {torch.cuda.memory_allocated(local_rank) / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(local_rank) / 1024**3:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated(local_rank) / 1024**3:.2f} GB")
    print(f"[Rank {rank}] Suggestion: Reduce batch_size (current: {config.batch_size}) or check for other processes using GPU {local_rank}")
    traceback.print_exc()
    if use_ddp:
        _finalize_ddp()
    sys.exit(1)
except Exception as e:
    print(f"[Rank {rank}] Training failed with error: {e}")
    traceback.print_exc()
    if use_ddp:
        _finalize_ddp()
    sys.exit(1)

def test(model: nn.Module, adata: DataLoader) -> float:
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene,
    )

    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
        "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }
    num_workers = _safe_num_workers(eval_batch_size)
    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers= num_workers,
        pin_memory=True,
    )

    model.eval()
    predictions = evaluate(
        model,
        loader=test_loader,
        return_raw=True,
        epoch=0,
    )

    # compute accuracy, precision, recall, f1
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(celltypes_labels, predictions)
    precision = precision_score(celltypes_labels, predictions, average="macro")
    recall = recall_score(celltypes_labels, predictions, average="macro")
    macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

    logger.info(
        f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
        f"Macro F1: {macro_f1:.3f}"
    )

    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
    }

    return predictions, celltypes_labels, results
##### -----------inference 
model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
predictions, labels, results = test(best_model, adata_test)
adata_test_raw.obs["predictions"] = [id2type[p] for p in predictions]
#--------- new added 
#bm = best_model.module if hasattr(best_model, "module") else best_model
bm = best_model.module if isinstance(best_model, torch.nn.parallel.DistributedDataParallel) else best_model
adata_train_view = adata[adata.obs["str_batch"] == "0"].copy() if "str_batch" in adata.obs else adata.copy()
#adata_test_view  = adata_test_raw.copy()   
adata_test_view  = adata_test.copy() 
train_emb = extract_embeddings(bm, adata_train_view, tag="train", save_dir=save_dir, batch_size=eval_batch_size)
test_emb  = extract_embeddings(bm, adata_test_view,  tag="test",  save_dir=save_dir, batch_size=eval_batch_size)

print("saved embeddings to:", save_dir / "emb_train.npy", save_dir / "emb_test.npy")

with plt.rc_context({"figure.figsize": (6, 4), "figure.dpi": (300)}):
    '''
    sc.pl.umap(
        adata_test_raw,
        color=["celltype", "predictions"],
        palette=palette_,
        show=False,
    )
    plt.savefig(save_dir / "results.png", dpi=300)
    '''
    ad = adata_test_raw.copy()

    # Build a basic embedding pipeline (on the same matrix your plots should reflect)
    # If you want to visualize raw counts, first normalize/log; otherwise choose the layer you trained on.
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, subset=True)
    sc.tl.pca(ad, n_comps=50, svd_solver="arpack")
    sc.pp.neighbors(ad, n_neighbors=15, n_pcs=50)
    sc.tl.umap(ad)                 # <- creates ad.obsm['X_umap']

    # carry over the columns you colored byX
    ad.obs["celltype"]    = adata_test_raw.obs["celltype"]
    ad.obs["predictions"] = adata_test_raw.obs["predictions"]

    # your palette may reference categories not present in test; fix safely
    # (optional) regenerate a palette from present categories:
    cats = list(pd.Categorical(ad.obs["celltype"]).categories)
    palette_auto = {c: plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 10]
                    for i, c in enumerate(cats)}
    adata_test_raw = ad
    sc.pl.umap(ad, color=["celltype", "predictions"],
               palette=palette_auto, show=False)
    plt.savefig(save_dir / "results.png", dpi=300)
df = adata_test_raw.obs.copy()
df.index.name = "cell_id"
df.to_csv(save_dir / "obs_lymph.csv")   
save_dict = {
    "predictions": predictions,
    "labels": labels,
    "results": results,
    "id_maps": id2type
}
with open(save_dir / "results.pkl", "wb") as f:
    pickle.dump(save_dict, f)
'''
results["test/cell_umap"] = wandb.Image(
    str(save_dir / "results.png"),
    caption=f"predictions macro f1 {results['test/macro_f1']:.3f}",
)
'''
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   
if use_ddp:
    torch.distributed.barrier()

from sklearn.metrics import confusion_matrix
celltypes = list(celltypes)
for i in set([id2type[p] for p in predictions]):
    if i not in celltypes:
        celltypes.remove(i)
cm = confusion_matrix(labels, predictions)
cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
cm = pd.DataFrame(cm, index=celltypes[:cm.shape[0]], columns=celltypes[:cm.shape[1]])
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
plt.savefig(save_dir / "confusion_matrix.png", dpi=300)

results["test/confusion_matrix"] = wandb.Image(
    str(save_dir / "confusion_matrix.png"),
    caption=f"confusion matrix",
)

if use_ddp:
    try:
        dist.barrier()  
        print(f"[Rank {rank}] Before finalize")
        _finalize_ddp()
        print(f"[Rank {rank}] After finalize")
    except Exception as e:
        print(f"[Rank {rank}] Error in finalize: {e}")
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
if is_main:
    try:
        wandb.finish()
        print("[Rank 0] wandb finished")
    except Exception as e:
        print(f"[Rank 0] wandb finish error: {e}")
torch.cuda.synchronize()
end_time = time.perf_counter()
#torch.save(to_save.state_dict(), save_dir / "model.pt")
runtime = end_time - start_time
print(f"Total runtime (entire method): {end_time - start_time:.2f} seconds")
ds = pd.DataFrame({'linear_probing_time':runtime})
ds.to_csv(save_dir / "running_time.csv")
#torch.save(best_model.state_dict(), save_dir / "model.pt")