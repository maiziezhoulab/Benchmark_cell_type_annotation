import os
import sys
import argparse
import numpy as np
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score
)
from pathlib import Path
import torch
from transformers import Trainer
import pandas as pd 
EXC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(EXC_DIR)
from sccello.src.utils import config as utils_config
from sccello.src.utils import helpers, logging_util, data_loading
from sccello.src.model_prototype_contrastive import PrototypeContrastiveForSequenceClassification
from sccello.src.collator.collator_for_classification import DataCollatorForCellClassification
'''
torchrun --master_port 29519 --nproc_per_node=1 ./sccello/script/run_cell_type_classification.py \
    --pretrained_ckpt katarinayuan/scCello-zeroshot \
    --training_type linear_probing \
    --wandb_run_name classification_test \
    --further_downsample 0.01 \
    --output_dir ./output/
'''
logging.basicConfig(level=logging.INFO)
#pretrained_ckpt=/maiziezhou_lab2/yuling/scCello/scCello-zeroshot
#output_dir=/maiziezhou_lab2/yuling/scCello/output
#wandb_run_name=test
import os, json
import numpy as np
import pandas as pd
from scipy.special import softmax
import os.path as osp 
def _save_predictions_as_csv(dataset, pred_out, split, label_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Map ids <-> labels (if provided)
    id2label = {v: k for k, v in label_dict.items()} if label_dict else {}
    logits = pred_out.predictions
    y_true = pred_out.label_ids

    # Support binary (logit) or multi-class (logits) outputs
    if logits.ndim == 1:
        # binary case: single logit per example
        probs = 1 / (1 + np.exp(-logits))
        y_pred = (probs >= 0.5).astype(int)
        max_prob = probs
    else:
        # multi-class case: logits shape (N, C)
        probs = softmax(logits, axis=1)
        y_pred = probs.argmax(axis=1)
        max_prob = probs.max(axis=1)

    # Base dataframe
    df = pd.DataFrame({
        "row_idx": np.arange(len(y_pred)),
        "y_true_id": y_true,
        "y_pred_id": y_pred,
        "y_true": [id2label.get(int(i), str(i)) for i in y_true],
        "y_pred": [id2label.get(int(i), str(i)) for i in y_pred],
        "max_prob": max_prob,
    })

    # If the dataset already has an identifier column, include it
    for col in ["id", "guid", "uid", "sample_id", "cell_id"]:
        if col in dataset.column_names:
            df[col] = dataset[col]

    # For multi-class, add per-class probabilities as columns
    if logits.ndim == 2:
        # Ensure deterministic column order by class id
        for class_id in range(probs.shape[1]):
            label_name = id2label.get(class_id, f"class_{class_id}")
            df[f"prob_{label_name}"] = probs[:, class_id]

    # Write CSV + metrics JSON
    csv_path = os.path.join(out_dir, f"{split}_predictions.csv")
    df.to_csv(csv_path, index=False)

    metrics_path = os.path.join(out_dir, f"{split}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(pred_out.metrics, f, indent=2)

    return csv_path, metrics_path

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", default="cell_type_classification", 
                        choices=["cell_type_classification"])
    parser.add_argument("--training_type", default="linear_probing", 
                            choices=["linear_probing", "from_scratch_linear"])
    parser.add_argument("--training_config", help="huggingface training configuration file")
    parser.add_argument("--pretrained_ckpt", type=str, help="pretrained model checkpoints", required=True)
    parser.add_argument("--output_dir", type=str, default="./single_cell_output", required=True)
    parser.add_argument("--seed", help="random seed", type=int, default=42)

    parser.add_argument("--data_branch", default="example_hf", choices=["frac100"])
    parser.add_argument("--data_source", type=str, help="specify which dataset is being used")
    parser.add_argument("--model_source", type=str, default="model_prototype_contrastive")

    parser.add_argument("--indist", type=int, default=1)
    parser.add_argument("--normalize", type=int, default=0)
    parser.add_argument("--pass_cell_cls", type=int, default=0)
    parser.add_argument("--batch_effect", type=int, default=None)
    parser.add_argument("--further_downsample", type=float, default=0.01)
    
    ### change configurations in the training config yaml file ###
    parser.add_argument("--change_num_train_epochs", type=int, default=None)
    parser.add_argument("--change_learning_rate", type=float, default=None)
    parser.add_argument("--change_per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--change_lr_scheduler_type", type=str, default=None)

    parser.add_argument("--wandb_project", help="wandb project name", type=str, default="cell_type_classification")
    parser.add_argument("--wandb_run_name", help="wandb run name", type=str, 
                        default="test", required=True)
    args = parser.parse_args()

    # training_config
    file_name = "./sccello/configs/cell_level/cell_type_classification_bert_training"
    if args.training_type == "linear_probing":
        file_name += "_probing"
    args.training_config = os.path.join(EXC_DIR, f"{file_name}.json")

    if args.pretrained_ckpt.endswith("/"):
        args.pretrained_ckpt = args.pretrained_ckpt[:-1]

    # model_source
    args.model_class = {
        "model_prototype_contrastive": "PrototypeContrastiveForSequenceClassification",
    }[args.model_source]

    print("args: ", args)

    return args


def build_supervised_trainer(args, training_args, model, train_dataset, eval_dataset):

    def compute_metrics(eval_preds):
        probs, labels = eval_preds
        preds = np.argmax(probs, axis=-1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average='macro')
        try:
            if probs.shape[1] > 2:
                auroc = roc_auc_score(labels, probs, multi_class="ovo", average='macro') # insensitive to class imbalance
            else:
                auroc = roc_auc_score(labels, probs[:, 1], average='macro') # insensitive to class imbalance
        except:
            auroc = 0

        return {
            f'{args.data_source}/accuracy': acc,
            f'{args.data_source}/macro_f1': macro_f1,
            f'{args.data_source}/auroc': auroc
        }

    def preprocess_logits_for_metrics(logits, labels):
        probs = torch.softmax(logits, dim=-1)
        return probs
########################################################
    max_len = int(getattr(model.config, "max_position_embeddings", 2048))

    collator = DataCollatorForCellClassification(
        model_input_name=["input_ids"],
        pad_to_multiple_of=None,
        model_max_length=max_len,   # 传给 collator
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator= collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    return trainer
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
###################------------------------------------------------------ new added
def extract_embeddings(model, dataset, batch_size=32):
    """Extract embeddings from the model"""
    model.eval()
    embeddings = []
    labels = []
    
    # Create dataloader
    from torch.utils.data import DataLoader
    max_len = int(getattr(model.config, "max_position_embeddings", 2048))
    collator = DataCollatorForCellClassification(
        model_input_name=["input_ids"],
        pad_to_multiple_of=None,
        model_max_length=max_len,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            
            # Get model outputs
            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Extract [CLS] token embedding (last hidden state)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
            
            embeddings.append(cls_embeddings.cpu().numpy())
            labels.append(batch['labels'].cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return embeddings, labels

def get_predictions(model, dataset, batch_size=32):
    """Get model predictions"""
    model.eval()
    predictions = []
    
    from torch.utils.data import DataLoader
    max_len = int(getattr(model.config, "max_position_embeddings", 2048))
    collator = DataCollatorForCellClassification(
        model_input_name=["input_ids"],
        pad_to_multiple_of=None,
        model_max_length=max_len,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            
            # Get logits
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            predictions.append(preds.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    return predictions

def plot_umap_comparison(embeddings, true_labels, pred_labels, label_dict, save_path, 
                         title="UMAP Visualization", split_name="combined"):
    """Plot side-by-side UMAP: predicted (left) vs ground truth (right)"""
    
    # Define color mapping
    color_map = {
        'CAF': '#1f77b4',
        'CAM': '#aec7e8',
        'Cortex_CCL21': '#ff7f0e',
        'Cytotoxic_IFN_signaling': '#ffbb78',
        'Fibroblasts': '#2ca02c',
        'Germinal_Center_Plasma_IgM_B_cell': '#98df8a',
        'High_Endothelial_Venules': '#d62728',
        'M1_macrophages': '#ff9896',
        'Macrophages': '#9467bd',
        'Plasma_IgA': '#c5b0d5',
        'Plasma_IgG': '#8c564b',
        'T_cell': '#c49c94',
        'Tumor': '#e377c2',
        'Tumor_Keratin_Pearl': '#f7b6d2',
        'unknown': '#7f7f7f'
    }
    
    # Create label name mapping
    id2label = {v: k for k, v in label_dict.items()}
    
    # Compute UMAP once for both plots
    print("Computing UMAP projection...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Save UMAP coordinates
    umap_coords_path = save_path.replace('.png', '_coordinates.npz')
    np.savez(umap_coords_path, 
             umap_coords=embedding_2d, 
             true_labels=true_labels, 
             pred_labels=pred_labels)
    print(f"Saved UMAP coordinates to {umap_coords_path}")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Get unique labels for legend
    unique_true_labels = np.unique(true_labels)
    unique_pred_labels = np.unique(pred_labels)
    all_unique_labels = np.unique(np.concatenate([unique_true_labels, unique_pred_labels]))
    
    # Plot 1: Predicted cell types (left)
    ax1 = axes[0]
    pred_label_names = [id2label.get(int(label), 'unknown') for label in pred_labels]
    pred_colors = [color_map.get(name, '#7f7f7f') for name in pred_label_names]
    
    for label_id in all_unique_labels:
        mask = pred_labels == label_id
        label_name = id2label.get(int(label_id), 'unknown')
        color = color_map.get(label_name, '#7f7f7f')
        ax1.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                   c=color, label=label_name, alpha=0.6, s=5, edgecolors='none')
    
    ax1.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax1.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax1.set_title('Predicted Cell Type', fontsize=16, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, markerscale=3)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ground truth cell types (right)
    ax2 = axes[1]
    true_label_names = [id2label.get(int(label), 'unknown') for label in true_labels]
    true_colors = [color_map.get(name, '#7f7f7f') for name in true_label_names]
    
    for label_id in all_unique_labels:
        mask = true_labels == label_id
        label_name = id2label.get(int(label_id), 'unknown')
        color = color_map.get(label_name, '#7f7f7f')
        ax2.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                   c=color, label=label_name, alpha=0.6, s=5, edgecolors='none')
    
    ax2.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax2.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax2.set_title('Ground Truth Cell Type', fontsize=16, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, markerscale=3)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"UMAP comparison plot saved to {save_path}")
    
    # Calculate and print accuracy
    accuracy = np.mean(true_labels == pred_labels)
    print(f"Overall accuracy for {split_name}: {accuracy:.4f}")

def load_model_supervised(args, training_args, training_cfg, label_dict):

    model_kwargs = {
        "num_labels": len(label_dict.keys()), # variable in huggingface
        "total_logging_steps": training_cfg["logging_steps"] * training_args.gradient_accumulation_steps,
        "data_source": args.data_source,
        "normalize_flag": args.normalize,
        "pass_cell_cls": args.pass_cell_cls,
    }

    if args.training_type in ["from_scratch_linear"]:
        cfg, model_kwargs = eval(args.model_class).config_class.from_pretrained(
                                args.pretrained_ckpt, return_unused_kwargs=True, **model_kwargs)
        model = eval(args.model_class)(cfg, **model_kwargs).to("cuda")
    elif args.training_type in ["linear_probing"]:
        model = eval(args.model_class).from_pretrained(args.pretrained_ckpt, **model_kwargs).to("cuda")
        for param in model.bert.parameters():
            param.requires_grad = False
    else:
        raise NotImplementedError

    logging_util.basic_info_logging(model)

    return model

def solve_classification_supervised(args, all_datasets):

    trainset, evalset, testset, label_dict, df1, df2 = all_datasets
    #trainset = trainset.train_test_split(train_size = 0.8, seed=args.seed)["train"]
     
    df1.to_csv("/maiziezhou_lab2/yuling/scCello/linear_eval_.csv", index=False)
     
    df2.to_csv("/maiziezhou_lab2/yuling/scCello/linear_test_.csv", index=False)
    #trainset = trainset.train_test_split(train_size=args.further_downsample, seed=args.seed)["train"]
    #evalset = evalset.train_test_split(train_size=args.further_downsample, seed=args.seed)["train"]

    # define output directory path
    args.output_dir = helpers.create_downstream_output_dir(args)
    
    training_args, training_cfg = utils_config.build_training_args(args, metric_for_best_model=f"{args.data_source}/macro_f1")
    assert training_cfg["do_eval"] is True and training_cfg["do_train"] is True

    model = load_model_supervised(args, training_args, training_cfg, label_dict)

    trainer = build_supervised_trainer(args, training_args, model, trainset, evalset)
    trainer.train()

    trainer.evaluate(evalset) # metric_key_prefix default as "eval"
    eval_predictions = trainer.predict(evalset)
    trainer.evaluate(testset, metric_key_prefix="test")
    test_predictions = trainer.predict(testset)
    eval_csv, eval_json = _save_predictions_as_csv(
    evalset, eval_predictions, "eval", label_dict, args.output_dir
    )
    test_csv, test_json = _save_predictions_as_csv(
        testset, test_predictions, "test", label_dict, args.output_dir
    )
    print(f"Saved eval predictions to {eval_csv} and metrics to {eval_json}")
    print(f"Saved test predictions to {test_csv} and metrics to {test_json}")

    print("eval_predictions.metrics: ", eval_predictions.metrics)
    print("test_predictions.metrics: ", test_predictions.metrics)
    # ============ NEW: Extract and visualize embeddings ============
    print("\n" + "="*50)
    print("Extracting embeddings and creating UMAP visualizations...")
    print("="*50)
    
    # Extract embeddings and predictions for eval set
    print("\nExtracting eval set embeddings and predictions...")
    eval_embeddings, eval_true_labels = extract_embeddings(model, evalset, batch_size=32)
    eval_pred_labels = get_predictions(model, evalset, batch_size=32)
    
    # Extract embeddings and predictions for test set
    print("\nExtracting test set embeddings and predictions...")
    test_embeddings, test_true_labels = extract_embeddings(model, testset, batch_size=32)
    test_pred_labels = get_predictions(model, testset, batch_size=32)
    
    # Combine eval and test sets
    print("\nCombining eval and test sets...")
    combined_embeddings = np.concatenate([eval_embeddings, test_embeddings], axis=0)
    combined_true_labels = np.concatenate([eval_true_labels, test_true_labels], axis=0)
    combined_pred_labels = np.concatenate([eval_pred_labels, test_pred_labels], axis=0)
    
    # Save combined embeddings
    combined_emb_path = os.path.join(args.output_dir, "combined_embeddings.npz")
    np.savez(combined_emb_path, 
             embeddings=combined_embeddings, 
             true_labels=combined_true_labels,
             pred_labels=combined_pred_labels,
             eval_size=len(eval_embeddings),
             test_size=len(test_embeddings))
    print(f"Saved combined embeddings to {combined_emb_path}")
    
    # Plot combined UMAP comparison
    combined_umap_path = os.path.join(args.output_dir, "combined_umap_comparison.png")
    plot_umap_comparison(combined_embeddings, combined_true_labels, combined_pred_labels, 
                        label_dict, combined_umap_path, 
                        title=f"UMAP Visualization - Combined Eval + Test ({args.data_source})",
                        split_name="combined")
    
    # Also save individual plots for reference
    eval_umap_path = os.path.join(args.output_dir, "eval_umap_comparison.png")
    plot_umap_comparison(eval_embeddings, eval_true_labels, eval_pred_labels,
                        label_dict, eval_umap_path,
                        title=f"UMAP Visualization - Eval Set ({args.data_source})",
                        split_name="eval")
    
    test_umap_path = os.path.join(args.output_dir, "test_umap_comparison.png")
    plot_umap_comparison(test_embeddings, test_true_labels, test_pred_labels,
                        label_dict, test_umap_path,
                        title=f"UMAP Visualization - Test Set ({args.data_source})",
                        split_name="test")
    
    print("\n" + "="*50)
    print("Embedding extraction and visualization completed!")
    print("="*50 + "\n")
    # ============ END NEW CODE ============
    return {**eval_predictions.metrics, **test_predictions.metrics}


if __name__ == "__main__":
    import time
    start = time.time()
    args = parse_args()
    logging_util.set_environ(args)

    logging_util.init_wandb(args)
    
    #assert args.data_branch.startswith("frac")
    helpers.set_seed(args.seed)
    args.data_source = f"frac_indist"
    all_datasets = data_loading.get_fracdata("/maiziezhou_lab2/yuling/scCello/data/example_data_saved_1", args.data_branch, args.indist, args.batch_effect)
    
    solve_classification_supervised(args, all_datasets)
    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds")
    # print("sync dir: ", wandb.__dict__['run'].__dict__['_settings']['sync_dir'])
    elapsed = end - start
    df = pd.DataFrame({
        "method": ["scCello_linear_probing"],
        "runtime_sec": [elapsed]
    })

    df.to_csv(osp.join(args.output_dir, "runtime.csv"), index=False)