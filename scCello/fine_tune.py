import os
import sys
import argparse
import numpy as np
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    precision_score, recall_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import torch
from transformers import Trainer
import os.path as osp 
# fine_tune.py is in /maiziezhou_lab2/yuling/scCello/
# We need to add the parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXC_DIR = SCRIPT_DIR  # scCello directory
sys.path.append(EXC_DIR)

from sccello.src.utils import config as utils_config
from sccello.src.utils import helpers, logging_util, data_loading
from sccello.src.model_prototype_contrastive import PrototypeContrastiveForSequenceClassification
from sccello.src.collator.collator_for_classification import DataCollatorForCellClassification

'''
torchrun --master_port 29518 --nproc_per_node=1 \
  /maiziezhou_lab2/yuling/scCello/fine_tune.py \
  --pretrained_ckpt katarinayuan/scCello-zeroshot \
  --training_type partial_finetune \
  --unfreeze_layers 2 \
  --wandb_run_name partial_ft_top2 \
  --further_downsample 1.0 \
  --output_dir ./output/partial_ft_top2 \
  --change_learning_rate 5e-5 \
  --change_num_train_epochs 15

torchrun --master_port 29517 --nproc_per_node=1 ./sccello/script/run_cell_type_classification.py \
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
                            choices=["linear_probing", "from_scratch_linear", "partial_finetune"])
    parser.add_argument("--unfreeze_layers", type=int, default=0, 
                            help="Number of top BERT layers to unfreeze (0=linear probing, -1=full finetune)")
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
    parser.add_argument("--further_downsample", type=float, default=1.0)  
    
    ### change configurations in the training config yaml file ###
    parser.add_argument("--change_num_train_epochs", type=int, default=None)
    parser.add_argument("--change_learning_rate", type=float, default=None)
    parser.add_argument("--change_per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--change_lr_scheduler_type", type=str, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--use_class_weights", type=int, default=0, help="Use class weights for imbalanced data")

    parser.add_argument("--wandb_project", help="wandb project name", type=str, default="cell_type_classification")
    parser.add_argument("--wandb_run_name", help="wandb run name", type=str, 
                        default="test", required=True)
    args = parser.parse_args()

    # training_config
    file_name = "sccello/configs/cell_level/cell_type_classification_bert_training"
    if args.training_type == "linear_probing":
        file_name += "_probing"
    args.training_config = os.path.join(EXC_DIR, f"{file_name}.json")
    
    # Verify config file exists
    if not os.path.exists(args.training_config):
        raise FileNotFoundError(f"Training config not found: {args.training_config}")

    if args.pretrained_ckpt.endswith("/"):
        args.pretrained_ckpt = args.pretrained_ckpt[:-1]

    # model_source
    args.model_class = {
        "model_prototype_contrastive": "PrototypeContrastiveForSequenceClassification",
    }[args.model_source]

    print("args: ", args)

    return args


def build_supervised_trainer(args, training_args, model, train_dataset, eval_dataset):
    
    # Compute class weights if enabled
    class_weights = None
    if args.use_class_weights:
        train_labels = np.array([example['label'] for example in train_dataset])
        unique_classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
        class_weights = torch.FloatTensor(class_weights).to(model.device)
        logging.info(f"Using class weights: {class_weights}")
        # Monkey patch the model's forward to use class weights
        original_forward = model.forward
        def forward_with_weights(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            if 'loss' in outputs and class_weights is not None:
                # Recompute loss with class weights
                logits = outputs.logits
                labels = kwargs.get('labels', args[1] if len(args) > 1 else None)
                if labels is not None:
                    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
                    outputs.loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return outputs
        model.forward = forward_with_weights

    def compute_metrics(eval_preds):
        probs, labels = eval_preds
        preds = np.argmax(probs, axis=-1)
        acc = accuracy_score(labels, preds)
        
        # F1 scores
        macro_f1 = f1_score(labels, preds, average='macro')
        weighted_f1 = f1_score(labels, preds, average='weighted')
        
        # Precision and Recall
        macro_precision = precision_score(labels, preds, average='macro', zero_division=0)
        macro_recall = recall_score(labels, preds, average='macro', zero_division=0)
        
        # AUROC
        try:
            if probs.shape[1] > 2:
                auroc = roc_auc_score(labels, probs, multi_class="ovo", average='macro')
            else:
                auroc = roc_auc_score(labels, probs[:, 1], average='macro')
        except:
            auroc = 0
        
        # Per-class F1 (for detailed analysis)
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        min_f1 = np.min(per_class_f1)
        max_f1 = np.max(per_class_f1)
        
        metrics = {
            f'{args.data_source}/accuracy': acc,
            f'{args.data_source}/macro_f1': macro_f1,
            f'{args.data_source}/weighted_f1': weighted_f1,
            f'{args.data_source}/macro_precision': macro_precision,
            f'{args.data_source}/macro_recall': macro_recall,
            f'{args.data_source}/auroc': auroc,
            f'{args.data_source}/min_class_f1': min_f1,
            f'{args.data_source}/max_class_f1': max_f1,
        }
        
        return metrics

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
    elif args.training_type in ["partial_finetune"]:
        model = eval(args.model_class).from_pretrained(args.pretrained_ckpt, **model_kwargs).to("cuda")
        # Freeze all BERT parameters first
        for param in model.bert.parameters():
            param.requires_grad = False
        
        # Selectively unfreeze layers based on unfreeze_layers argument
        if args.unfreeze_layers == -1:
            # Full fine-tuning
            for param in model.bert.parameters():
                param.requires_grad = True
            logging.info("Unfreezing all BERT layers for full fine-tuning")
        elif args.unfreeze_layers > 0:
            # Partial fine-tuning: unfreeze top N layers
            total_layers = len(model.bert.encoder.layer)
            layers_to_unfreeze = min(args.unfreeze_layers, total_layers)
            for i in range(total_layers - layers_to_unfreeze, total_layers):
                for param in model.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
            logging.info(f"Unfreezing top {layers_to_unfreeze} BERT layers out of {total_layers}")
        
        # Always unfreeze pooler and embeddings layer norm for better adaptation
        if hasattr(model.bert, 'pooler') and model.bert.pooler is not None:
            for param in model.bert.pooler.parameters():
                param.requires_grad = True
        if hasattr(model.bert.embeddings, 'LayerNorm'):
            for param in model.bert.embeddings.LayerNorm.parameters():
                param.requires_grad = True
    else:
        raise NotImplementedError

    logging_util.basic_info_logging(model)

    return model

def solve_classification_supervised(args, all_datasets):

    trainset, evalset, testset, label_dict, eval1, test1 = all_datasets
    #trainset = trainset.train_test_split(train_size = 0.8, seed=args.seed)["train"]
    
    # further downsample 
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
    eval1.to_csv(osp.join(args.output_dir, 'eval_ids.csv'))
    test1.to_csv(osp.join(args.output_dir, 'test_ids.csv'))
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
    
    return {**eval_predictions.metrics, **test_predictions.metrics}


if __name__ == "__main__":

    args = parse_args()
    logging_util.set_environ(args)

    logging_util.init_wandb(args)
    
    #assert args.data_branch.startswith("frac")
    helpers.set_seed(args.seed)
    args.data_source = f"frac_indist"
    #all_datasets = data_loading.get_fracdata("/maiziezhou_lab2/yuling/scCello/data/example_data_saved_1", args.data_branch, args.indist, args.batch_effect)
    all_datasets = data_loading.get_fracdata("/maiziezhou_lab2/yuling/scCello/data/example_data_saved_test_2", args.data_branch, args.indist, args.batch_effect)
    
    solve_classification_supervised(args, all_datasets)
    
    