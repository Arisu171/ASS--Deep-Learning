"""Ablation study evaluating LSTM architecture variants on AG News classification.

CLI execution:
  python -m srcs.pipelines.day06_task02_lstm_variants --epochs 5 --batch_size 128

Inputs:
  data/ag_news (Hugging Face AG News official dataset partitions)

Outputs:
  docs/day06/task02/learning_curves.png
  docs/day06/task02/confusion_matrix_test.png
  bin/day06/task02/models/best_baseline.pt
  bin/day06/task02/models/best_embedding_128.pt
  bin/day06/task02/models/best_hidden_128.pt
  bin/day06/task02/models/best_bilstm.pt
  bin/day06/task02/models/best_dropout_03.pt
  bin/day06/task02/results/learning_curves.png
  bin/day06/task02/results/so_sanh_test.csv
  bin/day06/task02/results/summary.json
  bin/day06/task02/results/vocab.json
  bin/day06/task02/results/confusion_matrix_test_*.png
  bin/day06/task02/results/wrong_test_*.csv
"""

import os
import sys
import time
import re
import json
import csv
import random
import argparse
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_SEED = 42
DEFAULT_VAL_SIZE = 6000
DEFAULT_MAX_LEN = 100
DEFAULT_MAX_VOCAB = 20000
DEFAULT_MIN_FREQ = 2
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 5
DEFAULT_LR = 1e-3
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_TRAIN_SIZE = None
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ag_news")
DEFAULT_DOCS_DIR = os.path.join(PROJECT_ROOT, "docs", "day06", "task02")
DEFAULT_BIN_DIR = os.path.join(PROJECT_ROOT, "bin", "day06", "task02", "results")
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "bin", "day06", "task02", "models")
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
PAD_ID = 0
UNK_ID = 1

EXPERIMENT_CONFIGS = {
    "baseline": {
        "embedding_dim": 64,
        "hidden_dim": 96,
        "bidirectional": False,
        "dropout": 0.0
    },
    "embedding_128": {
        "embedding_dim": 128,
        "hidden_dim": 96,
        "bidirectional": False,
        "dropout": 0.0
    },
    "hidden_128": {
        "embedding_dim": 64,
        "hidden_dim": 128,
        "bidirectional": False,
        "dropout": 0.0
    },
    "bilstm": {
        "embedding_dim": 64,
        "hidden_dim": 96,
        "bidirectional": True,
        "dropout": 0.0
    },
    "dropout_03": {
        "embedding_dim": 64,
        "hidden_dim": 96,
        "bidirectional": False,
        "dropout": 0.3
    }
}


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Initialize pseudorandom number generators for reproducible experiment execution.

    Input:  seed -- integer random seed value (default DEFAULT_SEED).
    Output: None.
    Flow:   seed standard random library, numpy, torch cpu, and torch cuda backend if present.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> List[str]:
    """Tokenize English text into lowercase alphanumeric words.

    Input:  text -- input string containing news headlines and descriptions.
    Output: list of extracted lowercase word token strings.
    Flow:   apply lowercasing and extract alphanumeric tokens matching word boundaries.
    """
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def load_agnews_full(data_dir: str = DEFAULT_DATA_DIR,
                     val_size: int = DEFAULT_VAL_SIZE,
                     train_size: Optional[int] = DEFAULT_TRAIN_SIZE,
                     seed: int = DEFAULT_SEED) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load official AG News dataset and split training samples into train and validation.

    Input:  data_dir   -- filesystem path where dataset cache is preserved;
            val_size   -- integer number of samples dedicated to validation;
            train_size -- optional integer cap on training samples for resource efficiency;
            seed       -- random seed integer for stratified partitioning.
    Output: tuple of (train_rows, val_rows, test_rows).
    Flow:   load fancyzhx/ag_news from datasets library, verify sample counts,
            stratify split train set into train and validation partitions,
            optionally subsample train set if requested, and return partition row lists.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing datasets library. Please install via pip install datasets.") from exc

    data = load_dataset("fancyzhx/ag_news", cache_dir=data_dir)
    train_full = data["train"]
    test_raw = data["test"]

    if not 0 < val_size < len(train_full):
        raise ValueError(f"val_size must be between 1 and {len(train_full) - 1}")

    label_values = np.array(train_full["label"], dtype=int)
    train_idx, valid_idx = train_test_split(
        np.arange(len(train_full)),
        test_size=val_size,
        stratify=label_values,
        random_state=seed
    )

    if train_size is not None:
        if not 1 <= train_size <= len(train_idx):
            raise ValueError(f"train_size must be between 1 and {len(train_idx)}")
        if train_size < len(train_idx):
            train_idx, _ = train_test_split(
                train_idx,
                train_size=train_size,
                stratify=label_values[train_idx],
                random_state=seed
            )

    train_rows = [train_full[int(i)] for i in train_idx]
    valid_rows = [train_full[int(i)] for i in valid_idx]
    test_rows = [test_raw[i] for i in range(len(test_raw))]

    print(f"[AG News Variants] Train={len(train_rows):,} | Val={len(valid_rows):,} | Test={len(test_rows):,}")
    return train_rows, valid_rows, test_rows


def build_vocab(train_rows: List[Dict[str, Any]],
                max_vocab: int = DEFAULT_MAX_VOCAB,
                min_freq: int = DEFAULT_MIN_FREQ) -> Dict[str, int]:
    """Construct token vocabulary strictly from the training partition.

    Input:  train_rows -- list of sample dictionaries containing text field;
            max_vocab  -- upper bound on vocabulary size;
            min_freq   -- occurrence threshold below which tokens are omitted.
    Output: dictionary mapping token strings to unique integer identifiers.
    Flow:   accumulate token frequencies from training rows, assign pad and unk IDs,
            and populate dictionary with most frequent tokens satisfying min_freq.
    """
    counter = Counter(token for r in train_rows for token in tokenize(r["text"]))
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for token, frequency in counter.most_common():
        if len(vocab) >= max_vocab or frequency < min_freq:
            break
        vocab[token] = len(vocab)
    return vocab


class NewsDataset(Dataset):
    """Dataset wrapper for text tokens with sequence length and truncation metadata.

    Input:  rows    -- list of sample dictionaries with text and label;
            vocab   -- token to integer identifier dictionary;
            max_len -- maximum token limit before truncation.
    Output: Dataset instance yielding tokens, label, raw text, and metadata.
    Flow:   process each text sample into token IDs, compute original length,
            calculate unknown token ratio, record truncation status, and store records.
    """

    def __init__(self, rows: List[Dict[str, Any]], vocab: Dict[str, int], max_len: int = DEFAULT_MAX_LEN):
        self.examples = []
        for row in rows:
            tokens = tokenize(row["text"])
            ids = [vocab.get(t, UNK_ID) for t in tokens[:max_len]] or [UNK_ID]
            original_length = len(tokens)
            unknown_ratio = sum(i == UNK_ID for i in ids) / len(ids)
            was_truncated = original_length > max_len
            self.examples.append((
                ids,
                int(row["label"]),
                row["text"],
                original_length,
                unknown_ratio,
                was_truncated
            ))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[List[int], int, str, int, float, bool]:
        return self.examples[index]


def collate_news(batch: List[Tuple[List[int], int, str, int, float, bool]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[str, int, float, bool]]]:
    """Collate variable length news samples into dynamically padded batch tensors.

    Input:  batch -- list of item tuples from NewsDataset.
    Output: tuple of (token_ids [B, T], lengths [B], labels [B], metadata list).
    Flow:   extract sequence lengths, allocate padded tensor filled with PAD_ID,
            populate token sequences, construct label tensor, and pack sample metadata.
    """
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    batch_size = len(batch)
    max_len = int(lengths.max())
    ids = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
    for i, (sequence, _, _, _, _, _) in enumerate(batch):
        ids[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    metadata = [(x[2], x[3], x[4], x[5]) for x in batch]
    return ids, lengths, labels, metadata


class LSTMClassifier(nn.Module):
    """Configurable LSTM classifier supporting ablation of embedding, hidden dim, BiLSTM, and dropout.

    Input:  vocab_size    -- vocabulary size including special tokens;
            embedding_dim -- token representation dimensionality;
            hidden_dim    -- LSTM hidden state dimension;
            bidirectional -- boolean selecting bidirectional vs unidirectional recurrent processing;
            dropout       -- float dropout probability applied to embeddings and penultimate features.
    Output: initialized PyTorch nn.Module.
    Flow:   instantiate nn.Embedding, optional nn.Dropout, nn.LSTM with specified directionality,
            and nn.Linear classifier sized for 2*hidden_dim if bidirectional else hidden_dim.
    """

    def __init__(self, vocab_size: int,
                 embedding_dim: int = 64,
                 hidden_dim: int = 96,
                 bidirectional: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        classifier_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(classifier_input_dim, 4)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Execute forward computation through embedding, dropout, packed LSTM, and classifier.

        Input:  token_ids -- integer tensor of shape [batch_size, sequence_length];
                lengths   -- integer tensor containing valid sequence lengths before padding.
        Output: classification logits tensor of shape [batch_size, 4].
        Flow:   project tokens through embedding table, apply input dropout, pack sequence,
            propagate through LSTM, concatenate forward and backward hidden states if bidirectional,
            apply feature dropout, and compute output logits through linear projection.
        """
        embedded = self.embedding(token_ids)
        embedded = self.drop(embedded)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)

        if self.bidirectional:
            features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            features = hidden[-1]

        features = self.drop(features)
        return self.classifier(features)


def count_parameters(model: nn.Module) -> int:
    """Calculate total count of trainable parameters within a neural network module.

    Input:  model -- PyTorch module to inspect.
    Output: integer sum of requires_grad parameter elements.
    Flow:   traverse model.parameters() and sum numel() for all active parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: nn.Module,
                    device: torch.device,
                    max_grad_norm: float = DEFAULT_GRAD_CLIP) -> float:
    """Execute training for one epoch across all batches with gradient clipping.

    Input:  model         -- LSTMClassifier to train;
            loader        -- DataLoader providing training batches;
            optimizer     -- optimization algorithm updating weights;
            loss_fn       -- loss criterion module;
            device        -- computation target device;
            max_grad_norm -- maximum gradient clipping norm threshold (default 1.0).
    Output: average training loss per sample across the entire epoch.
    Flow:   set model to train mode, iterate batches, transfer inputs, reset optimizer gradients,
            compute forward logits, calculate cross entropy, backpropagate gradients,
            clip gradient norms to max_grad_norm, step optimizer, and accumulate total loss.
    """
    model.train()
    cumulative_loss = 0.0
    total_count = 0
    for ids, lengths, labels, _ in loader:
        ids_dev = ids.to(device)
        labels_dev = labels.to(device)
        batch_count = labels.size(0)

        optimizer.zero_grad()
        logits = model(ids_dev, lengths)
        loss = loss_fn(logits, labels_dev)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        cumulative_loss += float(loss.item()) * batch_count
        total_count += batch_count

    return cumulative_loss / total_count


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             loss_fn: nn.Module,
             device: torch.device,
             collect_errors: bool = False) -> Dict[str, Any]:
    """Evaluate model performance on an entire dataset partition.

    Input:  model          -- LSTMClassifier module to assess;
            loader         -- DataLoader providing evaluation minibatches;
            loss_fn        -- loss criterion module;
            device         -- computation device;
            collect_errors -- boolean flag controlling accumulation of detailed error records.
    Output: dictionary containing loss, accuracy, macro_f1, confusion_matrix, and error list.
    Flow:   set model to eval mode, iterate minibatches without gradient tracking,
            compute probabilities and class predictions, accumulate confusion matrix,
            record detailed prediction metadata for misclassified samples, and return metrics.
    """
    model.eval()
    cumulative_loss = 0.0
    total_count = 0
    true_labels = []
    predictions = []
    wrong_examples = []

    for ids, lengths, labels, meta in loader:
        ids_dev = ids.to(device)
        labels_dev = labels.to(device)
        batch_count = labels.size(0)

        logits = model(ids_dev, lengths)
        loss = loss_fn(logits, labels_dev)
        probs = torch.softmax(logits, dim=1).cpu()
        pred = torch.argmax(probs, dim=1).tolist()
        gold = labels.tolist()

        total_count += batch_count
        cumulative_loss += float(loss.item()) * batch_count
        true_labels.extend(gold)
        predictions.extend(pred)

        if collect_errors:
            for (text, original_length, unk_ratio, was_truncated), y, p, probability in zip(
                meta, gold, pred, probs
            ):
                if y != p:
                    wrong_examples.append({
                        "text": text,
                        "true_label": LABEL_NAMES[y],
                        "predicted_label": LABEL_NAMES[p],
                        "predicted_probability": round(float(probability[p]), 5),
                        "true_label_probability": round(float(probability[y]), 5),
                        "original_tokens": original_length,
                        "was_truncated": was_truncated,
                        "unk_ratio": round(float(unk_ratio), 4),
                    })

    cm = confusion_matrix(true_labels, predictions, labels=[0, 1, 2, 3])
    return {
        "loss": cumulative_loss / total_count,
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "macro_f1": float(f1_score(true_labels, predictions, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "cm": cm,
        "true": true_labels,
        "predicted": predictions,
        "errors": wrong_examples,
    }


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Save dictionary records into a CSV file with UTF-8 encoding.

    Input:  path -- destination file path;
            rows -- list of row dictionaries.
    Output: None.
    Flow:   ensure parent directory exists, extract header fields from first record, and write rows.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves(histories: Dict[str, List[Dict[str, Any]]], paths: List[Path]) -> None:
    """Render 3-panel comparative learning curves for all evaluated LSTM configurations.

    Input:  histories -- mapping from configuration name to epoch metric list;
            paths     -- list of file paths where figure will be exported.
    Output: None.
    Flow:   create 1x3 subplot figure, plot train loss, validation loss, and validation accuracy
            for each configuration, apply labels and grids, and save to requested destination paths.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for name, history in histories.items():
        epochs = [r["epoch"] for r in history]
        axes[0].plot(epochs, [r["train_loss"] for r in history], marker="o", label=name)
        axes[1].plot(epochs, [r["validation_loss"] for r in history], marker="o", label=name)
        axes[2].plot(epochs, [100.0 * r["validation_accuracy"] for r in history], marker="o", label=name)

    axes[0].set_title("Training Loss across Epochs", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Cross-Entropy Loss", fontsize=10)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Validation Loss across Epochs", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Cross-Entropy Loss", fontsize=10)
    axes[1].grid(alpha=0.25)

    axes[2].set_title("Validation Accuracy (%) across Epochs", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=10)
    axes[2].set_ylabel("Accuracy (%)", fontsize=10)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, paths: List[Path], name: str) -> None:
    """Render confusion matrix heatmap for a specific configuration.

    Input:  cm    -- 2D confusion matrix array;
            paths -- list of output file paths;
            name  -- configuration name string.
    Output: None.
    Flow:   create figure, plot heatmap using Blues colormap, overlay integer count labels,
            configure axis ticks and labels, and write figure to destination paths.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_yticklabels(LABEL_NAMES, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10, fontweight="bold")
    ax.set_title(f"Test Confusion Matrix: {name}", fontsize=12, fontweight="bold")

    threshold = cm.max() * 0.55
    for row in range(len(LABEL_NAMES)):
        for col in range(len(LABEL_NAMES)):
            ax.text(col, row, str(int(cm[row, col])), ha="center", va="center",
                    color="white" if cm[row, col] > threshold else "black", fontsize=10)

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=160)
    plt.close(fig)


def choose_diverse_errors(errors: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """Select misclassified samples prioritizing diverse pairs of true and predicted classes.

    Input:  errors -- list of misclassified sample dictionaries;
            limit  -- maximum number of error samples to return (default 5).
    Output: list of diverse error dictionaries.
    Flow:   traverse errors tracking unique (true_label, predicted_label) pairs,
            collect distinct pairs first, and supplement with remaining errors up to limit.
    """
    chosen = []
    pair_set = set()
    for row in errors:
        pair = (row["true_label"], row["predicted_label"])
        if pair not in pair_set:
            chosen.append(row)
            pair_set.add(pair)
        if len(chosen) == limit:
            return chosen
    for row in errors:
        if row not in chosen:
            chosen.append(row)
        if len(chosen) == limit:
            break
    return chosen


def run_full_experiments(args: argparse.Namespace) -> Dict[str, Any]:
    """Orchestrate training and evaluation across 5 LSTM architecture variants on AG News.

    Input:  args -- argparse namespace holding hyperparameters and storage paths.
    Output: dictionary detailing comparative performance across all evaluated models.
    Flow:   set deterministic seed, load AG News partitions, build vocabulary from training data,
            iterate all 5 configurations through training epochs with validation checkpointing,
            select best architecture based on minimum validation loss, evaluate all models on official test set,
            render learning curves and confusion matrices, export comparative tables, and return summary.
    """
    set_seed(args.seed)
    docs_dir = Path(args.docs_dir)
    bin_dir = Path(args.bin_dir)
    models_dir = Path(args.models_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_rows, valid_rows, test_rows = load_agnews_full(
        data_dir=args.data_dir,
        val_size=args.val_size,
        train_size=args.train_size,
        seed=args.seed
    )

    vocab = build_vocab(train_rows, max_vocab=args.max_vocab, min_freq=args.min_freq)
    (bin_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"Vocab size: {len(vocab):,} | Max sequence length: {args.max_len}")

    train_ds = NewsDataset(train_rows, vocab, args.max_len)
    valid_ds = NewsDataset(valid_rows, vocab, args.max_len)
    test_ds = NewsDataset(test_rows, vocab, args.max_len)

    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_news, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_news, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Batch size: {args.batch_size} | Epochs: {args.epochs}")

    criterion = nn.CrossEntropyLoss()
    histories = {}
    records = []

    for name, model_cfg in EXPERIMENT_CONFIGS.items():
        print("\n" + "=" * 70)
        print(f"EVALUATING CONFIGURATION: {name} | Parameters: {model_cfg}")
        set_seed(args.seed)
        model = LSTMClassifier(len(vocab), **model_cfg).to(device)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_news, generator=torch.Generator().manual_seed(args.seed),
            num_workers=0
        )

        param_count = count_parameters(model)
        print(f"Trainable parameters: {param_count:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        checkpoint_path = models_dir / f"best_{name}.pt"
        best_loss = float("inf")
        best_epoch = None
        history = []

        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
            val_metrics = evaluate(model, valid_loader, criterion, device)
            elapsed = time.perf_counter() - start

            rec = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_metrics["loss"],
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
                "seconds": elapsed
            }
            history.append(rec)

            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_path)

            print(f"{name:15s} | Ep {epoch:02d}/{args.epochs:02d} | "
                  f"Train: {train_loss:.4f} | Val: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.2%} | Val F1: {val_metrics['macro_f1']:.4f} | {elapsed:.2f}s")

        histories[name] = history
        save_csv(bin_dir / f"history_{name}.csv", history)

        mean_seconds = sum(h["seconds"] for h in history) / len(history)
        records.append({
            "config": name,
            **model_cfg,
            "trainable_parameters": param_count,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "mean_seconds_per_epoch": round(mean_seconds, 2)
        })

    learning_curves_paths = [
        docs_dir / "learning_curves.png",
        bin_dir / "learning_curves.png"
    ]
    plot_learning_curves(histories, learning_curves_paths)

    selected_config = min(records, key=lambda r: r["best_validation_loss"])["config"]
    print(f"\nModel selected by minimum validation loss: {selected_config}")

    best_errors = []
    print("\n================ OFFICIAL TEST EVALUATION ================")
    for record in records:
        name = record["config"]
        model = LSTMClassifier(len(vocab), **EXPERIMENT_CONFIGS[name]).to(device)
        model.load_state_dict(torch.load(models_dir / f"best_{name}.pt", map_location=device, weights_only=True))

        result = evaluate(model, test_loader, criterion, device, collect_errors=True)
        cm = result["cm"]

        record["test_loss"] = round(result["loss"], 4)
        record["test_accuracy"] = round(result["accuracy"], 4)
        record["test_macro_f1"] = round(result["macro_f1"], 4)
        record["test_errors"] = len(result["errors"])

        cm_paths = [
            bin_dir / f"confusion_matrix_test_{name}.png"
        ]
        if name == selected_config:
            cm_paths.append(docs_dir / "confusion_matrix_test.png")
            cm_paths.append(bin_dir / "confusion_matrix_test.png")
            best_errors = result["errors"]

        plot_confusion_matrix(cm, cm_paths, name)
        save_csv(bin_dir / f"wrong_test_{name}.csv", result["errors"])

        print(f"TEST | {name:15s} | Loss: {result['loss']:.4f} | "
              f"Accuracy: {result['accuracy']:.2%} | Macro-F1: {result['macro_f1']:.4f} | "
              f"Errors: {len(result['errors'])}/7,600")

    save_csv(bin_dir / "so_sanh_test.csv", records)

    summary = {
        "dataset": "fancyzhx/ag_news",
        "train_samples": len(train_ds),
        "validation_samples": len(valid_ds),
        "test_samples": len(test_ds),
        "vocab_size": len(vocab),
        "best_by_validation_loss": selected_config,
        "test_results": records,
        "device": str(device)
    }
    (bin_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nAll experiments completed successfully. Comparative table written to:\n  {bin_dir / 'so_sanh_test.csv'}")
    return summary


def parse_arguments() -> argparse.Namespace:
    """Parse command line parameters for the LSTM variants ablation pipeline.

    Input:  None.
    Output: populated argparse.Namespace instance.
    Flow:   initialize ArgumentParser, declare hyperparameter and path flags, and return parsed values.
    """
    parser = argparse.ArgumentParser(description="Ablation study across LSTM architecture variants on AG News.")
    parser.add_argument("--val_size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--train_size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--max_vocab", type=int, default=DEFAULT_MAX_VOCAB)
    parser.add_argument("--min_freq", type=int, default=DEFAULT_MIN_FREQ)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--docs_dir", type=str, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--bin_dir", type=str, default=DEFAULT_BIN_DIR)
    parser.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    run_full_experiments(parse_arguments())

