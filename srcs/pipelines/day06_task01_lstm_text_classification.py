"""Text classification pipeline on AG News using pure single-layer LSTM.

CLI execution:
  python -m srcs.pipelines.day06_task01_lstm_text_classification --epochs 5 --batch_size 64

Inputs:
  data/ag_news (Hugging Face AG News official dataset partitions)

Outputs:
  docs/day06/task01/learning_curve.png
  docs/day06/task01/confusion_matrix_test.png
  bin/day06/task01/models/best_model.pt
  bin/day06/task01/results/history.csv
  bin/day06/task01/results/test_wrong_examples.csv
  bin/day06/task01/results/vocab.json
  bin/day06/task01/results/summary.json
  bin/day06/task01/results/learning_curve.png
  bin/day06/task01/results/confusion_matrix_test.png
"""

import os
import sys
import time
import re
import json
import csv
import random
import argparse
import collections
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
DEFAULT_MAX_LEN = 80
DEFAULT_MAX_VOCAB = 12000
DEFAULT_MIN_FREQ = 2
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_HIDDEN_DIM = 96
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 5
DEFAULT_LR = 1e-3
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_TRAIN_SIZE = None
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ag_news")
DEFAULT_DOCS_DIR = os.path.join(PROJECT_ROOT, "docs", "day06", "task01")
DEFAULT_BIN_DIR = os.path.join(PROJECT_ROOT, "bin", "day06", "task01", "results")
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "bin", "day06", "task01", "models")
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
PAD_ID = 0
UNK_ID = 1


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seed for deterministic execution across random, numpy, and torch.

    Input:  seed -- integer random seed value (default DEFAULT_SEED).
    Output: None.
    Flow:   seed standard random library, numpy, torch cpu, and torch cuda if available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_agnews_splits(data_dir: str = DEFAULT_DATA_DIR,
                       val_size: int = DEFAULT_VAL_SIZE,
                       train_size: Optional[int] = DEFAULT_TRAIN_SIZE,
                       seed: int = DEFAULT_SEED) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load AG News dataset partitions and split train into stratified train and validation subsets.

    Input:  data_dir   -- local directory path storing Hugging Face dataset cache;
            val_size   -- integer number of samples to partition for validation;
            train_size -- optional integer cap on training samples for resource-constrained execution;
            seed       -- integer seed controlling stratified partitioning.
    Output: tuple of (train_rows, val_rows, test_rows) containing text and label dictionaries.
    Flow:   import load_dataset from datasets library, fetch fancyzhx/ag_news from cache_dir,
            extract 120,000 train samples and 7,600 test samples, execute stratified train_test_split,
            optionally sub-sample training partition if train_size is specified, and return rows.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing datasets library. Please install via pip install datasets.") from exc

    raw = load_dataset("fancyzhx/ag_news", cache_dir=data_dir)
    full_train = raw["train"]
    official_test = raw["test"]

    indices = np.arange(len(full_train))
    labels = np.asarray(full_train["label"])

    if not 0 < val_size < len(indices):
        raise ValueError(f"val_size must be between 1 and {len(indices) - 1}")

    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, stratify=labels, random_state=seed
    )

    if train_size is not None:
        if not 1 <= train_size <= len(train_indices):
            raise ValueError(f"train_size must be between 1 and {len(train_indices)}")
        if train_size < len(train_indices):
            train_indices, _ = train_test_split(
                train_indices, train_size=train_size,
                stratify=labels[train_indices], random_state=seed
            )

    def extract_rows(dataset_split: Any, selected_indices: np.ndarray) -> List[Dict[str, Any]]:
        subset = dataset_split.select([int(i) for i in selected_indices])
        return [{"text": row["text"], "label": int(row["label"])} for row in subset]

    train_rows = extract_rows(full_train, train_indices)
    val_rows = extract_rows(full_train, val_indices)
    test_rows = [{"text": row["text"], "label": int(row["label"])} for row in official_test]

    print(f"[AG News] Train={len(train_rows):,} | Validation={len(val_rows):,} | Test={len(test_rows):,}")
    return train_rows, val_rows, test_rows


def tokenize(text: str) -> List[str]:
    """Tokenize English text into lowercase alphanumeric words.

    Input:  text -- raw input string containing news headlines or descriptions.
    Output: list of lowercase word token strings.
    Flow:   convert text to lowercase and apply regex pattern matching words and contractions.
    """
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def build_vocab(train_rows: List[Dict[str, Any]],
                max_vocab: int = DEFAULT_MAX_VOCAB,
                min_freq: int = DEFAULT_MIN_FREQ) -> Dict[str, int]:
    """Construct token vocabulary mapping exclusively from training rows.

    Input:  train_rows -- list of training row dictionaries containing text field;
            max_vocab  -- maximum allowed vocabulary size including special tokens;
            min_freq   -- minimum token occurrence count required for inclusion.
    Output: dictionary mapping token strings to unique integer identifiers.
    Flow:   tokenize all training rows, accumulate token frequencies with Counter,
            reserve index 0 for <pad> and index 1 for <unk>, and append frequent tokens.
    """
    counts = collections.Counter(token for row in train_rows for token in tokenize(row["text"]))
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for word, frequency in counts.most_common():
        if len(vocab) >= max_vocab or frequency < min_freq:
            break
        vocab[word] = len(vocab)
    return vocab


class AGNewsTextDataset(Dataset):
    """PyTorch Dataset wrapper for AG News text samples with vocabulary mapping.

    Input:  rows    -- list of sample dictionaries with text and integer label;
            vocab   -- token to integer identifier dictionary;
            max_len -- maximum token sequence length before truncation.
    Output: Dataset instance yielding unpadded token tensors, labels, and raw texts.
    Flow:   store inputs, implement __len__ over row count, and __getitem__ to tokenize,
            truncate to max_len, map tokens to vocab IDs with unk fallback, and return tensor.
    """

    def __init__(self, rows: List[Dict[str, Any]],
                 vocab: Dict[str, int],
                 max_len: int = DEFAULT_MAX_LEN):
        self.rows = rows
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        row = self.rows[index]
        tokens = tokenize(row["text"])[:self.max_len]
        token_ids = [self.vocab.get(token, UNK_ID) for token in tokens]
        if not token_ids:
            token_ids = [UNK_ID]
        return torch.tensor(token_ids, dtype=torch.long), int(row["label"]), row["text"]


def collate_batch(samples: List[Tuple[torch.Tensor, int, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Collate variable length text samples into dynamically padded batch tensors.

    Input:  samples -- list of tuples from AGNewsTextDataset (token_ids, label, text).
    Output: tuple of (ids [B, T], lengths [B], labels [B], texts list).
    Flow:   determine max length within current batch, allocate full tensor initialized with PAD_ID,
            copy token IDs into prefix positions, gather real sequence lengths, and return tensors.
    """
    lengths = torch.tensor([len(token_ids) for token_ids, _, _ in samples], dtype=torch.long)
    batch_size = len(samples)
    max_batch_len = int(lengths.max())
    ids = torch.full((batch_size, max_batch_len), PAD_ID, dtype=torch.long)
    for i, (token_ids, _, _) in enumerate(samples):
        ids[i, :len(token_ids)] = token_ids
    labels = torch.tensor([label for _, label, _ in samples], dtype=torch.long)
    texts = [text for _, _, text in samples]
    return ids, lengths, labels, texts


class LSTMClassifier(nn.Module):
    """Pure single-layer unidirectional LSTM classifier for news categorization.

    Input:  vocab_size    -- total vocabulary size including pad and unk tokens;
            embedding_dim -- dense representation dimensionality for each token (default 64);
            hidden_dim    -- LSTM hidden state vector dimension (default 96);
            n_classes     -- number of output classification categories (default 4).
    Output: initialized PyTorch nn.Module.
    Flow:   initialize nn.Embedding with padding_idx=0, nn.LSTM with batch_first=True,
            and nn.Linear projection head mapping hidden_dim to n_classes.
    """

    def __init__(self, vocab_size: int,
                 embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 n_classes: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Execute forward pass through Embedding, packed LSTM, and Linear classifier.

        Input:  token_ids -- integer tensor of shape [batch_size, max_sequence_length];
                lengths   -- 1D integer tensor containing true sequence lengths before padding.
        Output: raw unnormalized logits tensor of shape [batch_size, n_classes].
        Flow:   convert token indices to dense embeddings [B, T, E],
                pack padded sequences using pack_padded_sequence on CPU lengths,
                propagate through LSTM to obtain final hidden state [1, B, H],
                extract last layer hidden state hidden[-1], and project through classifier.
        """
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        features = hidden[-1]
        return self.classifier(features)


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    max_grad_norm: float = DEFAULT_GRAD_CLIP) -> float:
    """Execute training loop for one epoch with gradient clipping.

    Input:  model         -- LSTMClassifier module to train;
            loader        -- DataLoader yielding batches of training tokens, lengths, and labels;
            optimizer     -- optimization algorithm updating model parameters;
            criterion     -- loss function module;
            device        -- computation device (CPU or CUDA);
            max_grad_norm -- maximum gradient norm threshold for clipping (default 1.0).
    Output: float average training cross-entropy loss across all training samples.
    Flow:   set model to training mode, iterate minibatches, transfer tensors to device,
            zero previous gradients, forward pass through model, compute loss,
            backward propagate loss, clip parameter gradient norm to max_grad_norm,
            step optimizer, accumulate sample-weighted loss, and return mean.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    for token_ids, lengths, labels, _ in loader:
        token_ids_dev = token_ids.to(device)
        labels_dev = labels.to(device)
        batch_count = labels.size(0)

        optimizer.zero_grad()
        logits = model(token_ids_dev, lengths)
        loss = criterion(logits, labels_dev)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * batch_count
        total_samples += batch_count

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             collect_errors: bool = False) -> Dict[str, Any]:
    """Evaluate classifier performance across an entire dataset partition.

    Input:  model          -- LSTMClassifier module to evaluate;
            loader         -- DataLoader providing evaluation minibatches;
            criterion      -- loss function module;
            device         -- computation device (CPU or CUDA);
            collect_errors -- boolean indicating whether to accumulate misclassified samples.
    Output: dictionary containing loss, accuracy, macro_f1, true labels, predictions, and errors.
    Flow:   set model to eval mode, iterate batches without gradient computation,
            accumulate batch loss and prediction vectors, compute accuracy and macro F1,
            conditionally collect misclassified text records, and return metrics dictionary.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_true = []
    all_pred = []
    errors = []

    for token_ids, lengths, labels, texts in loader:
        token_ids_dev = token_ids.to(device)
        labels_dev = labels.to(device)
        batch_count = labels.size(0)

        logits = model(token_ids_dev, lengths)
        loss = criterion(logits, labels_dev)

        predicted = logits.argmax(dim=1).cpu().tolist()
        gold_labels = labels.tolist()

        total_loss += loss.item() * batch_count
        total_samples += batch_count
        all_true.extend(gold_labels)
        all_pred.extend(predicted)

        if collect_errors:
            for text, target, pred in zip(texts, gold_labels, predicted):
                if target != pred:
                    errors.append({
                        "text": text,
                        "true_label": CLASS_NAMES[target],
                        "predicted_label": CLASS_NAMES[pred],
                    })

    return {
        "loss": total_loss / total_samples,
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "macro_f1": float(f1_score(all_true, all_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "true": all_true,
        "predicted": all_pred,
        "errors": errors,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    """Save structured records to CSV file format with UTF-8 encoding.

    Input:  path    -- Path instance specifying destination file;
            rows    -- list of row dictionaries to write;
            columns -- list of column header field names.
    Output: None.
    Flow:   open target path with utf-8-sig encoding, instantiate DictWriter, write header and rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def draw_learning_curve(history: List[Dict[str, Any]], paths: List[Path]) -> None:
    """Render training and validation learning curves across epochs.

    Input:  history -- list of epoch metric dictionaries;
            paths   -- list of file paths where figure will be saved.
    Output: None.
    Flow:   create 2-subplot figure, plot cross-entropy loss on subplot 0,
            plot validation accuracy percentage on subplot 1, format layout,
            and save image to all requested destination paths.
    """
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["validation_loss"] for row in history]
    val_accs = [100.0 * row["validation_accuracy"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_losses, "o-", label="Train loss", color="#1f77b4")
    axes[0].plot(epochs, val_losses, "s-", label="Valid loss", color="#ff7f0e")
    axes[0].set_title("Cross-entropy Loss across Epochs", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_accs, "o-", color="#2ca02c")
    axes[1].set_title("Validation Accuracy (%) across Epochs", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy (%)", fontsize=10)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_confusion_matrix(y_true: List[int],
                          y_pred: List[int],
                          paths: List[Path],
                          class_names: List[str] = CLASS_NAMES) -> List[List[int]]:
    """Compute and render test partition confusion matrix heatmap.

    Input:  y_true      -- list of ground truth integer class labels;
            y_pred      -- list of predicted integer class labels;
            paths       -- list of destination file paths;
            class_names -- list of class label name strings.
    Output: 2D integer list representing confusion matrix cells.
    Flow:   calculate confusion matrix with sklearn, create matplotlib figure,
            render blue heatmap with colorbar, annotate numeric count in each cell,
            and export image to all designated paths.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10, fontweight="bold")
    ax.set_title("AG News Official Test: Confusion Matrix", fontsize=12, fontweight="bold")

    threshold = cm.max() * 0.55
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=10)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    plt.close(fig)
    return cm.tolist()


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute end-to-end LSTM text classification pipeline on AG News.

    Input:  args -- argparse namespace containing training hyperparameters and path configs.
    Output: dictionary summarizing dataset statistics, training history, and test performance.
    Flow:   set deterministic seed, load AG News partitions, construct vocabulary from train set,
            instantiate DataLoaders, verify tensor shapes, iterate training epochs with validation selection,
            evaluate best checkpoint on official test set, analyze top errors, and export all artifacts.
    """
    set_seed(args.seed)
    docs_dir = Path(args.docs_dir)
    bin_dir = Path(args.bin_dir)
    models_dir = Path(args.models_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows, test_rows = load_agnews_splits(
        data_dir=args.data_dir,
        val_size=args.val_size,
        train_size=args.train_size,
        seed=args.seed
    )

    vocab = build_vocab(train_rows, max_vocab=args.max_vocab, min_freq=args.min_freq)
    (bin_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_data = AGNewsTextDataset(train_rows, vocab, args.max_len)
    val_data = AGNewsTextDataset(val_rows, vocab, args.max_len)
    test_data = AGNewsTextDataset(test_rows, vocab, args.max_len)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_batch,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_batch, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_batch, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(len(vocab), args.embedding_dim, args.hidden_dim, n_classes=4).to(device)

    sample_ids, sample_lengths, sample_labels, sample_texts = next(iter(train_loader))
    with torch.no_grad():
        emb = model.embedding(sample_ids.to(device))
        check_logits = model(sample_ids.to(device), sample_lengths)

    print("\n================ TENSOR SHAPE VERIFICATION ================")
    print(f"Device: {device} | Vocab size: {len(vocab):,} | Max len: {args.max_len}")
    print(f"Sample tokens: {tokenize(sample_texts[0])[:10]}")
    print(f"Sample IDs: {sample_ids[0, :min(10, int(sample_lengths[0]))].tolist()}")
    print(f"Batch IDs shape [B, T]: {tuple(sample_ids.shape)}")
    print(f"Lengths shape [B]: {tuple(sample_lengths.shape)} (Min: {int(sample_lengths.min())}, Max: {int(sample_lengths.max())})")
    print(f"Embedding shape [B, T, E]: {tuple(emb.shape)}")
    print(f"Logits shape [B, 4]: {tuple(check_logits.shape)}")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_path = models_dir / "best_model.pt"

    history = []
    best_val_loss = float("inf")
    best_epoch = None

    print("\n================ COMMENCING TRAINING ================")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "seconds": elapsed
        }
        history.append(record)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train loss: {train_loss:.4f} | "
              f"Val loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2%} | "
              f"Val F1: {val_metrics['macro_f1']:.4f} | {elapsed:.2f}s")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    selected_valid = evaluate(model, val_loader, criterion, device)

    print("\n================ FINAL TEST EVALUATION ================")
    print(f"Selected checkpoint: Epoch {best_epoch} (Validation loss: {best_val_loss:.4f})")
    test_metrics = evaluate(model, test_loader, criterion, device, collect_errors=True)
    print(f"Official Test Loss: {test_metrics['loss']:.4f} | "
          f"Test Accuracy: {test_metrics['accuracy']:.2%} | "
          f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")

    print("\n================ TOP MISCLASSIFIED TEST EXAMPLES ================")
    for idx, err in enumerate(test_metrics["errors"][:3], start=1):
        print(f"\nExample {idx}:")
        print(f"  True Label:      {err['true_label']}")
        print(f"  Predicted Label: {err['predicted_label']}")
        print(f"  Text:            {err['text'][:300]}")

    write_csv(bin_dir / "history.csv", history,
              ["epoch", "train_loss", "validation_loss", "validation_accuracy", "validation_macro_f1", "seconds"])
    write_csv(bin_dir / "test_wrong_examples.csv", test_metrics["errors"],
              ["text", "true_label", "predicted_label"])

    learning_curve_paths = [
        docs_dir / "learning_curve.png",
        bin_dir / "learning_curve.png"
    ]
    draw_learning_curve(history, learning_curve_paths)

    confusion_matrix_paths = [
        docs_dir / "confusion_matrix_test.png",
        bin_dir / "confusion_matrix_test.png"
    ]
    cm = draw_confusion_matrix(test_metrics["true"], test_metrics["predicted"], confusion_matrix_paths)

    summary = {
        "dataset": "AG News (fancyzhx/ag_news)",
        "train_samples": len(train_rows),
        "validation_samples": len(val_rows),
        "test_samples": len(test_rows),
        "seed": args.seed,
        "vocab_size": len(vocab),
        "max_length": args.max_len,
        "trainable_parameters": param_count,
        "best_epoch": best_epoch,
        "best_validation_loss": selected_valid["loss"],
        "best_validation_accuracy": selected_valid["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "total_test_errors": len(test_metrics["errors"]),
        "confusion_matrix": cm,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": str(device)
    }

    (bin_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nPipeline successfully finished. Artifacts saved in:\n  {bin_dir}\n  {docs_dir}")
    return summary


def parse_arguments() -> argparse.Namespace:
    """Parse command line configuration flags for AG News text classification.

    Input:  None.
    Output: argparse.Namespace populated with parsed parameters.
    Flow:   instantiate ArgumentParser, register hyperparameter arguments, and return parsed args.
    """
    parser = argparse.ArgumentParser(description="AG News LSTM text classification pipeline.")
    parser.add_argument("--val_size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--train_size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--max_vocab", type=int, default=DEFAULT_MAX_VOCAB)
    parser.add_argument("--min_freq", type=int, default=DEFAULT_MIN_FREQ)
    parser.add_argument("--embedding_dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM)
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
    run_experiment(parse_arguments())

