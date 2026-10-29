"""BUỔI 5 – BÀI TRÊN LỚP: LSTM PHÂN LOẠI VĂN BẢN AG NEWS.


Tập dữ liệu: AG News , toàn bộ 120.000 mẫu train chính thức
  → 114.000 mẫu train + 6.000 mẫu validation; 7.600 mẫu test độc lập.

Đề bài và các phần cần làm:
  [YC1] Kiểm tra cách tải/chia train–validation–test, nêu vì sao test tách riêng.
  [YC2] Đọc/giải thích tokenize(), build_vocab(), AGNewsTextDataset, collate_batch().
  [YC3] In và giải thích vocab size, chiều dài chuỗi, token IDs, batch/embedding shape.
  [YC4] TODO 1: Viết LSTMClassifier.forward() với pack_padded_sequence.
  [YC5] TODO 2: Viết train_one_epoch(); chạy đủ epochs và báo cáo kết quả.
  [YC5] Tìm và giải thích ít nhất 3 tin tức dự đoán sai trên TEST chính thức.

HÀNH VI ĐÃ CUNG CẤP SẴN:
  - Các hàm tải dữ liệu, tiền xử lý, đánh giá và tạo file báo cáo.
  - Checkpoint chọn bằng validation loss (KHÔNG chọn dựa vào test).
  - Chỉ có LSTM một chiều, không dùng BiLSTM/GRU/Transformer.

Cài đặt: pip install torch datasets scikit-learn matplotlib numpy
Chạy:     python bai05_class_lstm_text.py

LƯU Ý: Phải hoàn thành TODO 1 và TODO 2 trước khi chạy huấn luyện.

"""
import argparse
import collections
import csv
import json
import random
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Có thể lưu hình cả khi chạy terminal không có màn hình.
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
PAD_ID, UNK_ID = 0, 1


def set_seed(seed: int = 42) -> None:
    """Giữ cùng phép chia dữ liệu và khởi tạo khi chạy lại thí nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# YÊU CẦU 1 – DATASET: AG News có 120.000 mẫu train, 7.600 mẫu test.
# Chỉ chia train/validation từ 120.000 mẫu train chính thức.
# Test chính thức KHÔNG tham gia tạo vocabulary hoặc chọn mô hình.
# ============================================================================
def load_agnews_splits(val_size: int, train_size: int | None, seed: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('Thiếu thư viện datasets. Cài: pip install datasets') from exc
    try:
        raw = load_dataset('fancyzhx/ag_news')
    except Exception as exc:
        raise RuntimeError(
            'Không tải được AG News THẬT. Kết nối Internet lần đầu hoặc sử dụng '
            'bộ nhớ đệm của Hugging Face; chương trình KHÔNG thay bằng dữ liệu giả.'
        ) from exc

    full_train = raw['train']
    official_test = raw['test']
    indices = np.arange(len(full_train))
    labels = np.asarray(full_train['label'])
    if not 0 < val_size < len(indices):
        raise ValueError(f'--val-size phải từ 1 đến {len(indices) - 1}')

    # Stratified split: giữ tỷ lệ 4 chủ đề trên tập train và validation.
    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, stratify=labels, random_state=seed
    )
    if train_size is not None:
        if not 1 <= train_size <= len(train_indices):
            raise ValueError(f'--train-size phải từ 1 đến {len(train_indices)}')
        # Nếu muốn chạy trên CPU, chỉ giảm train; validation/test vẫn là dữ liệu thật.
        if train_size < len(train_indices):
            train_indices, _ = train_test_split(
                train_indices, train_size=train_size,
                stratify=labels[train_indices], random_state=seed
            )

    def pick(dataset, idx):
        subset = dataset.select([int(i) for i in idx])
        return [{'text': r['text'], 'label': int(r['label'])} for r in subset]

    tr = pick(full_train, train_indices)
    va = pick(full_train, val_indices)
    te = [{'text': r['text'], 'label': int(r['label'])} for r in official_test]
    print(f'[AG News thật] Train={len(tr):,} | Validation={len(va):,} | Test={len(te):,}')
    return tr, va, te


# ============================================================================
# YÊU CẦU 2 – TOKENIZATION, VOCABULARY, TOKEN ID VÀ PADDING.
# Vocabulary CHỈ xây dựng từ train để không rò rỉ dữ liệu validation/test.
# [YC2] Sinh viên ĐỌC, CHẠY và GIẢI THÍCH các hàm đã cung cấp; không cần viết lại:
#   tokenize, build_vocab, AGNewsTextDataset, collate_batch.
# [YC3] Các tensor shape được in sẵn trong run_experiment; SV giải thích ý nghĩa.
# ============================================================================
def tokenize(text: str) -> list[str]:
    """Tách từ tiếng Anh đơn giản, chuyển chữ thường."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def build_vocab(train_rows, max_vocab: int = 12000, min_freq: int = 2):
    counts = collections.Counter(t for row in train_rows for t in tokenize(row['text']))
    vocab = {'<pad>': PAD_ID, '<unk>': UNK_ID}
    for word, frequency in counts.most_common():
        if len(vocab) >= max_vocab or frequency < min_freq:
            break
        vocab[word] = len(vocab)
    return vocab


class AGNewsTextDataset(Dataset):
    """Mỗi mẫu trả về (token_ids KHÔNG padding, nhãn, văn bản gốc)."""
    def __init__(self, rows, vocab, max_len: int):
        self.rows, self.vocab, self.max_len = rows, vocab, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tokens = tokenize(row['text'])[:self.max_len]  # Truncation: cắt câu quá dài.
        token_ids = [self.vocab.get(t, UNK_ID) for t in tokens]
        if not token_ids:  # LSTM cần độ dài >= 1.
            token_ids = [UNK_ID]
        return torch.tensor(token_ids, dtype=torch.long), int(row['label']), row['text']


def collate_batch(samples):
    """Padding động theo chuỗi dài nhất của batch; length là độ dài THẬT."""
    lengths = torch.tensor([len(token_ids) for token_ids, _, _ in samples], dtype=torch.long)
    batch_size, max_batch_len = len(samples), int(lengths.max())
    ids = torch.full((batch_size, max_batch_len), PAD_ID, dtype=torch.long)
    for i, (token_ids, _, _) in enumerate(samples):
        ids[i, :len(token_ids)] = token_ids
    labels = torch.tensor([label for _, label, _ in samples], dtype=torch.long)
    texts = [text for _, _, text in samples]
    return ids, lengths, labels, texts


# ============================================================================
# YÊU CẦU 4 – KIẾN TRÚC LSTM THUẦN TÚY: Embedding -> LSTM -> Linear.
# Không dùng GRU / BiLSTM / Transformer / attention.
# pack_padded_sequence bỏ qua các vị trí <pad> trong tính toán LSTM.
# ============================================================================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64,
                 hidden_dim: int = 96, n_classes: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden_dim, n_classes)  # Sinh ra 4 logits.

    def forward(self, token_ids, lengths):
        """TODO 1 [YC4]: Xây dựng Embedding -> LSTM -> Linear.

        Đầu vào:
          token_ids: Tensor [B, T] chứa token ID (có các vị trí <pad>).
          lengths: Tensor [B] chứa số token thực (không tính padding).
        Đầu ra: logits Tensor [B, 4], KHÔNG áp dụng softmax.

        Các bước sinh viên cần viết:
          1. Dùng self.embedding để đổi token ID thành [B, T, embedding_dim].
          2. Dùng pack_padded_sequence(..., lengths.cpu(),
             batch_first=True, enforce_sorted=False) để bỏ qua padding.
          3. Truyền chuỗi đã pack qua self.lstm, lấy hidden_n cuối.
          4. Lấy hidden state cuối của LSTM một chiều, một lớp; đưa vào
             self.classifier để tạo 4 logits.
        """
        # TODO 1: Viết mã ở đây.
        raise NotImplementedError('TODO 1: Hoàn thiện LSTMClassifier.forward()')


# ============================================================================
# YÊU CẦU 5 – TRAIN LOSS, VALIDATION ACCURACY VÀ TEST SAU KHI CHỌN MODEL.
# CrossEntropyLoss tự tính log-softmax từ logits, không softmax trước loss.
# ============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    """TODO 2 [YC5]: Huấn luyện LSTM trong một epoch.

    Các bước sinh viên phải lập trình:
      1. model.train(); duyệt toàn bộ mini-batch của train_loader.
      2. Chuyển token_ids và labels sang device (lengths do pack sử dụng CPU).
      3. optimizer.zero_grad(...); logits = model(token_ids, lengths).
      4. loss = criterion(logits, labels); loss.backward() (BPTT).
      5. clip_grad_norm_(model.parameters(), max_norm=1.0) chống exploding gradient.
      6. optimizer.step() để cập nhật trọng số.
      7. Tính TRUNG BÌNH loss theo tổng số MẪU (có nhân batch_size).

    Trả về: train_loss (float) của epoch.
    """
    # TODO 2: Viết mã ở đây.
    raise NotImplementedError('TODO 2: Hoàn thiện train_one_epoch()')


@torch.no_grad()
def evaluate(model, loader, criterion, device, collect_errors=False):
    model.eval()  # Không cập nhật trọng số ở validation/test.
    total_loss, total = 0.0, 0
    all_true, all_pred, errors = [], [], []
    for token_ids, lengths, labels, texts in loader:
        logits = model(token_ids.to(device), lengths)
        loss = criterion(logits, labels.to(device))
        predicted = logits.argmax(dim=1).cpu().tolist()
        true_labels = labels.tolist()
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        all_true.extend(true_labels)
        all_pred.extend(predicted)
        if collect_errors:
            for text, target, pred in zip(texts, true_labels, predicted):
                if target != pred:
                    errors.append({
                        'text': text,
                        'true_label': CLASS_NAMES[target],
                        'predicted_label': CLASS_NAMES[pred],
                    })
    return {
        'loss': total_loss / total,
        'accuracy': accuracy_score(all_true, all_pred),
        'macro_f1': f1_score(all_true, all_pred, labels=[0, 1, 2, 3],
                             average='macro', zero_division=0),
        'true': all_true, 'predicted': all_pred, 'errors': errors,
    }


# ============================================================================
# BÁO CÁO: lịch sử huấn luyện, biểu đồ, confusion matrix và câu dự đoán sai.
# ============================================================================
def write_csv(path: Path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def draw_learning_curve(history, path: Path):
    epochs = [row['epoch'] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row['train_loss'] for row in history], 'o-', label='Train loss')
    axes[0].plot(epochs, [row['validation_loss'] for row in history], 's-', label='Valid loss')
    axes[0].set(title='Cross-entropy loss', xlabel='Epoch', ylabel='Loss')
    axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(epochs, [100 * row['validation_accuracy'] for row in history], 'o-')
    axes[1].set(title='Validation accuracy', xlabel='Epoch', ylabel='Accuracy (%)')
    axes[1].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_confusion_matrix(true, predicted, path: Path):
    cm = confusion_matrix(true, predicted, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(4), CLASS_NAMES, rotation=30, ha='right')
    ax.set_yticks(range(4), CLASS_NAMES)
    ax.set(xlabel='Predicted label', ylabel='True label', title='AG News - Official Test')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() * .55 else 'black')
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return cm.tolist()


# ============================================================================
# YÊU CẦU 3 + QUY TRÌNH THỰC NGHIỆM ĐẦY ĐỦ:
#   train AG News -> validation chọn best epoch -> test AG News CHÍNH THỨC.
# ============================================================================
def run_experiment(cfg):
    set_seed(cfg.seed)
    destination = Path(cfg.out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    # (1) Chỉ dùng AG News thật, không tồn tại nhánh smoke/synthetic.
    train_rows, val_rows, test_rows = load_agnews_splits(
        val_size=cfg.val_size, train_size=None, seed=cfg.seed
    )

    # (2) Tạo từ vựng từ train; các token ngoài từ vựng nhận <unk>=1.
    vocab = build_vocab(train_rows, max_vocab=cfg.max_vocab, min_freq=cfg.min_freq)
    (destination / 'vocab.json').write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    train_data = AGNewsTextDataset(train_rows, vocab, cfg.max_len)
    val_data = AGNewsTextDataset(val_rows, vocab, cfg.max_len)
    test_data = AGNewsTextDataset(test_rows, vocab, cfg.max_len)

    train_loader = DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_batch,
        generator=torch.Generator().manual_seed(cfg.seed),
        num_workers=0,
    )
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size,
                            shuffle=False, collate_fn=collate_batch, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size,
                             shuffle=False, collate_fn=collate_batch, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(len(vocab), cfg.embedding_dim, cfg.hidden_dim).to(device)

    # (3) KIỂM TRA shape, in rõ sequence length thật và chiều đã được pad.
    sample_ids, sample_lengths, sample_labels, sample_texts = next(iter(train_loader))
    with torch.no_grad():
        emb = model.embedding(sample_ids.to(device))
        check_logits = model(sample_ids.to(device), sample_lengths)
    print('\n===== KIỂM TRA INPUT / EMBEDDING / OUTPUT =====')
    print(f'Device: {device} | Vocabulary size: {len(vocab):,} | Max sequence length: {cfg.max_len}')
    print(f'Tokenization mẫu: {tokenize(sample_texts[0])[:12]}')
    print(f'Token ID mẫu: {sample_ids[0, :min(12, int(sample_lengths[0]))].tolist()}')
    print(f'Batch IDs shape [B,T]: {tuple(sample_ids.shape)}')
    print(f'Lengths shape [B]: {tuple(sample_lengths.shape)} | Min/Max: {int(sample_lengths.min())}/{int(sample_lengths.max())}')
    print(f'Embedding shape [B,T,E]: {tuple(emb.shape)}')
    print(f'Logits shape [B,4]: {tuple(check_logits.shape)}')
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history, best_val_loss, best_epoch = [], float('inf'), None
    checkpoint = destination / 'best_model.pt'

    # (5) TRAIN: cập nhật trọng số. VALIDATION: KHÔNG cập nhật trọng số.
    print('\n===== HUẤN LUYỆN AG NEWS THẬT =====')
    for epoch in range(1, cfg.epochs + 1):
        start = time.perf_counter()
        loss_train = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid = evaluate(model, val_loader, criterion, device)
        record = {
            'epoch': epoch,
            'train_loss': loss_train,
            'validation_loss': valid['loss'],
            'validation_accuracy': valid['accuracy'],
            'validation_macro_f1': valid['macro_f1'],
            'seconds': time.perf_counter() - start,
        }
        history.append(record)
        if valid['loss'] < best_val_loss:
            best_val_loss, best_epoch = valid['loss'], epoch
            torch.save(model.state_dict(), checkpoint)
        print(f"Epoch {epoch:02d}/{cfg.epochs} | Train loss={loss_train:.4f} | "
              f"Valid loss={valid['loss']:.4f} | Valid Acc={valid['accuracy']:.2%} | "
              f"Valid Macro-F1={valid['macro_f1']:.4f} | {record['seconds']:.1f}s")

    # (5) TEST chỉ sau khi đã chọn checkpoint bằng VALIDATION loss.
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    selected_valid = evaluate(model, val_loader, criterion, device)
    print(f'\nCheckpoint được chọn: epoch {best_epoch} (validation loss nhỏ nhất).')
    print('===== ĐÁNH GIÁ CUỐI CÙNG – OFFICIAL TEST =====')
    test = evaluate(model, test_loader, criterion, device, collect_errors=True)
    print(f"Test loss={test['loss']:.4f} | Test Accuracy={test['accuracy']:.2%} | "
          f"Test Macro-F1={test['macro_f1']:.4f}")

    # [YC5] Phân tích ít nhất 3 câu dự đoán sai từ test.
    # TODO 3 (BÁO CÁO, không phải code): Chọn 3 câu từ danh sách được in/lưu,
    # giải thích khả năng nhầm do từ khóa đa nghĩa, chủ đề giao thoa,
    # thiếu ngữ cảnh hoặc giới hạn vocabulary/độ dài (nêu bằng chứng từ câu).
    print('\n===== 3 CÂU DỰ ĐOÁN SAI TRÊN TEST =====')
    for i, example in enumerate(test['errors'][:3], 1):
        print(f"{i}. Nhãn đúng: {example['true_label']} | Dự đoán: {example['predicted_label']}")
        print('   Văn bản:', example['text'][:400])
        print('   Gợi ý phân tích: có từ khóa đa nghĩa, tin đa chủ đề, hay thiếu ngữ cảnh?')
    if len(test['errors']) < 3:
        print(f'Chỉ tìm được {len(test["errors"])} câu sai; không tự tạo ví dụ sai.')

    write_csv(destination / 'history.csv', history,
              ['epoch', 'train_loss', 'validation_loss', 'validation_accuracy',
               'validation_macro_f1', 'seconds'])
    write_csv(destination / 'test_wrong_examples.csv', test['errors'],
              ['text', 'true_label', 'predicted_label'])
    draw_learning_curve(history, destination / 'learning_curve.png')
    cm = draw_confusion_matrix(test['true'], test['predicted'],
                               destination / 'confusion_matrix_test.png')
    summary = {
        'dataset': 'AG News (fancyzhx/ag_news) - REAL ONLY',
        'n_train': len(train_rows), 'n_validation': len(val_rows),
        'n_test': len(test_rows), 'seed': cfg.seed,
        'vocab_size': len(vocab), 'max_len': cfg.max_len,
        'batch_shape': list(sample_ids.shape), 'embedding_shape': list(emb.shape),
        'logits_shape': list(check_logits.shape),
        'best_epoch_selected_by_validation_loss': best_epoch,
        'best_validation_loss': selected_valid['loss'],
        'best_validation_accuracy': selected_valid['accuracy'],
        'test_loss': test['loss'], 'test_accuracy': test['accuracy'],
        'test_macro_f1': test['macro_f1'], 'test_confusion_matrix': cm,
        'wrong_test_count': len(test['errors']), 'device': str(device),
        'embedding_dim': cfg.embedding_dim, 'hidden_dim': cfg.hidden_dim,
        'epochs': cfg.epochs, 'lr': cfg.lr,
    }
    (destination / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\nĐã lưu kết quả thật vào: {destination.resolve()}')
    return summary


# ============================================================================
# CẤU HÌNH [YC1] – BẮT BUỘC dùng đủ 114.000 train + 6.000 valid + 7.600 test.
# Các siêu tham số đồng bộ với notebook đáp án AG News (LSTM thuần túy).
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='LSTM thuần túy - AG News thật')
    parser.add_argument('--val-size', type=int, default=6000)
    parser.add_argument('--max-len', type=int, default=80)
    parser.add_argument('--max-vocab', type=int, default=12000)
    parser.add_argument('--min-freq', type=int, default=2)
    parser.add_argument('--embedding-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default='outputs_bai05_agnews_real')
    return parser.parse_args()


if __name__ == '__main__':
    run_experiment(parse_args())
