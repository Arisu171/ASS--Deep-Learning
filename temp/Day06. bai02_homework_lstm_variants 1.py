"""BÀI VỀ NHÀ BUỔI 5 – SO SÁNH CÁC BIẾN THỂ LSTM TRÊN AG NEWS.

FILE GIAO SINH VIÊN – CHƯA PHẢI LỜI GIẢI. Các vị trí cần hoàn thiện
được đánh dấu TODO. Không có chế độ smoke test/dữ liệu giả.

YÊU CẦU:
(1) Cơ bản: hoàn thiện huấn luyện LSTM baseline, báo cáo validation
    accuracy, Macro-F1, train loss, validation loss theo từng epoch.
(2) Khá: thay đổi embedding_dim HOẶC hidden_dim; so sánh công bằng.
(3) Nâng cao: thử BiLSTM HOẶC Dropout; so sánh accuracy, số tham số,
    thời gian/epoch.
(4) Đánh giá: mô hình được chọn bằng VALIDATION, báo cáo trên TEST
    chính thức AG News, vẽ learning curve, confusion matrix;
    đọc và giải thích ít nhất 5 trường hợp dự đoán sai.

DỮ LIỆU: 120.000 train chính thức được chia train/validation 114.000/6.000;
         7.600 mẫu test chính thức dùng chỉ ở giai đoạn đánh giá cuối.
CHẠY: pip install torch datasets scikit-learn numpy matplotlib
       python bai05_homework_lstm_variants.py

Đọc file HUONG_DAN_GIAO_SINH_VIEN.md trước khi thực hiện.
"""

# ======================================================================
# PHẦN 0. CÀI ĐẶT – GIỮ CÙNG ĐIỀU KIỆN CHO TẤT CẢ CẤU HÌNH
# ======================================================================
import csv
import json
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
PAD_ID, UNK_ID = 0, 1


@dataclass
class Settings:
    seed: int = 42
    val_size: int = 6000                # Chỉ tách validation từ train chính thức
    max_len: int = 100                 # Tối đa 100 token / bản tin
    max_vocab: int = 20000            # Kích thước tối đa, gồm <pad>, <unk>
    min_freq: int = 2
    batch_size: int = 128
    epochs: int = 5                    # Cùng số epoch cho mọi cấu hình
    lr: float = 1e-3                   # Cùng optimizer/lr cho mọi cấu hình
    output_dir: str = "outputs_bai05_homework_AGNews_full"


CFG = Settings()

# TODO 2 – MỨC KHÁ/NÂNG CAO: Bổ sung cấu hình so sánh với baseline.
# Mỗi thí nghiệm chỉ thay MỘT yếu tố so với baseline, giữ nguyên:
# dataset, train/validation, seed, batch_size, optimizer, learning rate, epochs.
# - Mức khá: thêm ít nhất MỘT cấu hình với embedding_dim HOẶC hidden_dim khác.
# - Mức nâng cao: thêm ít nhất MỘT cấu hình BiLSTM HOẶC Dropout (sau khi
#   hoàn thiện TODO 3 trong lớp LSTMClassifier bên dưới).
# Gợi ý tên: "embedding_lon_hon", "hidden_lon_hon", "bilstm", "dropout".
# Không thay baseline vì đây là mốc đối chứng.
EXPERIMENTS = {
    "baseline": dict(embedding_dim=64, hidden_dim=96,
                     bidirectional=False, dropout=0.0),
    # TODO 2A (KHÁ): thêm thí nghiệm thay embedding_dim hoặc hidden_dim.
    # "ten_cau_hinh": dict(embedding_dim=..., hidden_dim=...,
    #                       bidirectional=False, dropout=0.0),
    # TODO 2B (NÂNG CAO): thêm thí nghiệm BiLSTM hoặc Dropout.
}


def set_seed(seed: int):
    """Giữ seed cố định; mỗi thí nghiệm khởi tạo lại seed độc lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# PHẦN 1. DỮ LIỆU – AG NEWS THẬT, ĐẦY ĐỦ TRAIN/VALIDATION/TEST
# ======================================================================
def tokenize(text: str):
    """Tokenization mức từ (tiếng Anh); chuyển thành chữ thường."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def load_agnews_full(settings: Settings):
    """Tải 120.000 train + 7.600 test; chia 114.000 train / 6.000 validation.

    Tập test chính thức KHÔNG được dùng để chọn mô hình hay tạo vocabulary.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Thiếu thư viện datasets; hãy chạy: pip install datasets") from exc

    try:
        data = load_dataset("fancyzhx/ag_news")
    except Exception as exc:
        raise RuntimeError(
            "Không tải được AG News thật. Cần kết nối Internet lần đầu "
            "hoặc cache dataset Hugging Face đã tải trước đó."
        ) from exc

    train_full = data["train"]
    test_raw = data["test"]
    print(f"AG News chính thức: train={len(train_full):,}, test={len(test_raw):,}")
    if len(train_full) != 120000 or len(test_raw) != 7600:
        raise ValueError("Số lượng AG News khác 120.000/7.600; hãy kiểm tra nguồn dữ liệu.")
    if not 0 < settings.val_size < len(train_full):
        raise ValueError("val_size phải lớn hơn 0 và nhỏ hơn 120000")

    label_values = np.array(train_full["label"], dtype=int)
    train_idx, valid_idx = train_test_split(
        np.arange(len(train_full)), test_size=settings.val_size,
        stratify=label_values, random_state=settings.seed
    )
    train_rows = [train_full[int(i)] for i in train_idx]
    valid_rows = [train_full[int(i)] for i in valid_idx]
    test_rows = [test_raw[i] for i in range(len(test_raw))]
    assert len(train_rows) + len(valid_rows) == 120000
    print(f"Dùng TOÀN BỘ train chính thức: train={len(train_rows):,}, "
          f"validation={len(valid_rows):,}; test độc lập={len(test_rows):,}")
    return train_rows, valid_rows, test_rows


def build_vocab(train_rows, max_vocab=20000, min_freq=2):
    """Chỉ thống kê token của TRAIN (không dùng validation/test)."""
    counter = Counter(token for r in train_rows for token in tokenize(r["text"]))
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for token, frequency in counter.most_common():
        if len(vocab) >= max_vocab or frequency < min_freq:
            break
        vocab[token] = len(vocab)
    return vocab


class NewsDataset(Dataset):
    """Token IDs / length thực; chỉ padding khi ghép mini-batch."""
    def __init__(self, rows, vocab, max_len):
        self.examples = []
        for row in rows:
            tokens = tokenize(row["text"])
            ids = [vocab.get(t, UNK_ID) for t in tokens[:max_len]] or [UNK_ID]
            original_length = len(tokens)
            unknown_ratio = sum(i == UNK_ID for i in ids) / len(ids)
            self.examples.append((ids, int(row["label"]), row["text"],
                                  original_length, unknown_ratio, original_length > max_len))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_news(batch):
    """Padding đến chuỗi dài nhất trong batch; giữ lengths TRƯỚC padding."""
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    ids = torch.full((len(batch), int(lengths.max())), PAD_ID, dtype=torch.long)
    for i, (sequence, _, _, _, _, _) in enumerate(batch):
        ids[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    metadata = [(x[2], x[3], x[4], x[5]) for x in batch]
    return ids, lengths, labels, metadata


# ======================================================================
# PHẦN 2. KIẾN TRÚC – BASELINE / EMBEDDING / HIDDEN / BiLSTM / DROPOUT
# ======================================================================
class LSTMClassifier(nn.Module):
    """LSTM 1 chiều baseline đã được cung cấp để sinh viên tập trung thực nghiệm.

    TODO 3 (NÂNG CAO): tự mở rộng phần này cho BiLSTM HOẶC Dropout.
    Chú ý BiLSTM có hai hidden states cần ghép và số chiều vào Linear x2.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 bidirectional=False, dropout=0.0):
        super().__init__()
        # TODO 3A: Nếu chọn BiLSTM, thay chỗ chặn dưới đây bằng kiến trúc:
        #   nn.LSTM(..., bidirectional=True), Linear(2*hidden_dim, 4),
        #   và ghép đúng hai hướng trong forward().
        if bidirectional:
            raise NotImplementedError(
                "TODO 3A: Tự cài BiLSTM và điều chỉnh Linear + forward()."
            )
        # TODO 3B: Nếu chọn Dropout, đặt nn.Dropout(p) vào phần embedding
        # và/hoặc trước classifier trong forward(); chỉ tác dụng khi model.train().
        if dropout > 0:
            raise NotImplementedError(
                "TODO 3B: Tự cài Dropout cho LSTM 1 tầng."
            )
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Linear(hidden_dim, 4)

    def forward(self, token_ids, lengths):
        # Đáp án baseline: Embedding → Pack (bỏ padding) → LSTM → 4 logits.
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        features = hidden[-1]  # hidden cuối theo độ dài thật của mỗi tin.
        # TODO 3A: nếu thực hiện BiLSTM, phải ghép hidden chiều xuôi/ngược.
        # TODO 3B: nếu thực hiện Dropout, áp dụng dropout vào đúng vị trí.
        return self.classifier(features)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ======================================================================
# PHẦN 3. HUẤN LUYỆN BASELINE – TRAIN LOSS & VALIDATION MỖI EPOCH
# ======================================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """TODO 1 (CƠ BẢN): hoàn thiện training loop trong thân vòng for.

    Các thao tác bắt buộc: model.train() → zero_grad → forward
    → CrossEntropyLoss → backward → clip gradient → optimizer.step().
    Hàm phải trả về TRAIN LOSS trung bình theo số mẫu, không chỉ batch cuối.
    """
    model.train()
    cumulative_loss, n = 0.0, 0
    for ids, lengths, labels, _ in loader:
        ids, labels = ids.to(device), labels.to(device)
        # TODO 1.1: Đặt gradient của optimizer về 0.
        # TODO 1.2: Gọi model(ids, lengths) để tạo logits.
        # TODO 1.3: Tính CrossEntropyLoss giữa logits và labels.
        # TODO 1.4: loss.backward() (tính gradient bằng BPTT).
        # TODO 1.5: Clip gradient (max_norm=1.0) rồi optimizer.step().
        # TODO 1.6: Cộng loss * số ảnh trong batch và tổng số ảnh.
        raise NotImplementedError(
            "TODO 1: Sinh viên tự hoàn thiện training loop trước khi chạy."
        )
    return cumulative_loss / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, collect_errors=False):
    """Tính metric trên TOÀN TẬP, không lấy trung bình accuracy từng batch."""
    model.eval()
    cumulative_loss, count = 0.0, 0
    true_labels, predictions, wrong_examples = [], [], []
    for ids, lengths, labels, meta in loader:
        logits = model(ids.to(device), lengths)
        loss = loss_fn(logits, labels.to(device))
        probs = torch.softmax(logits, dim=1).cpu()
        pred = torch.argmax(probs, dim=1).tolist()
        gold = labels.tolist()
        count += len(gold)
        cumulative_loss += float(loss.item()) * len(gold)
        true_labels.extend(gold)
        predictions.extend(pred)
        if collect_errors:
            for (text, original_length, unk_ratio, was_truncated), y, p, probability in zip(
                meta, gold, pred, probs
            ):
                if y != p:
                    wrong_examples.append({
                        "text": text, "true_label": LABEL_NAMES[y],
                        "predicted_label": LABEL_NAMES[p],
                        "predicted_probability": round(float(probability[p]), 5),
                        "true_label_probability": round(float(probability[y]), 5),
                        "original_tokens": original_length,
                        "was_truncated": was_truncated,
                        "unk_ratio": round(float(unk_ratio), 4),
                    })
    return {
        "loss": cumulative_loss / count,
        "accuracy": accuracy_score(true_labels, predictions),
        "macro_f1": f1_score(true_labels, predictions, labels=[0, 1, 2, 3],
                             average="macro", zero_division=0),
        "cm": confusion_matrix(true_labels, predictions, labels=[0, 1, 2, 3]),
        "errors": wrong_examples,
    }


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# ======================================================================
# PHẦN 4. ĐÁNH GIÁ – BIỂU ĐỒ, MA TRẬN NHẦM LẪN, 5 CÂU SAI
# ======================================================================
def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves(histories, out_dir):
    """Vẽ train loss, validation loss và validation accuracy theo epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for name, history in histories.items():
        epoch = [r["epoch"] for r in history]
        for ax, field in zip(axes, ["train_loss", "validation_loss", "validation_accuracy"]):
            ax.plot(epoch, [r[field] for r in history], marker="o", label=name)
    for ax, title in zip(axes, ["Train loss", "Validation loss", "Validation accuracy"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Cross-entropy")
    axes[1].set_ylabel("Cross-entropy")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=160)
    plt.close(fig)


def plot_cm(cm, output, name):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), LABEL_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(4), LABEL_NAMES)
    ax.set_xlabel("Nhãn mô hình dự đoán (Predicted)")
    ax.set_ylabel("Nhãn đúng (True)")
    ax.set_title(f"Ma trận nhầm lẫn trên TEST: {name}")
    for row in range(4):
        for col in range(4):
            ax.text(col, row, str(int(cm[row, col])), ha="center", va="center",
                    color="white" if cm[row, col] > cm.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def choose_diverse_errors(errors, limit=5):
    """Ưu tiên 5 ví dụ thuộc các cặp nhãn nhầm khác nhau."""
    chosen, pair_set = [], set()
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


def write_error_analysis(errors, path, max_len):
    """Xuất ít nhất 5 câu sai cùng khung phân tích có căn cứ dữ liệu.

    Nguyên nhân ngữ nghĩa là giả thuyết sinh viên cần kiểm chứng, không tự bịa.
    """
    chosen = choose_diverse_errors(errors, limit=5)
    lines = ["# PHÂN TÍCH ÍT NHẤT 5 CÂU DỰ ĐOÁN SAI – TEST AG NEWS", "",
             "Các ví dụ dưới đây là kết quả dự đoán THẬT của mô hình được chọn bằng validation loss.",
             "Hãy đọc nội dung và viết nguyên nhân cụ thể cho từng bản tin.", ""]
    if len(chosen) < 5:
        lines.append(f"**Chỉ tìm được {len(chosen)} câu dự đoán sai trong test.**")
    for i, r in enumerate(chosen, 1):
        lines += [f"## Ví dụ {i}", "", f"**Nhãn đúng:** {r['true_label']}",
                  f"**Dự đoán:** {r['predicted_label']} (p={r['predicted_probability']:.3f})",
                  f"**Số token gốc:** {r['original_tokens']} | **Tỉ lệ `<unk>`:** {r['unk_ratio']:.2%}",
                  "", f"> {r['text'].replace(chr(10), ' ')[:800]}", ""]
        if r["was_truncated"]:
            lines.append(f"- **Quan sát kỹ thuật:** Tin bị cắt sau {max_len} token; kiểm tra xem phần bị cắt có chứa thông tin phân loại hay không.")
        elif r["unk_ratio"] >= 0.10:
            lines.append("- **Quan sát kỹ thuật:** Tỉ lệ từ ngoài vocabulary cao; kiểm tra từ khóa quan trọng có bị thay bằng `<unk>` hay không.")
        else:
            lines.append("- **Quan sát kỹ thuật:** Không thấy dấu hiệu nổi bật về cắt chuỗi/OOV; kiểm tra ngữ nghĩa câu, chủ đề giao thoa và từ khóa gây nhầm.")
        lines += ["- **Giải thích nguyên nhân cụ thể (BẮT BUỘC):** ...",
                  "- **Dẫn chứng từ nội dung bản tin:** ...",
                  "- **Hướng cải thiện có thể thử:** ...", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ======================================================================
# PHẦN 5. CHẠY FULL AG NEWS – CHỌN THEO VALIDATION, BÁO CÁO TEST
# ======================================================================
def run_full_experiments(settings: Settings = CFG):
    set_seed(settings.seed)
    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "settings.json").write_text(
        json.dumps({"settings": asdict(settings), "experiments": EXPERIMENTS},
                   indent=2, ensure_ascii=False), encoding="utf-8"
    )

    train_rows, valid_rows, test_rows = load_agnews_full(settings)
    vocab = build_vocab(train_rows, settings.max_vocab, settings.min_freq)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"Vocab size={len(vocab):,}; max sequence length={settings.max_len}")
    train_ds = NewsDataset(train_rows, vocab, settings.max_len)
    valid_ds = NewsDataset(valid_rows, vocab, settings.max_len)
    test_ds = NewsDataset(test_rows, vocab, settings.max_len)
    valid_loader = DataLoader(valid_ds, batch_size=settings.batch_size,
                              shuffle=False, collate_fn=collate_news)
    test_loader = DataLoader(test_ds, batch_size=settings.batch_size,
                             shuffle=False, collate_fn=collate_news)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | batch_size={settings.batch_size} | epochs={settings.epochs}")
    if device.type == "cpu":
        print("Lưu ý: 5 mô hình × toàn bộ AG News trên CPU có thể cần nhiều thời gian.")

    criterion = nn.CrossEntropyLoss()
    histories = {}
    records = []

    # TODO 4 (ĐÁNH GIÁ): sau khi chạy, mở so_sanh_test.csv để nhận xét:
    # - baseline so với cấu hình mức khá;
    # - nếu chọn nâng cao, baseline so với BiLSTM/Dropout;
    # - lập luận accuracy/macro-F1/loss, tham số, thời gian/epoch.
    # Không dùng test để chọn tên mô hình hoặc tinh chỉnh siêu tham số.

    # Chỉ train/validation trong vòng lặp này. Không đụng tới tập test.
    for name, model_config in EXPERIMENTS.items():
        print("\n" + "=" * 75 + f"\nCẤU HÌNH: {name} – {model_config}")
        set_seed(settings.seed)
        model = LSTMClassifier(len(vocab), **model_config).to(device)
        train_loader = DataLoader(
            train_ds, batch_size=settings.batch_size, shuffle=True,
            collate_fn=collate_news, generator=torch.Generator().manual_seed(settings.seed)
        )
        ids, lengths, _, _ = next(iter(train_loader))
        if name == "baseline":
            with torch.no_grad():
                embeddings = model.embedding(ids.to(device))
                logits = model(ids.to(device), lengths)
            print("Token IDs shape:", tuple(ids.shape),
                  "| lengths shape:", tuple(lengths.shape),
                  "| embedding shape:", tuple(embeddings.shape),
                  "| logits shape:", tuple(logits.shape))
        param_count = count_parameters(model)
        print("Trainable parameters:", format(param_count, ","))
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
        checkpoint = out_dir / f"best_{name}.pt"
        best_loss, best_epoch = float("inf"), None
        history = []
        for epoch in range(1, settings.epochs + 1):
            cuda_sync(device)
            start = time.perf_counter()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            validation = evaluate(model, valid_loader, criterion, device)
            cuda_sync(device)
            elapsed = time.perf_counter() - start
            rec = {"epoch": epoch, "train_loss": train_loss,
                   "validation_loss": validation["loss"],
                   "validation_accuracy": validation["accuracy"],
                   "validation_macro_f1": validation["macro_f1"],
                   "seconds": elapsed}
            history.append(rec)
            if validation["loss"] < best_loss:
                best_loss, best_epoch = validation["loss"], epoch
                torch.save(model.state_dict(), checkpoint)
            print(f"{name} | epoch {epoch}/{settings.epochs} | train_loss={train_loss:.4f} "
                  f"| val_loss={validation['loss']:.4f} "
                  f"| val_acc={validation['accuracy']:.2%} "
                  f"| val_macro_f1={validation['macro_f1']:.4f} | {elapsed:.1f}s")
        histories[name] = history
        save_csv(out_dir / f"history_{name}.csv", history)
        records.append({"config": name, **model_config,
                        "trainable_parameters": param_count,
                        "best_epoch": best_epoch,
                        "best_validation_loss": best_loss,
                        "mean_seconds_per_epoch_train_and_validation": sum(h["seconds"] for h in history)/len(history)})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_learning_curves(histories, out_dir)
    # Chỉ chọn mô hình tốt nhất theo validation loss, TRƯỚC khi tính test.
    selected = min(records, key=lambda r: r["best_validation_loss"])["config"]
    print("\nMô hình được chọn theo validation loss:", selected)

    # PHẦN 6. TEST CHÍNH THỨC: mọi cấu hình, chỉ sau khi chốt checkpoint.
    best_errors = []
    for record in records:
        name = record["config"]
        model = LSTMClassifier(len(vocab), **EXPERIMENTS[name]).to(device)
        model.load_state_dict(torch.load(out_dir / f"best_{name}.pt",
                                         map_location=device, weights_only=True))
        result = evaluate(model, test_loader, criterion, device, collect_errors=True)
        cm = result["cm"]
        assert int(cm.sum()) == len(test_ds)
        assert np.isclose(np.trace(cm)/np.sum(cm), result["accuracy"])
        record["test_loss"] = result["loss"]
        record["test_accuracy"] = result["accuracy"]
        record["test_macro_f1"] = result["macro_f1"]
        record["test_errors"] = len(result["errors"])
        plot_cm(cm, out_dir / f"confusion_matrix_test_{name}.png", name)
        save_csv(out_dir / f"wrong_test_{name}.csv", result["errors"])
        print(f"TEST | {name:15s} | loss={result['loss']:.4f} "
              f"| accuracy={result['accuracy']:.2%} | macro-F1={result['macro_f1']:.4f} "
              f"| correct={int(cm.trace())}/{int(cm.sum())}")
        if name == selected:
            best_errors = result["errors"]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_csv(out_dir / "so_sanh_test.csv", records)
    # TODO 5 (ĐÁNH GIÁ BẮT BUỘC): điền vào phần "Giải thích nguyên nhân"
    # trong phan_tich_5_loi_test.md ít nhất 5 trường hợp dự đoán sai.
    # Mã chỉ cung cấp VĂN BẢN, nhãn đúng/sai và dấu hiệu kỹ thuật;
    # sinh viên phải tự nêu nguyên nhân dựa trên nội dung từng câu.
    write_error_analysis(best_errors, out_dir / "phan_tich_5_loi_test.md", settings.max_len)
    summary = {"dataset": "fancyzhx/ag_news", "n_train": len(train_ds),
               "n_validation": len(valid_ds), "n_test": len(test_ds),
               "vocabulary_size": len(vocab), "best_by_validation_loss": selected,
               "test_results": records, "device": str(device)}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nĐã lưu kết quả thực nghiệm vào:", out_dir.resolve())
    print("Bảng TEST:", out_dir / "so_sanh_test.csv")
    print("Phân tích 5 lỗi TEST:", out_dir / "phan_tich_5_loi_test.md")
    return summary


if __name__ == "__main__":
    run_full_experiments(CFG)
