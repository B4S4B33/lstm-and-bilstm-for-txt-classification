"""
Train and evaluate the 4 BiLSTM models from training_plan_cpu.md on AG News.

Self-contained: this single file holds the data loading, the four models, and
the training loop. Run it in Colab with:

    python train.py --out_dir /content/drive/MyDrive/bilstm_results \
                    --cache_dir /content/drive/MyDrive/bilstm_cache

Reads data/train.csv and data/test.csv as-is (data/ folder next to this file);
they are never modified. The tokenization runs automatically on the first run
and is cached, so you never run a separate preparation step.

Models (BiLSTM direction only, no unidirectional LSTM):
    1. bilstm                -> BiLSTM baseline
    2. bilstm_additive       -> BiLSTM + additive attention
    3. bilstm_multiplicative -> BiLSTM + multiplicative (bilinear) attention
    4. bilstm_selfattn       -> BiLSTM + single-head self-attention (mean pooled)

Each run is saved to results.jsonl immediately and skipped on re-run, so a
Colab disconnect only costs the current run.
"""

import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

PAD_IDX = 0
UNK_IDX = 1
VOCAB_SIZE = 20000
MAX_LEN = 64  # covers the ~95th percentile of AG News title+description length


# ===========================================================================
# Data loading / preprocessing (runs automatically; not a manual step)
# ===========================================================================

def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def build_vocab(texts, min_freq=2, vocab_size=VOCAB_SIZE):
    """Build a vocabulary capped at `vocab_size` entries (including specials)."""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
    # Reserve two slots for the special tokens already added.
    for token, _ in counter.most_common(vocab_size - 2):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(text, vocab, max_len=MAX_LEN):
    tokens = tokenize(text)
    ids = [vocab.get(t, UNK_IDX) for t in tokens[:max_len]]
    length = len(ids)
    ids = ids + [PAD_IDX] * (max_len - length)  # right-pad to max_len
    return ids, length


def load_agnews(csv_path):
    """Load an AG News CSV.

    Handles the 'Class Index,Title,Description' header used by data/*.csv and
    falls back to positional columns for headerless dumps. Labels are 1-4 and
    are shifted to 0-3. Returns combined 'title description' texts and labels.
    """
    df = pd.read_csv(csv_path)
    if "Class Index" not in df.columns:
        df = pd.read_csv(csv_path, header=None,
                         names=["Class Index", "Title", "Description"])
    df = df[pd.to_numeric(df["Class Index"], errors="coerce").notnull()].copy()
    df["Class Index"] = pd.to_numeric(df["Class Index"]).astype(int)
    texts = (df["Title"].astype(str) + " " + df["Description"].astype(str)).tolist()
    labels = (df["Class Index"] - 1).tolist()  # shift 1-4 -> 0-3
    return texts, labels


def stratified_subsample(labels, n_samples, seed=42):
    """Return indices for a stratified subset of `n_samples`."""
    rng = random.Random(seed)
    by_class = {}
    for idx, c in enumerate(labels):
        by_class.setdefault(c, []).append(idx)
    for c in by_class:
        rng.shuffle(by_class[c])

    classes = sorted(by_class)
    per_class = n_samples // len(classes)
    remainder = n_samples - per_class * len(classes)
    selected = []
    for i, c in enumerate(classes):
        take = per_class + (1 if i < remainder else 0)
        selected.extend(by_class[c][:take])
    rng.shuffle(selected)
    return selected


def stratified_split(labels, val_frac=0.1, seed=42):
    """Split indices into train/val stratified by class."""
    rng = random.Random(seed)
    by_class = {}
    for idx, c in enumerate(labels):
        by_class.setdefault(c, []).append(idx)
    train_idx, val_idx = [], []
    for c in sorted(by_class):
        idxs = by_class[c][:]
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_frac))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


class AGNewsTensorDataset(Dataset):
    """Returns (input_ids, length, label)."""

    def __init__(self, input_ids, lengths, labels):
        self.input_ids = input_ids
        self.lengths = lengths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def prepare_tensors(texts, labels, vocab, max_len=MAX_LEN):
    input_ids, lengths = [], []
    for text in texts:
        ids, length = encode(text, vocab, max_len)
        input_ids.append(ids)
        lengths.append(length)
    return input_ids, lengths, list(labels)


def build_datasets(train_csv, test_csv, n_train_samples=None, max_len=MAX_LEN,
                   vocab_size=VOCAB_SIZE, val_frac=0.1, seed=42, cache_dir=None):
    """Build train/val/test datasets with a shared vocabulary.

    On the first run this tokenizes and encodes everything; if `cache_dir` is
    set it saves the tensors there and reuses them on later runs, so the
    tokenization cost is paid once.
    """
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        n_tag = "all" if n_train_samples is None else str(n_train_samples)
        cache_path = os.path.join(
            cache_dir, f"tensors_len{max_len}_vocab{vocab_size}_n{n_tag}_seed{seed}.npz")

    if cache_path is not None and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        vocab = data["vocab"].item()
        train_ds = AGNewsTensorDataset(data["train_ids"].tolist(),
                                       data["train_len"].tolist(), data["train_y"].tolist())
        val_ds = AGNewsTensorDataset(data["val_ids"].tolist(),
                                     data["val_len"].tolist(), data["val_y"].tolist())
        test_ds = AGNewsTensorDataset(data["test_ids"].tolist(),
                                      data["test_len"].tolist(), data["test_y"].tolist())
        return {"train": train_ds, "val": val_ds, "test": test_ds,
                "vocab": vocab, "vocab_size": len(vocab)}

    all_train_texts, all_train_labels = load_agnews(train_csv)
    test_texts, test_labels = load_agnews(test_csv)

    # Stratified train/val split.
    train_idx, val_idx = stratified_split(all_train_labels, val_frac=val_frac, seed=seed)
    train_texts = [all_train_texts[i] for i in train_idx]
    train_labels = [all_train_labels[i] for i in train_idx]
    val_texts = [all_train_texts[i] for i in val_idx]
    val_labels = [all_train_labels[i] for i in val_idx]

    # Stratified subsample of the training split for CPU speed.
    if n_train_samples is not None and n_train_samples < len(train_labels):
        keep = stratified_subsample(train_labels, n_train_samples, seed=seed)
        train_texts = [train_texts[i] for i in keep]
        train_labels = [train_labels[i] for i in keep]

    vocab = build_vocab(train_texts, min_freq=2, vocab_size=vocab_size)

    train_ids, train_len, train_y = prepare_tensors(train_texts, train_labels, vocab, max_len)
    val_ids, val_len, val_y = prepare_tensors(val_texts, val_labels, vocab, max_len)
    test_ids, test_len, test_y = prepare_tensors(test_texts, test_labels, vocab, max_len)

    train_ds = AGNewsTensorDataset(train_ids, train_len, train_y)
    val_ds = AGNewsTensorDataset(val_ids, val_len, val_y)
    test_ds = AGNewsTensorDataset(test_ids, test_len, test_y)

    if cache_path is not None:
        np.savez(
            cache_path,
            vocab=np.array([vocab], dtype=object),
            train_ids=np.array(train_ids), train_len=np.array(train_len), train_y=np.array(train_y),
            val_ids=np.array(val_ids), val_len=np.array(val_len), val_y=np.array(val_y),
            test_ids=np.array(test_ids), test_len=np.array(test_len), test_y=np.array(test_y),
        )

    return {"train": train_ds, "val": val_ds, "test": test_ds,
            "vocab": vocab, "vocab_size": len(vocab)}


# ===========================================================================
# Models (BiLSTM only; no unidirectional LSTM)
# ===========================================================================

def _zero_pad_positions(saliency, lengths, seq_len, device):
    """Zero out attention saliency at padded positions. Input/output: (batch, seq_len)."""
    if lengths is None:
        return saliency
    mask = torch.arange(seq_len, device=device).expand(saliency.size(0), seq_len) >= lengths.unsqueeze(1).to(device)
    return saliency.masked_fill(mask, 0.0)


class BiLSTMBase(nn.Module):
    """BiLSTM baseline: last non-pad hidden state (concat forward/backward)."""

    def __init__(self, vocab_size, embed_dim=100, hidden_dim=64, num_layers=1,
                 num_classes=4, dropout=0.3, attn_dim=None, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True,
            batch_first=True, dropout=0 if num_layers == 1 else dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None):
        embedded = self.dropout(self.embedding(x))
        if lengths is not None:
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(embedded)
        # last layer forward = h_n[-2], backward = h_n[-1]
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        last_hidden = self.dropout(last_hidden)
        return self.classifier(last_hidden)


class BiLSTMAdditive(nn.Module):
    """BiLSTM + additive attention: score = v^T tanh(W h + b)."""

    def __init__(self, vocab_size, embed_dim=100, hidden_dim=64, num_layers=1,
                 num_classes=4, dropout=0.3, attn_dim=32, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True,
            batch_first=True, dropout=0 if num_layers == 1 else dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_W = nn.Linear(hidden_dim * 2, attn_dim, bias=True)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None, return_attention=False):
        batch_size, seq_len = x.shape
        embedded = self.dropout(self.embedding(x))
        if lengths is not None:
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
        else:
            out, _ = self.lstm(embedded)

        out = self.dropout(out)
        attn_scores = self.attn_v(self.tanh(self.attn_W(out))).squeeze(-1)  # (batch, seq_len)
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(context)
        if return_attention:
            return logits, _zero_pad_positions(attn_weights, lengths, seq_len, x.device)
        return logits


class BiLSTMMultiplicative(nn.Module):
    """BiLSTM + multiplicative attention: score = (W h) dot q, learned query."""

    def __init__(self, vocab_size, embed_dim=100, hidden_dim=64, num_layers=1,
                 num_classes=4, dropout=0.3, attn_dim=None, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True,
            batch_first=True, dropout=0 if num_layers == 1 else dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.query = nn.Parameter(torch.randn(hidden_dim * 2))
        self.attn_W = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None, return_attention=False):
        batch_size, seq_len = x.shape
        embedded = self.dropout(self.embedding(x))
        if lengths is not None:
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
        else:
            out, _ = self.lstm(embedded)

        out = self.dropout(out)
        transformed = self.attn_W(out)
        q = self.query.unsqueeze(0).expand(batch_size, -1)
        attn_scores = torch.sum(transformed * q.unsqueeze(1), dim=2)
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(context)
        if return_attention:
            return logits, _zero_pad_positions(attn_weights, lengths, seq_len, x.device)
        return logits


class BiLSTMSelfAttention(nn.Module):
    """BiLSTM + single-head scaled dot-product self-attention, mean-pooled."""

    def __init__(self, vocab_size, embed_dim=100, hidden_dim=64, num_layers=1,
                 num_classes=4, dropout=0.3, attn_dim=32, pad_idx=0, pool='mean'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True,
            batch_first=True, dropout=0 if num_layers == 1 else dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(hidden_dim * 2, attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim * 2, attn_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim * 2, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, hidden_dim * 2)
        self.pool = pool
        if pool == 'query':
            self.pool_query = nn.Parameter(torch.randn(attn_dim))
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.scale = math.sqrt(attn_dim)

    def forward(self, x, lengths=None, return_attention=False):
        batch_size, seq_len = x.shape
        embedded = self.dropout(self.embedding(x))
        if lengths is not None:
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
        else:
            out, _ = self.lstm(embedded)

        out = self.dropout(out)
        Q = self.q_proj(out)
        K = self.k_proj(out)
        V = self.v_proj(out)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, seq_len, seq_len)
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            scores = scores.masked_fill(mask, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = self.out_proj(torch.matmul(attn_weights, V))  # (batch, seq_len, 2*hidden_dim)

        if self.pool == 'mean':
            if lengths is not None:
                mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)
                mask = mask.float().unsqueeze(-1)
                context = torch.sum(attn_out * mask, dim=1) / lengths.unsqueeze(1).float()
            else:
                context = attn_out.mean(dim=1)
        elif self.pool == 'query':
            q = self.pool_query.unsqueeze(0).expand(batch_size, -1)
            pool_scores = torch.matmul(attn_out, q.unsqueeze(-1)).squeeze(-1) / self.scale
            if lengths is not None:
                mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
                pool_scores = pool_scores.masked_fill(mask, -1e9)
            pool_weights = F.softmax(pool_scores, dim=1)
            context = torch.sum(attn_out * pool_weights.unsqueeze(-1), dim=1)

        logits = self.classifier(context)
        if return_attention:
            # Token saliency for the mean-pool model: average attention each
            # token receives across all query positions -> (batch, seq_len).
            saliency = attn_weights.mean(dim=1)
            return logits, _zero_pad_positions(saliency, lengths, seq_len, x.device)
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


_MODEL_REGISTRY = {
    "bilstm": BiLSTMBase,
    "bilstm_additive": BiLSTMAdditive,
    "bilstm_multiplicative": BiLSTMMultiplicative,
    "bilstm_selfattn": BiLSTMSelfAttention,
}


def get_model(model_name, vocab_size, **kwargs):
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[model_name](vocab_size, **kwargs)


# ===========================================================================
# Metrics + seeding
# ===========================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred):
    """Accuracy, macro-F1, per-class precision/recall, and confusion matrix."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
    )
    n = len(CLASS_NAMES)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0
    )
    per_class = [
        {"class": CLASS_NAMES[i], "precision": float(precision[i]),
         "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
        for i in range(n)
    ]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ===========================================================================
# Train / evaluate one epoch and one (model, seed)
# ===========================================================================

def run_epoch(model, loader, criterion, optimizer=None, clip=1.0, device="cpu"):
    """One epoch. optimizer is None -> evaluation (no grad)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    n_batches = 0
    all_preds, all_labels, all_lengths = [], [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for input_ids, lengths, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_lengths.append(lengths.cpu().numpy())

    avg_loss = total_loss / max(1, n_batches)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_len = np.concatenate(all_lengths)
    return avg_loss, float((y_pred == y_true).mean()), y_true, y_pred, y_len


def train_one(model_name, datasets, vocab_size, cfg, seed, device="cpu",
              checkpoint_dir=None, predictions_dir=None, log_fn=print):
    set_seed(seed)

    train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=cfg["batch_size"], shuffle=False)

    model = get_model(
        model_name, vocab_size,
        embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"], num_classes=cfg["num_classes"],
        dropout=cfg["dropout"], attn_dim=cfg["attn_dim"], pad_idx=0,
    ).to(device)
    n_params = count_parameters(model)
    log_fn(f"  [{model_name} | seed {seed}] params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["sched_patience"])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    epoch_times = []
    history = []

    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer,
                                                    clip=cfg["clip"], device=device)
        val_loss, val_acc, _, _, _ = run_epoch(model, val_loader, criterion, None, device=device)
        epoch_times.append(time.time() - t0)
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc,
                        "epoch_time": epoch_times[-1], "lr": optimizer.param_groups[0]["lr"]})
        log_fn(f"  [{model_name} | seed {seed}] epoch {epoch} "
               f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
               f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({epoch_times[-1]:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["early_stop_patience"]:
                log_fn(f"  [{model_name} | seed {seed}] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    t0 = time.time()
    test_loss, _, y_true, y_pred, y_len = run_epoch(model, test_loader, criterion, None, device=device)
    infer_time = time.time() - t0
    test_metrics = compute_metrics(y_true, y_pred)
    test_metrics["test_loss"] = test_loss
    test_metrics["val_loss"] = best_val_loss
    test_metrics["best_epoch"] = best_epoch

    result = {
        "model": model_name, "seed": seed, "n_params": n_params, "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "avg_epoch_time": float(np.mean(epoch_times)) if epoch_times else None,
        "infer_time": infer_time, **test_metrics, "history": history, "config": cfg,
    }
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{model_name}_seed{seed}.pt"))
    # Optionally dump per-example test outputs for downstream analysis
    # (sequence-length, McNemar). Kept out of results.jsonl to keep it lean.
    if predictions_dir is not None:
        os.makedirs(predictions_dir, exist_ok=True)
        np.savez(os.path.join(predictions_dir, f"{model_name}_seed{seed}.npz"),
                 y_true=y_true, y_pred=y_pred, lengths=y_len)
    return result


# ===========================================================================
# Results bookkeeping (incremental save, skip already-done runs)
# ===========================================================================

def load_done_keys(results_jsonl):
    done = set()
    if results_jsonl and os.path.exists(results_jsonl):
        with open(results_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    done.add((row["model"], row["seed"]))
                except json.JSONDecodeError:
                    continue
    return done


def append_result(results_jsonl, result):
    with open(results_jsonl, "a") as f:
        f.write(json.dumps(result) + "\n")


def summarize_by_model(results_jsonl):
    by_model = {}
    if os.path.exists(results_jsonl):
        with open(results_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                by_model.setdefault(row["model"], []).append(row)
    summaries = []
    for model_name, rows in by_model.items():
        accs = [r["accuracy"] for r in rows]
        f1s = [r["macro_f1"] for r in rows]
        ep_times = [r["avg_epoch_time"] for r in rows if r["avg_epoch_time"]]
        num = len(accs)
        summaries.append({
            "model": model_name, "n_runs": len(rows), "n_params": rows[0]["n_params"],
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs, ddof=1) if num > 1 else 0.0),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s, ddof=1) if num > 1 else 0.0),
            "avg_epoch_time": float(np.mean(ep_times)) if ep_times else None,
        })
    return summaries


# ===========================================================================
# Config + main
# ===========================================================================

DEFAULT_CONFIG = {
    "embed_dim": 100, "hidden_dim": 64, "num_layers": 1, "num_classes": 4,
    "attn_dim": 32, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-5,
    "batch_size": 64, "max_epochs": 7, "early_stop_patience": 2, "sched_patience": 1,
    "clip": 1.0,
}

MODELS = ["bilstm", "bilstm_additive", "bilstm_multiplicative", "bilstm_selfattn"]
SEEDS = [42, 123, 2024]


def parse_args():
    p = argparse.ArgumentParser(description="Train BiLSTM models on AG News (Colab CPU)")
    p.add_argument("--train_csv", default="/content/train.csv")
    p.add_argument("--test_csv", default="/content/test.csv")
    p.add_argument("--out_dir", default="/content/drive/MyDrive/bilstm_results")
    p.add_argument("--cache_dir", default="/content/drive/MyDrive/bilstm_cache")
    p.add_argument("--checkpoint_dir", default="/content/drive/MyDrive/bilstm_checkpoints")
    p.add_argument("--predictions_dir", default=None,
                   help="if set, save per-example test predictions/lengths per run here")
    p.add_argument("--n_train_samples", type=int, default=40000,
                   help="stratified subset of train for CPU speed; use -1 for full")
    p.add_argument("--max_len", type=int, default=MAX_LEN,
                   help="64 covers the ~95th percentile of AG News title+desc length")
    p.add_argument("--vocab_size", type=int, default=VOCAB_SIZE)
    p.add_argument("--debug", action="store_true",
                   help="tiny run: 5k samples, 2 epochs, one seed, to check the pipeline")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = dict(DEFAULT_CONFIG)
    seeds = SEEDS
    n_train = args.n_train_samples
    if args.debug:
        cfg["max_epochs"] = 2
        seeds = [42]
        n_train = 5000
        # Keep debug runs in separate folders so they cannot land in (and then
        # get skipped in favor of) the real results or checkpoints.
        args.out_dir = args.out_dir + "_debug"
        if args.checkpoint_dir is not None:
            args.checkpoint_dir = args.checkpoint_dir + "_debug"
        if args.predictions_dir is not None:
            args.predictions_dir = args.predictions_dir + "_debug"
        print("DEBUG MODE: 5000 samples, 2 epochs, seed 42 only ->", args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    data = build_datasets(
        train_csv=args.train_csv, test_csv=args.test_csv,
        n_train_samples=(None if n_train == -1 else n_train),
        max_len=args.max_len, vocab_size=args.vocab_size,
        val_frac=0.1, seed=42, cache_dir=args.cache_dir,
    )
    datasets = {"train": data["train"], "val": data["val"], "test": data["test"]}
    vocab_size = data["vocab_size"]
    print(f"Data ready in {time.time()-t0:.1f}s | vocab={vocab_size} "
          f"train={len(datasets['train'])} val={len(datasets['val'])} test={len(datasets['test'])}")

    # Record the full run setup so each result row is self-documenting for the
    # paper's reproducibility checklist.
    cfg["vocab_size"] = vocab_size
    cfg["max_len"] = args.max_len
    cfg["n_train_samples"] = len(datasets["train"])
    cfg["n_val_samples"] = len(datasets["val"])
    cfg["n_test_samples"] = len(datasets["test"])
    cfg["seeds"] = list(seeds)

    results_jsonl = os.path.join(args.out_dir, "results.jsonl")
    done = load_done_keys(results_jsonl)

    for model_name in MODELS:
        for seed in seeds:
            if (model_name, seed) in done:
                print(f"SKIP {model_name} seed {seed} (already in results)")
                continue
            print(f"TRAIN {model_name} seed {seed}")
            result = train_one(model_name, datasets, vocab_size, cfg, seed,
                               device=device, checkpoint_dir=args.checkpoint_dir,
                               predictions_dir=args.predictions_dir)
            append_result(results_jsonl, result)
            print(f"  -> test_acc={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f}")

    summaries = summarize_by_model(results_jsonl)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n=== SUMMARY (mean +/- std over seeds) ===")
    for s in summaries:
        print(f"  {s['model']:22s} acc={s['accuracy_mean']:.4f}+/-{s['accuracy_std']:.4f} "
              f"f1={s['macro_f1_mean']:.4f} params={s['n_params']:,} n={s['n_runs']}")


if __name__ == "__main__":
    main()
