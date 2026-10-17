"""
Attention heatmaps for the BiLSTM attention models (the paper's interpretability
figure). Run in Colab where the trained checkpoints live.

For each attention model (additive, multiplicative, self-attention) it loads the
saved checkpoint, runs the same few sample headlines through it, and plots the
per-token attention saliency. Needs the checkpoints from train.py and the tensor
cache (which stores the vocabulary used in training).

Colab usage:
    python attention_heatmaps.py \
        --checkpoint_dir /content/drive/MyDrive/bilstm_checkpoints \
        --cache_dir /content/drive/MyDrive/bilstm_cache \
        --test_csv /content/test.csv \
        --out /content/drive/MyDrive/bilstm_results/figures/fig_attention_heatmaps.png

Optional: --seed 123 to force a specific seed's checkpoint (default: first found).
The LSTM heatmaps are produced the same way once the partner's models expose
attention weights; this script covers the three BiLSTM attention heads.
"""

import argparse
import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train as T

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
ATTN_MODELS = [
    ("bilstm_additive", "Additive"),
    ("bilstm_multiplicative", "Multiplicative"),
    ("bilstm_selfattn", "Self-attention"),
]


def load_vocab_from_cache(cache_dir):
    cands = sorted(glob.glob(os.path.join(cache_dir, "tensors_*.npz")))
    if not cands:
        raise FileNotFoundError(
            f"no tensor cache (tensors_*.npz) under {cache_dir}; needed for the vocab")
    vocab = np.load(cands[-1], allow_pickle=True)["vocab"].item()
    return vocab, os.path.basename(cands[-1])


def find_checkpoint(checkpoint_dir, model_name, seed=None):
    if seed is not None:
        p = os.path.join(checkpoint_dir, f"{model_name}_seed{seed}.pt")
        return p if os.path.exists(p) else None
    cands = sorted(glob.glob(os.path.join(checkpoint_dir, f"{model_name}_seed*.pt")))
    return cands[-1] if cands else None


def load_model(model_name, vocab_size, ckpt_path, device):
    model = T.get_model(model_name, vocab_size, embed_dim=100, hidden_dim=64,
                        num_layers=1, num_classes=4, dropout=0.3, attn_dim=32, pad_idx=0)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()


def sample_headlines(test_csv, max_len, per_class=1, n=4):
    """First headline of each class, up to n, for a diverse spread."""
    texts, labels = T.load_agnews(test_csv)
    picked, seen = [], set()
    for t, y in zip(texts, labels):
        if y not in seen:
            seen.add(y)
            picked.append((t, y))
        if len(picked) >= n:
            break
    return picked


def main():
    ap = argparse.ArgumentParser(description="Attention heatmaps for BiLSTM attention models")
    ap.add_argument("--checkpoint_dir", default="/content/drive/MyDrive/bilstm_checkpoints")
    ap.add_argument("--cache_dir", default="/content/drive/MyDrive/bilstm_cache")
    ap.add_argument("--test_csv", default="/content/test.csv")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="output/figures/fig_attention_heatmaps.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, cache_name = load_vocab_from_cache(args.cache_dir)
    vocab_size = len(vocab)
    print(f"vocab={vocab_size} (from cache {cache_name})")

    headlines = sample_headlines(args.test_csv, args.max_len)
    print(f"{len(headlines)} sample headlines (one per class where possible)")

    models = []
    for name, disp in ATTN_MODELS:
        ckpt = find_checkpoint(args.checkpoint_dir, name, args.seed)
        if ckpt is None:
            print(f"  [skip] {name}: no checkpoint in {args.checkpoint_dir}")
            continue
        print(f"  loaded {name} <- {os.path.basename(ckpt)}")
        models.append((name, disp, load_model(name, vocab_size, ckpt, device)))
    if not models:
        print("no checkpoints found; nothing to plot.")
        return

    fig, axes = plt.subplots(len(headlines), len(models),
                             figsize=(4.2 * len(models), 1.5 * len(headlines) + 0.8),
                             squeeze=False)
    for i, (text, gold) in enumerate(headlines):
        ids, length = T.encode(text, vocab, args.max_len)
        tokens = T.tokenize(text)[:args.max_len]
        length = min(len(tokens), args.max_len)
        x = torch.tensor([ids[0]], dtype=torch.long)
        lens = torch.tensor([length], dtype=torch.long)
        for j, (name, disp, model) in enumerate(models):
            ax = axes[i][j]
            with torch.no_grad():
                logits, saliency = model(x, lens, return_attention=True)
            pred = int(logits.argmax(1).item())
            sal = saliency[0, :length].cpu().numpy().astype(float)
            sal = sal / sal.max() if sal.max() > 0 else sal  # normalise for visibility
            ax.matshow(sal[None, :], cmap="Reds", aspect="auto", vmin=0, vmax=1)
            for k, tok in enumerate(tokens[:length]):
                ax.text(k, 0, tok, rotation=90, ha="center", va="center", fontsize=6,
                        color="white" if sal[k] > 0.6 else "black")
            ax.set_yticks([])
            ax.set_xticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(disp, fontsize=10)
            if j == 0:
                gold_bits = text[:60].replace("\n", " ")
                ax.set_ylabel(f"gold={CLASS_NAMES[gold]}", fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=40)

    fig.suptitle("Attention saliency per token (brighter = attended more); "
                 "predicted class shown under each strip via gold labels", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("wrote", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
