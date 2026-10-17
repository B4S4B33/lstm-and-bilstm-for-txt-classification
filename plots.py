"""
Visualisation for the AG News attention-comparison paper.

Reads output/bilstm_results.jsonl and output/lstm_results.jsonl (they use
different schemas; this normalises both) and writes figures to output/figures/:

  fig_comparison.png        grouped bars: accuracy + macro-F1 per attention head,
                            LSTM vs BiLSTM, with mean +/- std over seeds
  fig_ablation_grid.png     2x4 accuracy heatmap (direction x attention head)
  ablation_grid.csv         the same 2x4 numbers
  fig_confusion_matrices.png  best-seed confusion matrix per model (both families)
  fig_training_curves.png   val accuracy + val loss vs epoch, per family/head

NOTE: LSTM and BiLSTM were trained with different max_len / train sizes, so the
cross-family comparison is illustrative until the setups are aligned.

Standalone: run `python plots.py` from the project root.
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

OUT = "output"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
HEADS = ["baseline", "additive", "multiplicative", "self"]
HEAD_LABEL = {"baseline": "None", "additive": "Additive",
              "multiplicative": "Multiplicative", "self": "Self-attn"}
DIRS = ["LSTM", "BiLSTM"]


def head_of(model):
    m = model.lower()
    if "additive" in m:
        return "additive"
    if "multiplicative" in m:
        return "multiplicative"
    if "self" in m:
        return "self"
    if "baseline" in m or m in ("bilstm", "lstm"):
        return "baseline"
    raise ValueError(f"cannot classify model '{model}'")


def load_records(path, direction):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if direction == "BiLSTM":
                acc, f1, params = r["accuracy"], r["macro_f1"], r["n_params"]
                cm = r.get("confusion_matrix")
                hist = [(h["epoch"], h["val_loss"], h["val_acc"],
                         h["train_loss"], h["train_acc"]) for h in r.get("history", [])]
                clean = r.get("config", {}).get("max_epochs", 7) != 2  # drop debug runs
                preds = None
            else:
                acc, f1, params = r["test_accuracy"], r["test_macro_f1"], r["parameter_count"]
                cm = None
                hist = [(h["epoch"], h["val_loss"], h["val_accuracy"],
                         h["train_loss"], h["train_accuracy"]) for h in r.get("epoch_history", [])]
                clean = r.get("run_type") == "full"
                preds = r.get("test_predictions")
            recs.append({
                "direction": direction, "model": r["model"], "head": head_of(r["model"]),
                "seed": r.get("seed"), "acc": acc, "f1": f1, "params": params,
                "cm": cm, "hist": hist, "preds": preds, "clean": clean,
            })
    return [r for r in recs if r["clean"]]


def true_test_labels():
    df = pd.read_csv(os.path.join("data", "test.csv"))
    if "Class Index" not in df.columns:
        df = pd.read_csv(os.path.join("data", "test.csv"), header=None,
                         names=["Class Index", "Title", "Description"])
    df = df[pd.to_numeric(df["Class Index"], errors="coerce").notnull()]
    return (pd.to_numeric(df["Class Index"]).astype(int) - 1).to_numpy()


def agg(recs, direction, head, metric):
    vals = [r[metric] for r in recs if r["direction"] == direction and r["head"] == head]
    if not vals:
        return None, None
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return float(np.mean(vals)), std


# ---- load ----
recs = load_records(os.path.join(OUT, "lstm_results.jsonl"), "LSTM") + \
       load_records(os.path.join(OUT, "bilstm_results.jsonl"), "BiLSTM")

print("Loaded runs (debug/full filtered):")
for d in DIRS:
    for h in HEADS:
        n = sum(1 for r in recs if r["direction"] == d and r["head"] == h)
        am, asd = agg(recs, d, h, "acc")
        if am is not None:
            print(f"  {d:7s} {h:14s} n={n}  acc={am:.4f}+/-{asd:.4f}")

# ---- 1. comparison chart ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
x = np.arange(len(HEADS))
width = 0.38
for ax, metric, title in [(axes[0], "acc", "Test accuracy"),
                          (axes[1], "f1", "Macro-F1")]:
    for i, d in enumerate(DIRS):
        means, stds = [], []
        for h in HEADS:
            m, s = agg(recs, d, h, metric)
            means.append(m if m is not None else 0.0)
            stds.append(s if s is not None else 0.0)
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=4, label=d)
    ax.set_xticks(x)
    ax.set_xticklabels([HEAD_LABEL[h] for h in HEADS])
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Model comparison on AG News (setup not fully aligned across families)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_comparison.png"), dpi=150)
plt.close(fig)

# ---- 2. ablation grid (2x4 heatmap + CSV) ----
grid = np.zeros((len(DIRS), len(HEADS)))
grid_std = np.zeros_like(grid)
for i, d in enumerate(DIRS):
    for j, h in enumerate(HEADS):
        m, s = agg(recs, d, h, "acc")
        grid[i, j] = m or 0.0
        grid_std[i, j] = s or 0.0
pd.DataFrame(grid, index=DIRS, columns=[HEAD_LABEL[h] for h in HEADS]) \
    .to_csv(os.path.join(FIG, "ablation_grid.csv"), float_format="%.4f")

fig, ax = plt.subplots(figsize=(7, 3.2))
im = ax.imshow(grid, cmap="Blues", vmin=grid.min().min() - 0.01, vmax=grid.max().max() + 0.01)
ax.set_xticks(range(len(HEADS)))
ax.set_xticklabels([HEAD_LABEL[h] for h in HEADS])
ax.set_yticks(range(len(DIRS)))
ax.set_yticklabels(DIRS)
for i in range(len(DIRS)):
    for j in range(len(HEADS)):
        ax.text(j, i, f"{grid[i, j]:.3f}\n±{grid_std[i, j]:.3f}",
                ha="center", va="center", fontsize=9,
                color="white" if grid[i, j] > grid.mean() else "black")
ax.set_title("Accuracy: direction x attention head (mean ± std)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_ablation_grid.png"), dpi=150)
plt.close(fig)

# ---- 3. confusion matrices (best seed per model, both families) ----
y_true = true_test_labels()
fig, axes = plt.subplots(2, len(HEADS), figsize=(3.0 * len(HEADS), 6.4))
for i, d in enumerate(DIRS):
    for j, h in enumerate(HEADS):
        ax = axes[i, j]
        cand = [r for r in recs if r["direction"] == d and r["head"] == h]
        if not cand:
            ax.axis("off")
            continue
        best = max(cand, key=lambda r: r["acc"])
        cm = best["cm"]
        if cm is None and best["preds"] is not None:
            pred = np.asarray(best["preds"])
            if len(pred) == len(y_true):
                recon_acc = float((pred == y_true).mean())
                if abs(recon_acc - best["acc"]) < 1e-6:
                    cm = sk_confusion_matrix(y_true, pred).tolist()
                else:
                    print(f"  [skip CM] {d}/{h}: pred order mismatch "
                          f"(recon {recon_acc:.4f} vs stored {best['acc']:.4f})")
        if cm is None:
            ax.axis("off")
            continue
        cm = np.array(cm)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{d} {HEAD_LABEL[h]}\nseed {best['seed']} acc={best['acc']:.3f}", fontsize=8)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(CLASS_NAMES, fontsize=6)
        ax.set_xlabel("pred", fontsize=7)
        ax.set_ylabel("true", fontsize=7)
        for a in range(len(CLASS_NAMES)):
            for b in range(len(CLASS_NAMES)):
                ax.text(b, a, str(cm[a, b]), ha="center", va="center", fontsize=6)
fig.suptitle("Confusion matrices (best seed per model)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_confusion_matrices.png"), dpi=150)
plt.close(fig)

# ---- 4. training curves (mean over seeds, +/- std band) ----
def curve(direction, head):
    runs = [r for r in recs if r["direction"] == direction and r["head"] == h and r["hist"]]
    if not runs:
        return None
    min_ep = min(len(r["hist"]) for r in runs)
    vloss = np.array([ [r["hist"][k][1] for k in range(min_ep)] for r in runs ])
    vacc  = np.array([ [r["hist"][k][2] for k in range(min_ep)] for r in runs ])
    return np.arange(1, min_ep + 1), vacc, vloss

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for col, d in enumerate(DIRS):
    for row, (idx, lab) in enumerate([(2, "val_acc"), (1, "val_loss")]):
        ax = axes[row, col]
        for h in HEADS:
            c = curve(d, h)
            if c is None:
                continue
            ep, vacc, vloss = c
            data = vacc if idx == 2 else vloss
            ax.plot(ep, data.mean(axis=0), marker="o", label=HEAD_LABEL[h])
            if len(data) > 1:
                ax.fill_between(ep, data.min(axis=0), data.max(axis=0), alpha=0.15)
        ax.set_title(f"{d} - validation {lab}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
fig.suptitle("Training curves (mean over seeds, shaded = min/max across seeds)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_training_curves.png"), dpi=150)
plt.close(fig)

print("\nWrote figures to", os.path.abspath(FIG), ":")
for fn in sorted(os.listdir(FIG)):
    print("  ", fn)
