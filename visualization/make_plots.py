"""
Visualisation for the AG News attention-comparison paper (fair comparison).

Reads output/summary.csv and output/results .jsonl (aligned settings:
full train.csv, MAX_LEN=48, seeds=[13,42], early_stop_patience=2) and
writes figures to output/plots/.

Run: `python visualization/make_plots.py` from the project root.
"""

import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

ATTN_ORDER = ["baseline", "additive", "multiplicative", "self_attention"]
ATTN_LABEL = {
    "baseline": "Baseline (last hidden)",
    "additive": "Additive attention",
    "multiplicative": "Multiplicative attention",
    "self_attention": "Self-attention",
}
ARCH_COLORS = {"LSTM": "#4C72B0", "BiLSTM": "#55A868"}

from matplotlib.patches import Patch


def split_model(name):
    """Return (architecture, attention_type) from 'lstm_additive' / 'bilstm_baseline'."""
    parts = name.split("_")
    arch = parts[0]
    attn = "_".join(parts[1:])
    return arch, attn


# ---------- Load summary.csv ----------
summary = []
with open(os.path.join(OUT_DIR, "summary.csv"), newline="") as f:
    for row in csv.DictReader(f):
        arch, attn = split_model(row["model"])
        summary.append({
            "model": row["model"],
            "arch": arch,
            "attn": attn,
            "arch_label": "BiLSTM" if arch == "bilstm" else "LSTM",
            "seeds": int(row["seeds"]),
            "acc_mean": float(row["accuracy_mean"]),
            "acc_std": float(row["accuracy_std"]),
            "f1_mean": float(row["macro_f1_mean"]),
            "f1_std": float(row["macro_f1_std"]),
            "params": float(row["parameters"]),
            "val_loss_mean": float(row["best_validation_loss_mean"]),
            "train_sec_mean": float(row["train_seconds_mean"]),
            "infer_ms_mean": float(row["inference_ms_per_example_mean"]),
        })

# ---------- Load per-run results ----------
runs = []
with open(os.path.join(OUT_DIR, "results .jsonl")) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        arch, attn = split_model(r["model"])
        runs.append({
            "model": r["model"],
            "arch": arch,
            "attn": attn,
            "arch_label": "BiLSTM" if arch == "bilstm" else "LSTM",
            "seed": r["seed"],
            "test_acc": r["test_accuracy"],
            "macro_f1": r["test_macro_f1"],
        })


def get_summary(arch, attn):
    return next(s for s in summary if s["arch"] == arch and s["attn"] == attn)


# ============================================================
# PLOT 1: Grouped bar — Test Accuracy by architecture + attention
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(ATTN_ORDER))
width = 0.35

lstm_acc = [get_summary("lstm", a)["acc_mean"] for a in ATTN_ORDER]
bilstm_acc = [get_summary("bilstm", a)["acc_mean"] for a in ATTN_ORDER]
lstm_acc_std = [get_summary("lstm", a)["acc_std"] for a in ATTN_ORDER]
bilstm_acc_std = [get_summary("bilstm", a)["acc_std"] for a in ATTN_ORDER]

bars1 = ax.bar(x - width / 2, lstm_acc, width, label="LSTM", color=ARCH_COLORS["LSTM"],
               yerr=lstm_acc_std, capsize=4, edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width / 2, bilstm_acc, width, label="BiLSTM", color=ARCH_COLORS["BiLSTM"],
               yerr=bilstm_acc_std, capsize=4, edgecolor="black", linewidth=0.5)

for bars in (bars1, bars2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Test Accuracy")
ax.set_title("Test Accuracy by Architecture and Attention Variant\n(mean ± std over 2 seeds, AG News)")
ax.set_xticks(x)
ax.set_xticklabels([ATTN_LABEL[a] for a in ATTN_ORDER], rotation=10, ha="right")
ax.set_ylim(0.90, 0.93)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "test_accuracy_grouped.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# PLOT 2: Grouped bar — Macro-F1 by architecture + attention
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

lstm_f1 = [get_summary("lstm", a)["f1_mean"] for a in ATTN_ORDER]
bilstm_f1 = [get_summary("bilstm", a)["f1_mean"] for a in ATTN_ORDER]
lstm_f1_std = [get_summary("lstm", a)["f1_std"] for a in ATTN_ORDER]
bilstm_f1_std = [get_summary("bilstm", a)["f1_std"] for a in ATTN_ORDER]

bars1 = ax.bar(x - width / 2, lstm_f1, width, label="LSTM", color=ARCH_COLORS["LSTM"],
               yerr=lstm_f1_std, capsize=4, edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width / 2, bilstm_f1, width, label="BiLSTM", color=ARCH_COLORS["BiLSTM"],
               yerr=bilstm_f1_std, capsize=4, edgecolor="black", linewidth=0.5)

for bars in (bars1, bars2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Macro-F1")
ax.set_title("Macro-F1 by Architecture and Attention Variant\n(mean ± std over 2 seeds, AG News)")
ax.set_xticks(x)
ax.set_xticklabels([ATTN_LABEL[a] for a in ATTN_ORDER], rotation=10, ha="right")
ax.set_ylim(0.90, 0.93)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "macro_f1_grouped.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# PLOT 3: Bar — Training time per epoch
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

all_models = [s["model"] for s in summary]
train_times = [s["train_sec_mean"] for s in summary]
colors = [ARCH_COLORS[s["arch_label"]] for s in summary]

bars = ax.bar(range(len(all_models)), train_times, color=colors, edgecolor="black", linewidth=0.5)
for bar, t in zip(bars, train_times):
    h = bar.get_height()
    ax.annotate(f"{t:.0f}s", xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Train Time per Epoch (seconds)")
ax.set_title("Training Time per Epoch by Model")
ax.set_xticks(range(len(all_models)))
ax.set_xticklabels([s["model"].replace("_", "\n") for s in summary], rotation=45, ha="right", fontsize=8)
ax.grid(axis="y", alpha=0.3)

legend_elements = [Patch(facecolor=ARCH_COLORS["LSTM"], label="LSTM"),
                   Patch(facecolor=ARCH_COLORS["BiLSTM"], label="BiLSTM")]
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "train_time.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# PLOT 4: Bar — Inference latency per example
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

infer_times = [s["infer_ms_mean"] for s in summary]
bars = ax.bar(range(len(all_models)), infer_times, color=colors, edgecolor="black", linewidth=0.5)
for bar, t in zip(bars, infer_times):
    h = bar.get_height()
    ax.annotate(f"{t:.2f}ms", xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Inference Latency (ms per example)")
ax.set_title("Inference Latency per Example by Model")
ax.set_xticks(range(len(all_models)))
ax.set_xticklabels([s["model"].replace("_", "\n") for s in summary], rotation=45, ha="right", fontsize=8)
ax.grid(axis="y", alpha=0.3)
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "inference_latency.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# PLOT 5: Bar — Parameter count
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

params = [s["params"] / 1e6 for s in summary]
bars = ax.bar(range(len(all_models)), params, color=colors, edgecolor="black", linewidth=0.5)
for bar, p in zip(bars, params):
    h = bar.get_height()
    ax.annotate(f"{p:.2f}M", xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Parameters (millions)")
ax.set_title("Model Parameter Count by Model")
ax.set_xticks(range(len(all_models)))
ax.set_xticklabels([s["model"].replace("_", "\n") for s in summary], rotation=45, ha="right", fontsize=8)
ax.grid(axis="y", alpha=0.3)
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "parameter_count.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# PLOT 6: Per-seed dot plot — Test Accuracy
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

model_order = []
for attn in ATTN_ORDER:
    model_order.append(("lstm", attn))
    model_order.append(("bilstm", attn))

y_pos = []
y_labels = []
for i, (arch, attn) in enumerate(model_order):
    y_pos.append(i)
    y_labels.append(f"{arch.upper()} + {ATTN_LABEL[attn]}")

for i, (arch, attn) in enumerate(model_order):
    seed_runs = [r for r in runs if r["arch"] == arch and r["attn"] == attn]
    for r in seed_runs:
        ax.scatter(r["test_acc"], i, color=ARCH_COLORS[r["arch_label"]], s=60, zorder=3,
                   edgecolor="black", linewidth=0.5)
    mean_acc = get_summary(arch, attn)["acc_mean"]
    ax.axvline(mean_acc, color=ARCH_COLORS[r["arch_label"]], alpha=0.4,
               linestyle="--", linewidth=1, zorder=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=8)
ax.set_xlabel("Test Accuracy")
ax.set_title("Per-Seed Test Accuracy by Model (dots = individual runs, dashed = mean)")
ax.set_xlim(0.90, 0.93)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "per_seed_accuracy.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# PLOT 7: Combined 2x2 overview figure
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
bars1 = ax.bar(x - width / 2, lstm_acc, width, label="LSTM", color=ARCH_COLORS["LSTM"],
               yerr=lstm_acc_std, capsize=3, edgecolor="black", linewidth=0.4)
bars2 = ax.bar(x + width / 2, bilstm_acc, width, label="BiLSTM", color=ARCH_COLORS["BiLSTM"],
               yerr=bilstm_acc_std, capsize=3, edgecolor="black", linewidth=0.4)
ax.set_title("Test Accuracy", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([ATTN_LABEL[a] for a in ATTN_ORDER], rotation=15, ha="right", fontsize=8)
ax.set_ylim(0.90, 0.93)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

ax = axes[0, 1]
bars1 = ax.bar(x - width / 2, lstm_f1, width, label="LSTM", color=ARCH_COLORS["LSTM"],
               yerr=lstm_f1_std, capsize=3, edgecolor="black", linewidth=0.4)
bars2 = ax.bar(x + width / 2, bilstm_f1, width, label="BiLSTM", color=ARCH_COLORS["BiLSTM"],
               yerr=bilstm_f1_std, capsize=3, edgecolor="black", linewidth=0.4)
ax.set_title("Macro-F1", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([ATTN_LABEL[a] for a in ATTN_ORDER], rotation=15, ha="right", fontsize=8)
ax.set_ylim(0.90, 0.93)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

ax = axes[1, 0]
bars = ax.bar(range(len(all_models)), train_times, color=colors, edgecolor="black", linewidth=0.4)
ax.set_title("Train Time per Epoch (s)", fontsize=11)
ax.set_xticks(range(len(all_models)))
ax.set_xticklabels([s["model"] for s in summary], rotation=45, ha="right", fontsize=7)
ax.grid(axis="y", alpha=0.3)

ax = axes[1, 1]
bars = ax.bar(range(len(all_models)), infer_times, color=colors, edgecolor="black", linewidth=0.4)
ax.set_title("Inference Latency (ms/ex)", fontsize=11)
ax.set_xticks(range(len(all_models)))
ax.set_xticklabels([s["model"] for s in summary], rotation=45, ha="right", fontsize=7)
ax.grid(axis="y", alpha=0.3)

plt.suptitle("LSTM vs BiLSTM Attention — AG News Comparison (fair, identical settings)",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(PLOT_DIR, "overview_combined.png"), dpi=150, bbox_inches="tight")
plt.close()

print("All plots saved to:", PLOT_DIR)
for fname in sorted(os.listdir(PLOT_DIR)):
    print("  ", fname)
