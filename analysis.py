"""
Deeper analysis + visualisation for the AG News attention paper.

Builds on plots.py. Reads output/bilstm_results.jsonl and output/lstm_results.jsonl
(their two schemas are normalised) and writes to output/figures/:

  fig_per_class_f1.png        per-class F1 heatmap across all 8 models
  fig_error_business_scitech.png  Business<->Sci/Tech confusion rate per model
  fig_seq_length.png          accuracy binned by input token length
  significance_tests.txt      McNemar + paired t-test (best attention vs baseline)

Predictions availability:
  - LSTM: inline `test_predictions` in lstm_results.jsonl (full support now).
  - BiLSTM: needs `--predictions_dir` (produced by train.py --predictions_dir).
    Without it, per-class and confusion figures still work (from stored metrics),
    but BiLSTM sequence-length and McNemar are skipped with a printed note.

Run: `python analysis.py` from the project root (optionally --predictions_dir).
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2 as _chi2, ttest_rel
from sklearn.metrics import confusion_matrix as sk_cm

OUT = "output"
FIG = os.path.join(OUT, "figures")
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
BUS, SCITECH = 2, 3  # class indices of interest
HEADS = ["baseline", "additive", "multiplicative", "self"]
HEAD_LABEL = {"baseline": "None", "additive": "Additive",
              "multiplicative": "Multiplicative", "self": "Self-attn"}
DIRS = ["LSTM", "BiLSTM"]


def head_of(model):
    m = model.lower()
    for key in ("additive", "multiplicative", "self"):
        if key in m:
            return key
    return "baseline"


def _clean_max_epochs(r):
    return r.get("config", {}).get("max_epochs", 7) != 2


def load_bilstm(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not re.search(r"[{}\[\]]", line):
                continue
            r = json.loads(line)
            if not _clean_max_epochs(r):
                continue
            recs.append({
                "direction": "BiLSTM", "model": r["model"], "head": head_of(r["model"]),
                "seed": r["seed"], "acc": r["accuracy"], "f1": r["macro_f1"],
                "per_class": r.get("per_class"), "cm": r.get("confusion_matrix"),
            })
    return recs


def load_lstm(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not re.search(r"[{}\[\]]", line):
                continue
            r = json.loads(line)
            if r.get("run_type") != "full":
                continue
            # normalise per-class into the list-of-dict form
            prd = r.get("per_class_precision_recall", {})
            per_class = [{"class": c, "precision": prd[c]["precision"],
                          "recall": prd[c]["recall"], "f1": prd[c]["f1"]}
                         for c in CLASS_NAMES if c in prd]
            recs.append({
                "direction": "LSTM", "model": r["model"], "head": head_of(r["model"]),
                "seed": r["seed"], "acc": r["test_accuracy"], "f1": r["test_macro_f1"],
                "per_class": per_class, "cm": None,
                "preds": r.get("test_predictions"),
                "max_len": r.get("max_sequence_length", 48),
            })
    return recs


def true_test_labels():
    df = pd.read_csv(os.path.join("data", "test.csv"))
    if "Class Index" not in df.columns:
        df = pd.read_csv(os.path.join("data", "test.csv"), header=None,
                         names=["Class Index", "Title", "Description"])
    df = df[pd.to_numeric(df["Class Index"], errors="coerce").notnull()]
    return (pd.to_numeric(df["Class Index"]).astype(int) - 1).to_numpy()


def token_lengths(max_len):
    import train as T
    texts, _ = T.load_agnews(os.path.join("data", "test.csv"))
    return np.array([min(len(T.tokenize(t)), max_len) for t in texts])


def agg_per_class_f1(recs):
    """model -> mean per-class F1 over seeds (list of 4)."""
    out = {}
    by = {}
    for r in recs:
        if not r["per_class"]:
            continue
        row = [c["f1"] for c in r["per_class"]]
        by.setdefault((r["direction"], r["head"]), []).append(row)
    for k, rows in by.items():
        out[k] = np.mean(np.array(rows), axis=0)
    return out


def get_cm(r):
    if r.get("cm") is not None:
        return np.array(r["cm"])
    if r.get("preds") is not None:
        y_true = true_test_labels()
        pred = np.asarray(r["preds"])
        if len(pred) == len(y_true):
            return sk_cm(y_true, pred)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_dir", default=os.path.join(OUT, "predictions"))
    args = ap.parse_args()

    bilstm = load_bilstm(os.path.join(OUT, "bilstm_results.jsonl"))
    lstm = load_lstm(os.path.join(OUT, "lstm_results.jsonl"))
    recs = lstm + bilstm

    # Load BiLSTM per-example predictions if available.
    bilstm_preds = {}
    if args.predictions_dir and os.path.isdir(args.predictions_dir):
        for p in glob.glob(os.path.join(args.predictions_dir, "bilstm*_seed*.npz")):
            base = os.path.basename(p)[:-4]
            m, seed = base.rsplit("_seed", 1)
            d = np.load(p)
            bilstm_preds[(m, int(seed))] = {"y_true": d["y_true"], "y_pred": d["y_pred"],
                                             "lengths": d["lengths"]}
    if bilstm_preds:
        print(f"BiLSTM predictions loaded from {args.predictions_dir}: {len(bilstm_preds)} runs")
    else:
        print(f"No BiLSTM predictions at {args.predictions_dir}; "
              "BiLSTM sequence-length and McNemar will be skipped.")

    # ---- fig 1: per-class F1 heatmap ----
    f1map = agg_per_class_f1(recs)
    rows, labels = [], []
    for d in DIRS:
        for h in HEADS:
            if (d, h) in f1map:
                rows.append(f1map[(d, h)])
                labels.append(f"{d} {HEAD_LABEL[h]}")
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(6.5, 0.6 * len(rows) + 2))
    im = ax.imshow(rows, cmap="Blues", aspect="auto",
                   vmin=rows.min() - 0.02, vmax=rows.max() + 0.02)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{rows[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if rows[i, j] > rows.mean() else "black")
    ax.set_title("Per-class F1 (mean over seeds)")
    ax.set_xlabel("class")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_per_class_f1.png"), dpi=150)
    plt.close(fig)

    # ---- fig 2: Business <-> Sci/Tech confusion rate ----
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(HEADS))
    w = 0.38
    for i, d in enumerate(DIRS):
        b2s, s2b = [], []
        for h in HEADS:
            runs = [r for r in recs if r["direction"] == d and r["head"] == h]
            cms = [get_cm(r) for r in runs]
            cms = [c for c in cms if c is not None]
            if not cms:
                b2s.append(0.0); s2b.append(0.0); continue
            cm = np.mean(cms, axis=0)
            b2s.append(cm[BUS, SCITECH] / cm[BUS].sum())
            s2b.append(cm[SCITECH, BUS] / cm[SCITECH].sum())
        ax.bar(x + (i - 0.5) * w - 0.09, b2s, w * 0.45, label=f"{d}: Business->Sci/Tech")
        ax.bar(x + (i - 0.5) * w + 0.09, s2b, w * 0.45, label=f"{d}: Sci/Tech->Business")
    ax.set_xticks(x)
    ax.set_xticklabels([HEAD_LABEL[h] for h in HEADS])
    ax.set_ylabel("off-diagonal rate within the class")
    ax.set_title("Business vs Sci/Tech confusion (lower is better)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_error_business_scitech.png"), dpi=150)
    plt.close(fig)

    # ---- fig 3: sequence-length accuracy ----
    bins = [1, 17, 33, 65]
    bin_labels = ["1-16", "17-32", "33-64"]
    len_cache = {}

    def bins_acc(y_true, y_pred, lengths):
        out = []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (lengths >= a) & (lengths < b)
            out.append(float((y_pred[m] == y_true[m]).mean()) if m.any() else np.nan)
        return out

    def collect(direction, head):
        curves = []
        for r in [r for r in recs if r["direction"] == direction and r["head"] == head]:
            if direction == "LSTM":
                if r.get("preds") is None or len(r["preds"]) != len(true_test_labels()):
                    continue
                if "lstm" not in len_cache:
                    len_cache["lstm"] = token_lengths(r["max_len"])
                curves.append(bins_acc(true_test_labels(), np.asarray(r["preds"]), len_cache["lstm"]))
            else:
                key = (r["model"], r["seed"])
                if key not in bilstm_preds:
                    continue
                dp = bilstm_preds[key]
                curves.append(bins_acc(dp["y_true"], dp["y_pred"], dp["lengths"]))
        return np.array(curves) if curves else None

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plotted = False
    for d, style in [(DIRS[0], "o"), (DIRS[1], "s")]:
        for h in HEADS:
            cur = collect(d, h)
            if cur is None:
                continue
            plotted = True
            mean = np.nanmean(cur, axis=0)
            ax.plot(range(len(bin_labels)), mean, marker=style, label=f"{d} {HEAD_LABEL[h]}")
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels([f"{b} tokens" for b in bin_labels])
    ax.set_ylabel("test accuracy")
    ax.set_xlabel("input length bin")
    ax.set_title("Accuracy by input sequence length (mean over seeds)")
    if plotted:
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, "no per-example predictions available", ha="center", va="center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_seq_length.png"), dpi=150)
    plt.close(fig)
    if not plotted:
        print("seq-length figure empty: no predictions (LSTM had none matching / BiLSTM dir absent)")

    # ---- significance tests ----
    lines = []
    lines.append("Significance tests: best attention model (additive) vs baseline")
    lines.append("=" * 60)

    def paired_preds(direction, m_attn, m_base):
        """[(seed, y_true, attn_pred, base_pred)] for shared seeds."""
        out = []
        if direction == "LSTM":
            by = {}
            for r in recs:
                if r["direction"] == "LSTM" and r["model"] in (m_attn, m_base):
                    by.setdefault(r["seed"], {})[r["model"]] = r
            for seed, d in by.items():
                if m_attn in d and m_base in d and d[m_attn].get("preds") is not None \
                        and d[m_base].get("preds") is not None:
                    yt = true_test_labels()
                    out.append((seed, yt, np.asarray(d[m_attn]["preds"]),
                                np.asarray(d[m_base]["preds"])))
        else:
            seeds = {r["seed"] for r in recs
                     if r["direction"] == "BiLSTM" and r["model"] in (m_attn, m_base)}
            for seed in seeds:
                ka, kb = (m_attn, seed), (m_base, seed)
                if ka in bilstm_preds and kb in bilstm_preds:
                    out.append((seed, bilstm_preds[ka]["y_true"],
                                bilstm_preds[ka]["y_pred"], bilstm_preds[kb]["y_pred"]))
        return out

    def mcnemar(y_true, p_attn, p_base):
        ca = (p_attn == y_true)
        cb = (p_base == y_true)
        b = int(np.sum(ca & ~cb))   # attention right, baseline wrong
        c = int(np.sum(~ca & cb))   # baseline right, attention wrong
        if b + c == 0:
            return b, c, None
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p = float(_chi2.sf(chi2, 1))
        return b, c, p

    for direction, m_attn, m_base in [("LSTM", "lstm_additive", "lstm_baseline"),
                                      ("BiLSTM", "bilstm_additive", "bilstm")]:
        pairs = paired_preds(direction, m_attn, m_base)
        # Paired t-test over seeds using stored per-run accuracies.
        accs_a = [r["acc"] for r in recs
                  if r["direction"] == direction and r["model"] == m_attn]
        accs_b = [r["acc"] for r in recs
                  if r["direction"] == direction and r["model"] == m_base]
        lines.append(f"\n[{direction}] {m_attn} vs {m_base}")
        if pairs:
            for seed, yt, pa, pb in sorted(pairs):
                b, c, p = mcnemar(yt, pa, pb)
                lines.append(f"  seed {seed}: McNemar b(attn-only-correct)={b} "
                             f"c(base-only-correct)={c} "
                             f"p={('%.4g' % p) if p is not None else 'n/a'}")
        else:
            lines.append("  McNemar: skipped (no paired per-example predictions available)")
        if len(accs_a) == len(accs_b) and len(accs_a) >= 2:
            t, p = ttest_rel(accs_a, accs_b)
            lines.append(f"  paired t-test over {len(accs_a)} seeds: t={t:.3f} p={p:.4g} "
                         f"mean delta={np.mean(np.array(accs_a)-np.array(accs_b)):+.4f}")
        else:
            lines.append("  paired t-test: skipped (need >=2 matching seeds)")

    sig_path = os.path.join(FIG, "significance_tests.txt")
    with open(sig_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nWrote:", ", ".join(sorted(os.listdir(FIG))))


if __name__ == "__main__":
    main()
