# Attention Comparison on AG News: Training Plan (Colab CPU)

Comparing 8 models on AG News text classification:

| # | Model |
|---|-------|
| 1 | LSTM (baseline) |
| 2 | BiLSTM (baseline) |
| 3 | LSTM + Additive |
| 4 | BiLSTM + Additive |
| 5 | LSTM + Multiplicative |
| 6 | BiLSTM + Multiplicative |
| 7 | LSTM + Self-Attention |
| 8 | BiLSTM + Self-Attention |

The 8 models form a 2x4 design: direction (uni / bi) x head (none / additive / multiplicative / self-attention).

---

## 1. Hyperparameters (CPU)

| Setting | Value |
|---|---|
| Vocab size | 20,000 |
| Max sequence length | 48 (verify against the 95th percentile of your data) |
| Embedding dim | 100 |
| LSTM hidden (per direction) | 64 |
| Layers | 1 |
| Attention dim (additive, self-attention) | 32 |
| Dropout | 0.3 (on embeddings and before the classifier) |
| Optimizer | Adam, lr 1e-3, weight decay 0 to 1e-5 |
| LR schedule | ReduceLROnPlateau (same for all models) |
| Batch size | 64 |
| Max epochs | 6-8 |
| Early stopping patience | 2 (on validation loss) |
| Gradient clip | 1.0 |
| Seeds | 2-3 |
| Mixed precision | None |

**Data size on CPU:** training on all 120k samples is slow. Use a stratified subset of about 30-40k training samples, and state this in the paper. Keep the full official test set (7,600) for evaluation.

**Debugging:** first run each model on a tiny subset (about 5k samples, 1-2 epochs) to check shapes, masking and the training loop before starting real runs.

---

## 2. Keep the setup identical across all 8 models

- **Same data split.** Use the official AG News train/test split. Carve a validation set (about 10%) out of train, stratified, with a fixed seed. Never tune on test.
- **Same preprocessing.** Same tokenizer, vocabulary size, max sequence length and padding for every model.
- **Same embeddings.** Random init for all (fastest, no downloads). Optionally add GloVe 100d as a second setting on one or two models only.
- **Same hidden size, dropout, optimizer, learning rate, batch size and epoch budget.**
- **BiLSTM size:** keep 64 per direction for both LSTM and BiLSTM. The BiLSTM then has about 2x the recurrent parameters, so always report parameter counts alongside accuracy. Optional extra if time allows: a BiLSTM at 32 per direction (roughly equal parameters).
- **Same attention dimension** for the additive and self-attention variants.
- **Labels:** the raw AG News CSV uses labels 1-4. Shift to 0-3.

---

## 3. Training practices

- **Multiple seeds.** At least 2-3 per model, report mean +/- std. Differences between attention types will likely be a few tenths of a percent, so a single run cannot tell them apart.
- **Early stopping** on validation loss, same patience for all. Save the best checkpoint.
- **Gradient clipping** at 1.0.
- **Mask padded tokens in the attention softmax.** This is the most common bug: if attention can attend to padding, the results and heatmaps are wrong.
- **Log everything:** train/val loss, accuracy, time per epoch.
- Set seeds for `random`, `numpy` and `torch`.

---

## 4. Implementation details

**Baselines (no attention):** use the last hidden state as the sentence vector (concatenated forward and backward states for BiLSTM). Use the true last non-pad token, via `pack_padded_sequence` or the lengths, not position -1.

**Additive:** `score = v^T tanh(W h + b)` over the hidden states, then softmax over time, using a learned context vector as the query.

**Multiplicative:** same learned context vector as the query, with a dot or bilinear score (`h^T W q`). Keep the query source identical to additive so the only difference is the scoring function.

**Self-attention:** single-head scaled dot-product over the LSTM outputs, followed by pooling (mean pool or a learned query) to get a fixed-size vector. Document which pooling you use.

---

## 5. Metrics to report

- Accuracy and macro-F1
- Per-class precision and recall
- Confusion matrix for the best model or two
- Parameter count, training time per epoch, inference time
- Mean +/- std over seeds
- Significance test (paired t-test or McNemar's test) between the best attention model and the baseline

---

## 6. Analysis for the paper

- **Ablation grid:** direction x attention type (the 2x4 table).
- **Attention heatmaps** for the same 3-4 headlines across all six attention models.
- **Interpretability check (optional):** remove the most-attended word and compare the prediction drop with removing a random word.
- **Error analysis:** where the models confuse Business and Sci/Tech, and whether attention helps.
- **Sequence-length analysis:** accuracy on short vs long inputs.

---

## 7. Sanity checks

- Typical accuracy for LSTM and attention models on AG News is roughly 91-92.5% (somewhat lower is plausible with a subset and small hidden size).
- About 25% accuracy means something is broken (usually masking or label indexing).
- 95%+ suggests leakage between train and test.

---

## 8. Colab practicalities

- Save one result row (JSON or CSV) per model and seed immediately after each run, and checkpoints to Google Drive, so a disconnect loses only the current run.
- Run models in a loop that skips anything already in the results file.
- Pre-tokenize once, save the tensors, and reuse them across runs.

---

## 9. Reproducibility checklist

- Fixed random seeds
- Library versions and hardware (Colab CPU)
- Full hyperparameter table (Section 1)
- Code released on GitHub
- A short "Experimental Setup" section in the paper with the table above
