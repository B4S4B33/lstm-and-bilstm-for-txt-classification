# Comparison of This Study with Existing Work

Items marked **(verify)** come from an abstract or search snippet only. Check the full paper before you cite them.

## 1. Comparison table

| Aspect | Liu et al. 2024 (arXiv 2409.15576) | Query-Value Interaction (arXiv 2010.03766) | RSS-LSTM (IJACSA 2024) | **This study** |
|---|---|---|---|---|
| Goal | Bi-LSTM + attention for news classification | New attention variant (QVI) that models query-value interaction | Metaheuristic (Ringed Seal Search) hyperparameter tuning for LSTM | Controlled comparison of attention *types* in LSTM and BiLSTM |
| Datasets | Not stated in abstract (verify) | AG News, Amazon Electronics, SIGHAN Bakeoff-3/4 (NER) | Reuters-21578, 20 Newsgroups, AG News | AG News |
| Encoder | BiLSTM | CNN, LSTM, HAN, Transformer | LSTM | LSTM and BiLSTM |
| Attention types compared | One (type not specified in abstract) (verify) | Additive vs. additive + QVI (also dot-product for Transformer) | None | Additive, multiplicative, self-attention |
| Uni- vs bi-directional ablation | No | No | No | **Yes (2x4 design)** |
| Plain baselines in the same table | "Common models" (unnamed in abstract) (verify) | Yes (CNN, LSTM, HAN, Transformer) | Compared against other optimizers | Yes (LSTM, BiLSTM) |
| Reported AG News accuracy | Not given in abstract (verify) | LSTM 91.66, LSTM-Att 92.20, LSTM-Att-QVI 92.68 | About 91% (verify) | To be filled in |
| Multiple seeds / mean +/- std | Not in abstract (verify) | Yes (mean +/- std) | Not verified | **Yes (planned, 2-3 seeds)** |
| Significance test | Not in abstract (verify) | Not seen in the table | Not verified | **Planned (paired t-test / McNemar)** |
| Interpretability analysis | Not in abstract | No | No | **Planned (heatmaps, word-removal test)** |
| Efficiency reporting (params, time) | Claims speed gains, no figures in abstract | Not seen | No | **Yes (params, time/epoch, inference time)** |

## 2. Where this study differs

1. **Attention types are the variable.** Existing work either adds one attention layer to a model or proposes a new variant. None of the papers above compares additive, multiplicative and self-attention against each other under identical settings.
2. **Direction is varied too.** The 2x4 design (LSTM/BiLSTM x none/additive/multiplicative/self-attention) shows whether attention helps unidirectional and bidirectional encoders differently.
3. **Controlled setup.** Same split, preprocessing, hidden size, optimizer and early stopping for all 8 models, with multiple seeds.
4. **Cost and interpretability are reported, not just accuracy.** Parameters, training time, attention heatmaps and a word-removal check.

## 3. Reference points from prior work

- The QVI paper reports LSTM at 91.66%, LSTM + additive attention at 92.20%, and LSTM + attention + QVI at 92.68% on AG News (full training set, mean +/- std). This is your closest numeric reference for the LSTM and LSTM + additive rows.
- Because you train on a subset with a small hidden size (CPU setup), expect your numbers to land somewhat below these. State this clearly and do not claim to beat them.
- Compare trends, not absolute numbers: does attention help over the plain LSTM, and by how much?

## 4. Draft paragraph for Related Work

> Several studies have applied attention to LSTM-based news classification. Liu et al. [1] combine a Bi-LSTM with an attention mechanism for news text classification, while Zhang et al. [2] (QVI) propose a query-value interaction attention and evaluate LSTM, CNN, HAN and Transformer encoders with additive attention on AG News, reporting about 92% accuracy for attention-augmented LSTMs. Other work improves LSTM classifiers through hyperparameter optimization, for example RSS-LSTM [3]. These studies typically add a single attention layer or propose a new variant. In contrast, we systematically compare three standard attention mechanisms (additive, multiplicative and self-attention) over both unidirectional and bidirectional LSTM encoders under identical training conditions, and we report parameter counts, training time and attention-based interpretability alongside accuracy.

**Before using this paragraph:** confirm each paper's author list and citation details from the originals. I have not verified author names for the QVI paper or the RSS-LSTM paper, so replace "Zhang et al." with the correct first author.

## 5. Suggested table for the paper

Add a column for "this study" in your results section, listing the LSTM and LSTM + additive numbers next to the QVI paper's reported numbers, with a note on the differences in training data size, hidden size and embeddings.
