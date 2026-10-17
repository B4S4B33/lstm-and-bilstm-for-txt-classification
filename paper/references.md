# References and citation notes

Full metadata for the sources cited in `paper.md`, plus verification notes. Verify every entry against the original before submission (see `plagiarism_guide.md` §6.8).

## Core references cited in the paper

1. **Hochreiter & Schmidhuber, 1997** — Long Short-Term Memory. *Neural Computation* 9(8):1735-1780. DOI: 10.1162/neco.1997.9.8.1735. (The LSTM.)

2. **Bahdanau, Cho & Bengio, 2015** — Neural Machine Translation by Jointly Learning to Align and Translate. ICLR 2015. arXiv:1409.0473. (Additive attention, cited in §3.4.)

3. **Luong, Pham & Manning, 2015** — Effective Approaches to Attention-based Neural Machine Translation. EMNLP 2015. arXiv:1508.04025. (Multiplicative attention, cited in §3.4.)

4. **Vaswani et al., 2017** — Attention Is All You Need. NeurIPS 2017. arXiv:1706.03762. (Self-attention / Transformer, cited in §3.4.)

5. **Zhang, Zhao & LeCun, 2015** — Character-level Convolutional Networks for Text Classification. NeurIPS 2015. arXiv:1509.01626. (Source of the AG News classification dataset: 120,000 train / 7,600 test, 4 classes.)

6. **Wu, Wu, Qi & Huang, 2020** — Improving Attention Mechanism with Query-Value Interaction. arXiv:2010.03766. Verified from the arXiv abstract. Reports on AG News: LSTM 91.66, LSTM+additive attention 92.20, LSTM+attention+QVI 92.68 (full training set, mean ± std).

7. **Liu, Chen, Wang, Huang, Luo & Wei, 2024** — Optimizing News Text Classification with Bi-LSTM and Attention Mechanism for Efficient Data Processing. arXiv:2409.15576. Verified from the arXiv abstract (Bi-LSTM + attention for news classification).

## Verification notes (important)

- **[6] author correction.** The working comparison file (`comparison_with_existing_work.md`) guessed the query-value interaction paper was "Zhang et al." It is actually **Wu, Wu, Qi & Huang**. Cite it as Wu et al. This was confirmed from the arXiv page.
- **[7] confirmed.** First author Bingyao Liu; "Liu et al." is correct.
- **Metaheuristic-tuned LSTM ("RSS-LSTM", IJACSA 2024).** The comparison table references this as reporting roughly 91% on AG News. I could not verify a full citation (authors, exact title, volume) from search results, so it is *not* given a numbered reference in the paper; it is flagged in §6 as "citation to be confirmed before submission." Confirm the full citation and add it as a numbered reference before submitting, or drop the mention.
- **Read-before-cite.** The claims drawn from [2]-[7] rest on their abstracts and the specific numbers marked as such; read the full papers before final submission if the venue requires it.

## Numeric consistency (see `../output/`)

All numbers in `paper.md` come from `../output/bilstm_results.jsonl`, `../output/lstm_results.jsonl`, the two summary files, and `../output/figures/significance_tests.txt`. Accuracy and macro-F1 are reported as mean ± one **sample** standard deviation (ddof=1) over seeds. If the results are re-run, regenerate the figures with `plots.py`/`analysis.py` and re-check the tables in `paper.md` and the numbers in `figures/significance_tests.txt`.

## Software / hardware (Colab CPU)

Unidirectional family (logged by the run): PyTorch 2.11.0+cpu, NumPy 2.1.3, pandas 2.2.3. Metrics via scikit-learn, plots via matplotlib. Re-confirm the exact versions of the bidirectional run from its logs before submission.
