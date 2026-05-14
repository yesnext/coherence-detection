# Coherence Checking Model Comparison Plan

## Current Baseline

The current baseline is a fine-tuned `bert-base-uncased` binary classifier trained to distinguish coherent ROCStories from shuffled sentence-order negatives.

Current reported held-out test result from the notebook:

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| BERT fine-tuned baseline | 0.9409 | 0.9414 | 0.9823 |

Important correction: the split should happen before negative sample generation. If shuffled versions of the same original story appear across train, validation, and test, the evaluation can become too optimistic. The new experiment script fixes this by splitting raw stories first.

## Model Variants

Use one fixed train/validation/test split for all models.

### Non-neural Baselines

These establish how much performance comes from simple lexical patterns:

| Variant | Purpose |
|---|---|
| Majority class dummy | Minimum baseline |
| Stratified random dummy | Random baseline with class balance |
| Word TF-IDF + Logistic Regression | Strong interpretable classical baseline |
| Character TF-IDF + Logistic Regression | Captures surface and punctuation/order patterns |
| Word TF-IDF + Linear SVM | Strong sparse-text classifier baseline |

### Transformer Variants

These test whether stronger pretrained encoders improve coherence detection:

| Variant | Expected Role |
|---|---|
| DistilBERT | Smaller/faster transformer |
| BERT base | Current baseline |
| RoBERTa base | Stronger masked-language-model pretraining |
| DeBERTa v3 base | Strong contextual encoder, often competitive |

### Negative-Sampling Variants

The task difficulty changes depending on how incoherent examples are created:

| Negative Type | Description |
|---|---|
| Full shuffle | Random sentence permutation |
| Adjacent swap | Harder negative; only two neighboring sentences are swapped |
| Reverse order | Strong temporal disruption |
| Sentence replacement | Replace one sentence with a sentence from another story |

For a strong academic comparison, train/evaluate the best model family under at least two negative strategies: `shuffle` and `adjacent_swap`.

## Required Measurements

Report these for every model on validation and final test:

| Metric | Why It Matters |
|---|---|
| Accuracy | Overall correctness |
| Precision | How reliable coherent predictions are |
| Recall | How many coherent stories are recovered |
| F1-score | Balanced precision/recall comparison |
| ROC-AUC | Ranking quality across thresholds |
| PR-AUC | Useful when class balance changes |
| MCC | Robust single-number binary classification metric |
| Confusion matrix | Shows false coherent vs false incoherent errors |

Use validation F1 to choose the best model. Report the test set only once for the selected model and all comparison models.

## Statistical Comparison

For the final selected model versus the current BERT baseline:

1. Use the exact same held-out test stories.
2. Compare paired predictions with McNemar's test.
3. Add 95% confidence intervals for accuracy/F1 using bootstrap resampling.
4. Include training time and inference time per 1,000 stories if possible.

## How To Run The Added Baselines

Smoke test:

```powershell
python coherence_experiments.py --quick
```

Full classical baseline comparison:

```powershell
python coherence_experiments.py
```

Harder adjacent-swap comparison:

```powershell
python coherence_experiments.py --negative-strategy adjacent_swap --output-dir outputs/experiments_adjacent_swap
```

The script saves:

| Output | Meaning |
|---|---|
| `outputs/experiments/splits/*.csv` | Fixed leak-free train/val/test data |
| `outputs/experiments/classical_model_comparison.csv` | Main comparison table |
| `outputs/experiments/*/test_confusion_matrix.csv` | Per-model confusion matrix |
| `outputs/experiments/*/test_classification_report.txt` | Per-model precision/recall/F1 report |

## Suggested Thesis Structure

1. Dataset: ROCStories, 52,665 five-sentence stories.
2. Task formulation: binary coherence classification.
3. Negative construction: original order as coherent, perturbed order as incoherent.
4. Leakage prevention: split raw stories before generating negatives.
5. Baselines: dummy, TF-IDF logistic regression, TF-IDF SVM.
6. Neural models: DistilBERT, BERT, RoBERTa, DeBERTa.
7. Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, MCC, confusion matrix.
8. Model selection: validation F1.
9. Final comparison: held-out test results and statistical significance.
