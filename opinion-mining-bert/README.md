# BERT Universal Opinion Mining (Folderized)

This folder re-implements the original notebook as a clean Python codebase for
multi-task opinion mining with BERT. It trains a single model to predict:

1. **Type** (multi-label): `agreement`, `arguing`, `expressive_subjectivity`,
   `intention`, `sentiment`
2. **Polarity** (multi-class): `negative`, `neutral`, `positive`
3. **Intensity** (3-bit encoding across 5 classes)

The code follows the same logic as the notebook: it builds a 35-dimension target
vector per (text, head) pair, fine-tunes a BERT encoder, and evaluates multiple
metrics across cross-validation folds.

## Structure

```
opinion-mining-bert/
  bert_universal/
    config.py     # experiment settings
    data.py       # dataset loading & preprocessing
    model.py      # BERT + custom head
    metrics.py    # loss + evaluation metrics
    train.py      # training loop entrypoint
  requirements.txt
  README.md
```

## Quick start

```bash
cd opinion-mining-bert
python -m bert_universal.train \
  --data-url "https://..." \
  --splits-url "https://raw.githubusercontent.com/theSaeed/opinion-mining-using-llms/master/dataset/folds/tpi-folds.json" \
  --model-name "bert-base-uncased"
```

If you want to use a local dataset instead of a URL, pass `--file-address`.

## Notes
- The training loop uses Hugging Face `Trainer`.
- The dataset format is expected to match the MPQA2.0 JSON schema used in the
  original notebook.
