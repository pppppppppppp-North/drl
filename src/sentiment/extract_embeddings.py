"""Daily news sentiment scores and text embeddings for the trading feature table.

Scorers:

- ``classifier`` (default): a Hugging Face text-classification model with sentiment labels. Each
  article's score is ``P(positive) - P(negative)`` (neutral counts as 0), then averaged per
  ``date, ticker``. The model's labels must be recognizable sentiment labels, or ``--label-map``
  must say which label is positive/negative/neutral. A base language model without a trained
  sentiment head (for example ``airesearch/wangchanberta-base-att-spm-uncased``) is rejected,
  because its randomly initialized head would produce meaningless scores.
- ``embedding``: mean-pooled hidden states from a language model such as WangchanBERTa, reduced to
  ``--embedding-dims`` principal components fitted only on rows dated on or before
  ``--fit-end-date`` (no validation/test leakage). Output columns are ``sentiment_emb_0..k``.
- ``placeholder``: the old deterministic character-code hash. It carries no sentiment information
  and exists only for pipeline smoke tests; outputs are labeled ``PLACEHOLDER_HASH``.

Run on BistKA GPU (``gpu4500-normal``) for real models; see ``slurm/train_sentiment_gpu.sbatch``.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import ensure_dirs

PLACEHOLDER_MODEL = "PLACEHOLDER_HASH"
POSITIVE_NAMES = {"positive", "pos", "bullish", "label_positive"}
NEGATIVE_NAMES = {"negative", "neg", "bearish", "label_negative"}
NEUTRAL_NAMES = {"neutral", "neu", "label_neutral", "q"}


def _hash_sentiment(text: str) -> float:
    """Deterministic placeholder; NOT a sentiment model. Kept for smoke tests and legacy imports."""

    if not isinstance(text, str) or not text:
        return 0.0
    code_sum = sum(ord(ch) for ch in text)
    return float(((code_sum % 200) - 100) / 100.0)


def resolve_label_values(labels: Sequence[str], label_map: dict[str, float] | None = None) -> dict[str, float]:
    """Map model labels to +1 / -1 / 0 values, failing closed on unrecognized label sets."""

    if label_map:
        unknown = set(label_map) - set(labels)
        if unknown:
            raise ValueError(f"--label-map names labels the model does not have: {sorted(unknown)}")
        return {label: float(label_map.get(label, 0.0)) for label in labels}
    values: dict[str, float] = {}
    for label in labels:
        name = str(label).strip().lower()
        if name in POSITIVE_NAMES:
            values[label] = 1.0
        elif name in NEGATIVE_NAMES:
            values[label] = -1.0
        elif name in NEUTRAL_NAMES:
            values[label] = 0.0
    if not any(v > 0 for v in values.values()) or not any(v < 0 for v in values.values()):
        raise ValueError(
            f"model labels {list(labels)} are not recognizable sentiment labels; pass --label-map "
            "(for example '{\"LABEL_0\": -1, \"LABEL_1\": 0, \"LABEL_2\": 1}') or use a model fine-tuned "
            "for sentiment"
        )
    for label in labels:
        values.setdefault(label, 0.0)
    return values


def score_with_classifier(
    texts: Sequence[str],
    classify: Callable[[list[str]], list[list[dict]]],
    label_values: dict[str, float],
    batch_size: int = 32,
) -> np.ndarray:
    """Expected sentiment value per text from a classifier returning all label probabilities."""

    scores = np.zeros(len(texts), dtype=np.float64)
    clean = [t if isinstance(t, str) else "" for t in texts]
    for start in range(0, len(clean), batch_size):
        batch = clean[start : start + batch_size]
        outputs = classify(batch)
        for offset, (text, label_scores) in enumerate(zip(batch, outputs)):
            if not text:
                continue
            if isinstance(label_scores, dict):
                label_scores = [label_scores]
            total = sum(float(item["score"]) for item in label_scores) or 1.0
            scores[start + offset] = sum(
                label_values.get(item["label"], 0.0) * float(item["score"]) for item in label_scores
            ) / total
    return scores


def fit_pca(train_matrix: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and component matrix fitted only on training rows."""

    if train_matrix.shape[0] < 2:
        raise ValueError("embedding PCA needs at least two training rows on or before --fit-end-date")
    dims = min(dims, train_matrix.shape[0], train_matrix.shape[1])
    mean = train_matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(train_matrix - mean, full_matrices=False)
    return mean, vt[:dims]


def daily_scores(news: pd.DataFrame, scores: np.ndarray, model_name: str) -> pd.DataFrame:
    daily = (
        news.assign(sentiment_score=scores)
        .groupby(["date", "ticker"], as_index=False)
        .agg(sentiment_score=("sentiment_score", "mean"), news_count=("text", "size"))
    )
    daily["embedding_model"] = model_name
    daily["embedding_dim"] = np.int64(1)
    return daily


def daily_embeddings(news: pd.DataFrame, embeddings: np.ndarray, fit_end_date: str, dims: int, model_name: str) -> pd.DataFrame:
    dates = pd.to_datetime(news["date"])
    train_mask = (dates <= pd.Timestamp(fit_end_date)).to_numpy()
    mean, components = fit_pca(embeddings[train_mask], dims)
    reduced = (embeddings - mean) @ components.T
    columns = [f"sentiment_emb_{i}" for i in range(reduced.shape[1])]
    frame = pd.concat([news[["date", "ticker"]].reset_index(drop=True), pd.DataFrame(reduced, columns=columns)], axis=1)
    frame["news_count"] = 1
    daily = frame.groupby(["date", "ticker"], as_index=False).agg(
        {**{c: "mean" for c in columns}, "news_count": "sum"}
    )
    daily["embedding_model"] = model_name
    daily["embedding_dim"] = np.int64(len(columns))
    daily["pca_fit_end_date"] = fit_end_date
    return daily


def _hf_classifier(model: str, device: int | None):  # pragma: no cover - needs model download
    from transformers import pipeline

    clf = pipeline("text-classification", model=model, tokenizer=model, top_k=None, truncation=True, device=device)
    labels = list(clf.model.config.id2label.values())
    return (lambda batch: clf(batch)), labels


def _hf_embeddings(model: str, texts: list[str], batch_size: int, device: str) -> np.ndarray:  # pragma: no cover
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    encoder = AutoModel.from_pretrained(model).to(device).eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(device)
            hidden = encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            rows.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(rows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="cleaned news table with date, ticker, text")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scorer", choices=["classifier", "embedding", "placeholder"], default="classifier")
    parser.add_argument("--model", default=None, help="Hugging Face model id or local path")
    parser.add_argument("--label-map", default=None, help="JSON mapping of model label -> sentiment value")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dims", type=int, default=8)
    parser.add_argument("--fit-end-date", default=None, help="last date used to fit embedding PCA (training end)")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    ensure_dirs(output_path.parent)
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} does not exist. Create a cleaned news table with date, ticker, and text columns first."
        )
    news = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    missing = {"date", "ticker", "text"} - set(news.columns)
    if missing:
        raise ValueError(f"news input is missing columns: {sorted(missing)}")
    texts = news["text"].fillna("").astype(str).tolist()

    if args.scorer == "placeholder":
        print("WARNING: placeholder hash scorer carries no sentiment information; use only for smoke tests")
        daily = daily_scores(news, np.array([_hash_sentiment(t) for t in texts]), PLACEHOLDER_MODEL)
    elif args.scorer == "classifier":
        if not args.model:
            raise SystemExit("--model is required for --scorer classifier (a model fine-tuned for sentiment)")
        device = None
        if args.device:
            device = 0 if args.device == "cuda" else (int(args.device.split(":")[1]) if ":" in args.device else -1)
        classify, labels = _hf_classifier(args.model, device)
        label_values = resolve_label_values(labels, json.loads(args.label_map) if args.label_map else None)
        daily = daily_scores(news, score_with_classifier(texts, classify, label_values, args.batch_size), args.model)
    else:
        if not args.model or not args.fit_end_date:
            raise SystemExit("--model and --fit-end-date are required for --scorer embedding")
        embeddings = _hf_embeddings(args.model, texts, args.batch_size, args.device or "cpu")
        daily = daily_embeddings(news, embeddings, args.fit_end_date, args.embedding_dims, args.model)

    if output_path.suffix == ".parquet":
        daily.to_parquet(output_path, index=False)
    else:
        daily.to_csv(output_path, index=False)
    print(f"wrote {output_path} rows={len(daily)} scorer={args.scorer}")


if __name__ == "__main__":
    main()
