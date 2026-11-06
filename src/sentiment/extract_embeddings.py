from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import ensure_dirs


def _hash_sentiment(text: str) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    code_sum = sum(ord(ch) for ch in text)
    return float(((code_sum % 200) - 100) / 100.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="airesearch/wangchanberta-base-att-spm-uncased")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    ensure_dirs(output_path.parent)

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} does not exist. Create a cleaned news table with date, ticker, and text columns first."
        )

    news = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    required = {"date", "ticker", "text"}
    missing = required - set(news.columns)
    if missing:
        raise ValueError(f"news input is missing columns: {sorted(missing)}")

    # Placeholder deterministic score keeps the pipeline runnable. Replace with
    # transformer embeddings after the GPU NLP environment is installed.
    daily = (
        news.assign(sentiment_score=news["text"].map(_hash_sentiment))
        .groupby(["date", "ticker"], as_index=False)
        .agg(sentiment_score=("sentiment_score", "mean"), news_count=("text", "size"))
    )
    daily["embedding_model"] = args.model
    daily["embedding_dim"] = np.int64(1)
    daily.to_parquet(output_path, index=False)
    print(f"wrote {output_path} rows={len(daily)}")


if __name__ == "__main__":
    main()

