# Historical Sentiment CSV Import

`src/data/ingest_sentiment_csv.py` normalizes a user-provided historical news or sentiment CSV into the same schema used by the Yahoo latest-news probe and the existing sentiment feature merge.

This is intended for a licensed news/disclosure export or another source with legitimate historical coverage. It does not create historical news data.

## Supported Input

Default columns:

| column | required | note |
| --- | --- | --- |
| `date` | yes | News date or score date. |
| `ticker` | yes | Stock ticker such as `PTT.BK`. |
| `published_at` | no | Timestamp. If absent, `date` is used. |
| `title` | no | Used to build `text` when `text` is absent. |
| `summary` | no | Used to build `text` when `text` is absent. |
| `text` | no | Main article, headline, or disclosure text. |
| `sentiment_score` | no | Numeric score. If absent, the current deterministic placeholder scorer can be used for the daily table. |
| `publisher` | no | Source/publisher label. |
| `url` | no | Source URL or identifier. |

Each row needs either text/title/summary content or a numeric `sentiment_score`.

## Commands

Normalize event-level historical news:

```bash
python -m src.data.ingest_sentiment_csv \
  --config config/real_ohlcv_sentiment.yaml \
  --input path/to/historical_news.csv \
  --output data/raw/news_historical.csv
```

Normalize the news and also write daily ticker sentiment:

```bash
python -m src.data.ingest_sentiment_csv \
  --config config/real_ohlcv_sentiment.yaml \
  --input path/to/historical_news.csv \
  --output data/raw/news_historical.csv \
  --daily-output data/processed/sentiment_daily.parquet
```

Then rebuild the sentiment feature table:

```bash
python -m src.features.sentiment_context --config config/real_ohlcv_sentiment.yaml
```

Column names can be overridden with options such as `--date-column`, `--ticker-column`, `--text-column`, and `--sentiment-column`.

## Output Schema

Event-level output:

```text
date,published_at,ticker,title,summary,publisher,url,text,source,sentiment_score
```

Optional daily output:

```text
date,ticker,sentiment_score,news_count,sentiment_source
```

The feature merge remains backward-only with `sentiment.max_age_days`, so future news cannot leak into earlier trading dates.

## Limitation

If the input has no `sentiment_score`, the daily output uses the existing deterministic placeholder scorer from `src.sentiment.extract_embeddings`. Replace that with a real NLP model or licensed sentiment field before interpreting sentiment results.
