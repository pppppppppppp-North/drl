# Ticker Universe CSV

Real-source ingesters can now read ticker symbols from a CSV or text file instead of requiring every ticker to be hard-coded in YAML.

## Config

Use `data.ticker_universe` in a real-data config:

```yaml
data:
  source: yahoo_ohlcv
  ticker_universe:
    path: data/reference/thai_starter_universe.csv
    column: ticker
  start: "2021-01-01"
  end: "2024-12-31"
```

If the source file contains bare SET symbols such as `PTT`, set `suffix: .BK`:

```yaml
data:
  ticker_universe:
    path: data/reference/set50_symbols.csv
    column: symbol
    suffix: .BK
```

The resolver normalizes symbols to uppercase, removes duplicates while preserving first occurrence, and rejects empty universes.

## Current Starter File

The current pilot universe is:

```text
data/reference/thai_starter_universe.csv
```

It contains the same five `.BK` tickers used by the existing pilot results. It is a wiring artifact, not a claim that the universe is complete.

## Supported Commands

The resolver is wired into:

```bash
python -m src.data.ingest_ohlcv --config config/real_ohlcv_universe_csv.yaml
python -m src.data.ingest_fundamentals --config config/real_ohlcv_universe_csv.yaml
python -m src.data.ingest_news_yahoo --config config/real_ohlcv_universe_csv.yaml
```

Training feature checks also use the resolver when a config points at a ticker universe file.
