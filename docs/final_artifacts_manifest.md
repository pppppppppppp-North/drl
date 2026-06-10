# Final Artifact Manifest

Last updated: 2026-06-09

This manifest lists the current pilot artifacts for the Thai stock DRL project on BistKA.

## Main Project Directory

```text
/lustrefs/project/25sfcs03/drl_thai_stock
```

## Progress And Writeups

| artifact | path |
| --- | --- |
| Progress checklist | `PROGRESS_CHECKLIST.md` |
| Final methodology/results writeup | `docs/final_methodology_results.md` |
| Final methodology/results LaTeX report | `final_methodology_results.tex` |
| Comprehensive introduction/literature-review source | `comprehensive_introduction_literature_review.tex` |
| Comprehensive introduction/literature-review PDF | `comprehensive_introduction_literature_review.pdf` |
| Comprehensive 100-frame Beamer presentation source | `final_project_beamer_100_pages.tex` |
| Comprehensive 100-frame Beamer presentation PDF | `final_project_beamer_100_pages.pdf` |
| Data source notes | `docs/data_sources.md` |
| External data handoff guide | `docs/external_data_handoff.md` |
| External data provider request packet | `docs/external_data_provider_request.md` |
| External data intake checklist | `docs/external_data_intake_checklist.md` |
| External data manifest template | `data/reference/external_data_manifest_template.csv` |
| External data intake validation CLI | `src/data/intake_validation.py` |
| Full external-data rebuild guide | `docs/full_external_rebuild.md` |
| BistKA sync status and verified handoff | `hpc/hpc_sync_status.md` |
| External source-readiness report | `reports/source_readiness.csv` |
| External intake validation CSV report | `reports/external_intake_validation.csv` |
| External intake validation Markdown report | `reports/external_intake_validation.md` |
| Real OHLCV data-quality report | `reports/data_quality/real_ohlcv_data_quality.md` |
| Ticker coverage CSV | `reports/data_quality/ticker_coverage.csv` |
| Column missingness CSV | `reports/data_quality/column_missingness.csv` |
| Walk-forward summary CSV | `results/walk_forward_summary_check/walk_forward_summary.csv` |
| Walk-forward pilot policy summary | `results/walk_forward_pilot_2869/aggregate/walk_forward_policy_summary.csv` |
| Walk-forward pilot all metrics | `results/walk_forward_pilot_2869/aggregate/walk_forward_all_metrics.csv` |
| Walk-forward pilot test equity curves | `results/walk_forward_pilot_2869/aggregate/walk_forward_test_equity_curves.csv` |
| Ablation pilot metrics | `results/ablation_pilot_2871/aggregate/ablation_metrics.csv` |
| Ablation pilot PPO summary | `results/ablation_pilot_2871/aggregate/ablation_ppo_summary.csv` |
| Ablation pilot bar chart | `results/ablation_pilot_2871/aggregate/ablation_ppo_sharpe.png` |
| Regime/stress-test slices | `results/regime_tests_real_ohlcv_20260531/regime_slices.csv` |
| Regime/stress-test metrics | `results/regime_tests_real_ohlcv_20260531/regime_metrics.csv` |
| Regime/stress-test summary | `results/regime_tests_real_ohlcv_20260531/regime_summary.md` |
| SET index raw prices | `data/raw/prices_market_indices.csv` |
| Real OHLCV + SET market-context features | `data/processed/features_real_ohlcv_market.parquet` |
| SET market-context data-quality report | `reports/data_quality_market/real_ohlcv_data_quality.md` |
| SET market-context pilot comparison | `results/real_ohlcv_market_pilot_20260531/comparison_metrics.csv` |
| Pilot sector mapping | `data/reference/sector_mapping_thai_pilot.csv` |
| Sector-index source probe note | `docs/sector_index_probe.md` |
| Sector-index source probe raw output | `data/raw/sector_index_yahoo_probe.csv` |
| Usable sector-index probe prices | `data/raw/prices_sector_indices_yahoo.csv` |
| Real OHLCV + SET + sector-context features | `data/processed/features_real_ohlcv_sector.parquet` |
| Sector-context data-quality report | `reports/data_quality_sector/real_ohlcv_data_quality.md` |
| Sector-context pilot comparison | `results/real_ohlcv_sector_pilot_20260531/comparison_metrics.csv` |
| Yahoo macro proxy raw prices | `data/raw/prices_macro_yahoo.csv` |
| Real OHLCV + SET + sector + macro-proxy features | `data/processed/features_real_ohlcv_macro.parquet` |
| Macro-proxy data-quality report | `reports/data_quality_macro/real_ohlcv_data_quality.md` |
| Macro-proxy pilot comparison | `results/real_ohlcv_macro_pilot_20260531/comparison_metrics.csv` |
| BOT official macro source probe | `data/raw/bot_official_macro.csv` |
| Official macro historical CSV importer | `src/data/ingest_official_macro_csv.py` |
| Official macro historical CSV importer notes | `docs/official_macro_csv_import.md` |
| Real OHLCV + SET + sector + macro + fundamentals + official macro scaffold features | `data/processed/features_real_ohlcv_official_macro.parquet` |
| Official macro scaffold data-quality report | `reports/data_quality_official_macro/real_ohlcv_data_quality.md` |
| Starter ticker universe CSV | `data/reference/thai_starter_universe.csv` |
| Ticker universe CSV config example | `config/real_ohlcv_universe_csv.yaml` |
| Ticker universe CSV notes | `docs/ticker_universe_csv.md` |
| Yahoo annual/quarterly fundamentals | `data/raw/fundamentals_yahoo_quarterly.csv` |
| Real OHLCV + SET + sector + macro + fundamentals features | `data/processed/features_real_ohlcv_fundamentals.parquet` |
| Fundamentals data-quality report | `reports/data_quality_fundamentals/real_ohlcv_data_quality.md` |
| Fundamentals pilot comparison | `results/real_ohlcv_fundamentals_pilot_20260531/comparison_metrics.csv` |
| Yahoo latest-news source probe | `data/raw/news_yahoo_latest.csv` |
| Historical sentiment CSV importer | `src/data/ingest_sentiment_csv.py` |
| Historical sentiment CSV importer notes | `docs/sentiment_csv_import.md` |
| Daily sentiment scaffold table | `data/processed/sentiment_daily.parquet` |
| Real OHLCV + SET + sector + macro + fundamentals + sentiment scaffold features | `data/processed/features_real_ohlcv_sentiment.parquet` |
| Sentiment scaffold data-quality report | `reports/data_quality_sentiment/real_ohlcv_data_quality.md` |

## Configs

| artifact | path |
| --- | --- |
| Real OHLCV comparison config | `config/real_ohlcv.yaml` |
| Real OHLCV CSV-universe config | `config/real_ohlcv_universe_csv.yaml` |
| Optuna search config | `config/optuna_search.yaml` |
| Best PPO real-OHLCV config | `config/best_ppo_real_ohlcv.yaml` |
| Synthetic technical config | `config/ppo_technical.yaml` |
| Real OHLCV returns ablation config | `config/ablation_returns_real_ohlcv.yaml` |
| Real OHLCV technical ablation config | `config/ablation_technical_real_ohlcv.yaml` |
| Real OHLCV technical+context ablation config | `config/ablation_technical_context_real_ohlcv.yaml` |
| Real OHLCV regime/stress-test config | `config/regime_tests_real_ohlcv.yaml` |
| Real OHLCV + SET market-context config | `config/real_ohlcv_market.yaml` |
| Real OHLCV + SET + sector-context config | `config/real_ohlcv_sector.yaml` |
| Real OHLCV + SET + sector + macro-proxy config | `config/real_ohlcv_macro.yaml` |
| Real OHLCV + SET + sector + macro + fundamentals + official macro scaffold config | `config/real_ohlcv_official_macro.yaml` |
| Real OHLCV + SET + sector + macro + fundamentals config | `config/real_ohlcv_fundamentals.yaml` |
| Real OHLCV + SET + sector + macro + fundamentals + sentiment scaffold config | `config/real_ohlcv_sentiment.yaml` |
| Full external-data config | `config/real_ohlcv_full_external.yaml` |

## Verified Result Directories

| purpose | path |
| --- | --- |
| Synthetic split-aware comparison | `results/algo_compare_2853` |
| Real OHLCV PPO/A2C comparison | `results/real_ohlcv_compare_2855` |
| Optuna sanity run | `results/optuna_sanity_2856` |
| Fixed Optuna search | `results/optuna_search_fixed_2863` |
| Longer best-parameter PPO validation | `results/best_ppo_real_ohlcv_2868` |
| Walk-forward pilot | `results/walk_forward_pilot_2869` |
| Ablation pilot | `results/ablation_pilot_2871` |
| Regime/stress-test pilot | `results/regime_tests_real_ohlcv_20260531` |
| SET market-context pilot | `results/real_ohlcv_market_pilot_20260531` |
| Sector-context pilot | `results/real_ohlcv_sector_pilot_20260531` |
| Macro-proxy pilot | `results/real_ohlcv_macro_pilot_20260531` |
| Fundamentals pilot | `results/real_ohlcv_fundamentals_pilot_20260531` |

## Invalidated Result Directories

| path | reason |
| --- | --- |
| `results/real_ohlcv_compare_2854` | Old training entrypoint regenerated synthetic data into real config paths. |
| `results/optuna_search_2858` | Pre-fix sampler repeated the same hyperparameters across repeated algorithm trials. |

## Archive

The latest final pilot archive is generated on BistKA at:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_set_context.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_sector_context.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_macro_context.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_fundamentals.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_sentiment_scaffold.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_official_macro_scaffold.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_sector_index_probe.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_official_macro_csv_import.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_sentiment_csv_import.tar.gz
/lustrefs/project/25sfcs03/drl_thai_stock/final_archive/20260531_final_project_pilot_with_ticker_universe_csv.tar.gz
```

The latest archive intentionally contains configs, report files, source code, tests, real OHLCV data, SET index data, sector-index Yahoo probe output, usable banking sector-index probe data, sector mapping, ticker-universe CSV support, macro proxy data, BOT official macro source probe data, the official macro historical CSV importer, the historical sentiment CSV importer, Yahoo statement fundamentals, Yahoo latest-news probe data, market-context, sector-context, macro-proxy, fundamentals, sentiment-scaffold, and official-macro-scaffold features, data-quality outputs, metric CSVs, summaries, selected figures, and relevant logs. It does not attempt to package every generated training cache, model binary, or TensorBoard event file.

Latest file-separation snapshot:

```text
/lustrefs/project/25sfcs03/drl_thai_stock/_file_separation/20260601_100216
```
