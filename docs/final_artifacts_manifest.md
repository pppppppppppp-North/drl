# Final Artifact Manifest

Last updated: 2026-09-07

This manifest lists the current pilot artifacts for the Thai stock DRL project on BistKA.

> **Preserved V2 state; local V3 evidence through 2026-09-06:** canonical whole-project completion remains **11/37 = 29.7%**. The active consolidated graph is **9 VERIFIED, 1 READY (`screening`), and 25 PENDING** across 35 stages; the smaller graph is structural because one lineage gate consolidates three former gates. Storage-safe release `56358f2a4996f654426814d88e622b7f217ac89bfc02f8bbdacc27e66873b55e` remains unchanged. The isolated V3 capsule includes an exact 540-cell blocked plan, a gate-first launcher with no V3 runner, canonical fold dataset slices, train-only preprocessing, and transition-result reconciliation. Its six provisional folds, 30 configurations, and seeds 42/43/44 total 63,000,000 planned timesteps in three batches of 180 cells. Eventual execution is capped at three array submissions and two concurrent jobs; zero V3 jobs have been submitted. Training, `test`, `all`, ranking, and performance claims remain blocked pending licensed market/calendar evidence, seven human policy decisions, approved locked policies/protocol, a reviewed separate runner, and a genuinely untouched post-2024 holdout. Aggregate metric conversion remains blocked by the missing approved Sharpe/turnover/drawdown contract. The SETSMART packet remains **DRAFT — UNSENT**. No scientific-compute stage or campaign state changed, and this local update makes no new remote-state claim. The V1 entries below remain historical evidence.

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
| Comprehensive Beamer presentation source | `final_project_beamer_100_pages.tex` |
| Comprehensive Beamer presentation PDF | `final_project_beamer_100_pages.pdf` |
| Data source notes | `docs/data_sources.md` |
| External data handoff guide | `docs/external_data_handoff.md` |
| External data provider request packet | `docs/external_data_provider_request.md` |
| External data intake checklist | `docs/external_data_intake_checklist.md` |
| External data manifest template | `data/reference/external_data_manifest_template.csv` |
| External data intake validation CLI | `src/data/intake_validation.py` |
| Full external-data rebuild guide | `docs/full_external_rebuild.md` |
| BistKA sync status and verified handoff | `hpc/hpc_sync_status.md` |
| External source-readiness report | `reports/source_readiness.csv` |
| Official SET constituent-history intake guide | `docs/set_constituent_history_intake.md` |
| Official SET constituent source manifest | `data/external/set_constituent_source_manifest.csv` |
| Preserved official SET constituent PDFs | `data/external/original/set_constituent_history/` |
| Point-in-time SET50/SET100 membership, 1,000 rows | `data/reference/set50_set100_membership_2021_2024.csv` |
| Acquisition-only 138-ticker union | `data/reference/set50_or_set100_universe.csv` |
| Latest-label sector compatibility map | `data/reference/sector_mapping.csv` |
| SET archive checksum-bound receipt and sidecar | `reports/data_sources/set_constituent_archive_20260731.json` and `reports/data_sources/set_constituent_archive_20260731.json.sha256` |
| SET archive interval-readiness CSV | `reports/set_constituent_archive_readiness.csv` |
| SET archive readiness report and sidecar | `reports/set_constituent_archive_readiness_20260731.md` and `.sha256` |
| SET constituent turnover visualization and sidecar | `reports/figures/set_constituent_turnover_2021_2024.svg` and `.sha256` |
| Isolated V3 source package | `src/v3/` |
| V3 draft development protocol | `config/v3/development_protocol.yaml` |
| V3 fixed-slot SET100 membership panel and sidecar | `data/processed/v3_development/set100_membership_session_mask_2021_2024.csv.gz` and `.sha256` |
| V3 session-record summary and sidecar | `reports/data_quality_v3/membership_session_summary.csv` and `.sha256` |
| V3 readiness receipt and sidecar | `reports/data_quality_v3/v3_membership_mask_readiness_20260731.json` and `.sha256` |
| V3 technical readiness report and sidecar | `reports/v3_membership_mask_readiness_20260731.md` and `.sha256` |
| V3 licensed-market manifest template | `config/v3/market_package_manifest_template.yaml` |
| V3 licensed-market package validator | `src/v3/market_data_readiness.py` |
| V3 active-member dated sector context | `src/v3/sector_context.py` |
| V3 self-financing execution ledger | `src/v3/execution_ledger.py` |
| V3 investability lifecycle and exit-path validator | `src/v3/investability_policy.py` |
| V3 checksum-bound draft investability policy and sidecar | `config/v3/investability_policy_draft.yaml` and `.sha256` |
| V3 market-package gap receipt and sidecar | `reports/data_quality_v3/v3_market_package_gap_20260731.json` and `.sha256` |
| V3 market-package checks and sidecar | `reports/data_quality_v3/v3_market_package_checks.csv` and `.sha256` |
| V3 market-package coverage table and sidecar | `reports/data_quality_v3/v3_market_package_coverage.csv` and `.sha256` |
| V3 market-package coverage chart and sidecar | `reports/figures/v3_market_package_coverage_20260731.png` and `.sha256` |
| V3 market-package technical report and sidecar | `reports/v3_market_package_gap_20260731.md` and `.sha256` |
| V3 market-package reproducible companion notebook | `notebooks/v3_market_package_readiness.ipynb` |
| V3 membership-exit path inventory and sidecar | `reports/data_quality_v3/v3_membership_exit_path_inventory_20260731.csv` and `.sha256` |
| V3 investability-policy receipt and sidecar | `reports/data_quality_v3/v3_investability_policy_readiness_20260731.json` and `.sha256` |
| V3 membership-exit chart and sidecar | `reports/figures/v3_membership_exit_transitions_20260731.png` and `.sha256` |
| V3 investability-policy technical report and sidecar | `reports/v3_investability_policy_readiness_20260731.md` and `.sha256` |
| V3 evaluation-policy implementation | `src/v3/evaluation_policy.py` |
| V3 checksum-bound draft evaluation policy and sidecar | `config/v3/evaluation_policy_draft.yaml` and `.sha256` |
| V3 provisional evaluation fold schedule and sidecar | `reports/data_quality_v3/v3_evaluation_fold_schedule_20260820.csv` and `.sha256` |
| V3 provisional fold summary and sidecar | `reports/data_quality_v3/v3_evaluation_fold_summary_20260820.csv` and `.sha256` |
| V3 fixed 30-configuration candidate ledger and sidecar | `reports/data_quality_v3/v3_candidate_ledger_20260820.csv` and `.sha256` |
| V3 evaluation-policy readiness receipt and sidecar | `reports/data_quality_v3/v3_evaluation_policy_readiness_20260820.json` and `.sha256` |
| V3 evaluation fold-schedule visualization and sidecar | `reports/figures/v3_evaluation_fold_schedule_20260820.png` and `.sha256` |
| V3 evaluation-policy technical report and sidecar | `reports/v3_evaluation_policy_readiness_20260820.md` and `.sha256` |
| V3 deterministic blocked execution-plan implementation | [`src/v3/execution_plan.py`](../src/v3/execution_plan.py) |
| V3 gate-first candidate/fold/seed launcher | [`src/v3/trial_launcher.py`](../src/v3/trial_launcher.py) |
| V3 canonical fold dataset and transition-ledger materializer | [`src/v3/fold_dataset.py`](../src/v3/fold_dataset.py) |
| V3 eligible-training-decision-only preprocessing | [`src/v3/fold_preprocessing.py`](../src/v3/fold_preprocessing.py) |
| V3 transition-result reconciliation and guarded metric conversion | [`src/v3/result_evidence.py`](../src/v3/result_evidence.py) |
| V3 execution-capsule builder, no execution authority | [`scripts/build_v3_execution_capsule.py`](../scripts/build_v3_execution_capsule.py) |
| V3 exact 540-cell blocked execution plan and sidecar | [`reports/data_quality_v3/v3_execution_plan_20260907.csv`](../reports/data_quality_v3/v3_execution_plan_20260907.csv) and [`.sha256`](../reports/data_quality_v3/v3_execution_plan_20260907.csv.sha256) |
| V3 six-fold workload, 719 provisional validation transitions, and sidecar | [`reports/data_quality_v3/v3_execution_plan_fold_workload_20260907.csv`](../reports/data_quality_v3/v3_execution_plan_fold_workload_20260907.csv) and [`.sha256`](../reports/data_quality_v3/v3_execution_plan_fold_workload_20260907.csv.sha256) |
| V3 checksum-bound blocked execution-capsule receipt and sidecar | [`reports/data_quality_v3/v3_execution_capsule_readiness_20260907.json`](../reports/data_quality_v3/v3_execution_capsule_readiness_20260907.json) and [`.sha256`](../reports/data_quality_v3/v3_execution_capsule_readiness_20260907.json.sha256) |
| V3 reader-facing execution readiness report | [`reports/v3_execution_readiness_20260907.md`](../reports/v3_execution_readiness_20260907.md) |
| V3 readiness QA, exact tests, and debug ledger | [`QA record`](../reports/v3_execution_readiness_20260907_qa.md) |
| V3 final focused and broader regression logs and JUnit records | [`focused log`](../reports/data_quality_v3/v3_focused_final_20260907.log), [`focused JUnit`](../reports/data_quality_v3/v3_focused_final_20260907.xml), [`broader log`](../reports/data_quality_v3/v3_broad_final_20260907.log), and [`broader JUnit`](../reports/data_quality_v3/v3_broad_final_20260907.xml), each with `.sha256` |
| V3 independently recomputed descriptor/row-hash verification | [`verification receipt`](../reports/data_quality_v3/v3_execution_verification_20260907.json) and [`.sha256`](../reports/data_quality_v3/v3_execution_verification_20260907.json.sha256) |
| V3 reproducible verification and figure helper | [`reports/verify_v3_execution_readiness_20260907.py`](../reports/verify_v3_execution_readiness_20260907.py) |
| V3 planned validation-transition chart and sidecar | [`chart`](../reports/figures/v3_validation_transitions_20260907.png) and [`.sha256`](../reports/figures/v3_validation_transitions_20260907.png.sha256) |
| V3 current denial / future execution-contract diagram and sidecar | [`diagram`](../reports/figures/v3_gate_first_path_20260907.png) and [`.sha256`](../reports/figures/v3_gate_first_path_20260907.png.sha256) |
| Preserved 4 September V3 execution evidence | [`plan`](../reports/data_quality_v3/v3_execution_plan_20260904.csv), [`fold workload`](../reports/data_quality_v3/v3_execution_plan_fold_workload_20260904.csv), and [`capsule receipt`](../reports/data_quality_v3/v3_execution_capsule_readiness_20260904.json), each with `.sha256` |
| V3 seven-decision author/advisor packet, all decisions pending | [`docs/v3_author_decision_packet_20260904.md`](v3_author_decision_packet_20260904.md) |
| SETSMART V3 request packet, DRAFT — UNSENT | [`requests/setsmart_v3_20260901/README.md`](../requests/setsmart_v3_20260901/README.md) |
| Exact five-file/33-field provider schema | [`requests/setsmart_v3_20260901/requested_five_file_schema.csv`](../requests/setsmart_v3_20260901/requested_five_file_schema.csv) |
| Byte-identical 138-ticker request acquisition union | [`requests/setsmart_v3_20260901/set50_set100_acquisition_union_138.csv`](../requests/setsmart_v3_20260901/set50_set100_acquisition_union_138.csv) |
| Request-packet checksum inventory and sidecar | [`SHA256SUMS`](../requests/setsmart_v3_20260901/SHA256SUMS) and [`SHA256SUMS.sha256`](../requests/setsmart_v3_20260901/SHA256SUMS.sha256) |
| V3 membership-development guide | `docs/v3_membership_development.md` |
| Draft SETSMART academic data-support letter | `docs/setsmart_academic_data_request_draft.md` |
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
| Canonical real OHLCV + SET market-context campaign features | [`data/processed/features_real_ohlcv_market.csv`](../data/processed/features_real_ohlcv_market.csv) |
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

## HPC Showcase Communication And Process Reports — 2026-07-20

These artifacts explain the verified project boundary at two levels: a
self-guided HPC Showcase presentation for a general audience, and five focused
technical process reports for research review. The bilingual PDF is a
visually faithful slide-render export so Thai typography is preserved; the
bilingual PPTX remains the editable source.

| artifact | path |
| --- | --- |
| English HPC Showcase presentation (editable PPTX) | [`output/presentations/hpc_showcase_hpc_thai_stock_ai_english_20260720.pptx`](../output/presentations/hpc_showcase_hpc_thai_stock_ai_english_20260720.pptx) |
| Thai-English HPC Showcase presentation (editable PPTX) | [`output/presentations/hpc_showcase_hpc_thai_stock_ai_bilingual_20260720.pptx`](../output/presentations/hpc_showcase_hpc_thai_stock_ai_bilingual_20260720.pptx) |
| English HPC Showcase presentation (display PDF) | [`output/pdf/hpc_showcase_hpc_thai_stock_ai_english_20260720.pdf`](../output/pdf/hpc_showcase_hpc_thai_stock_ai_english_20260720.pdf) |
| Thai-English HPC Showcase presentation (display PDF) | [`output/pdf/hpc_showcase_hpc_thai_stock_ai_bilingual_20260720.pdf`](../output/pdf/hpc_showcase_hpc_thai_stock_ai_bilingual_20260720.pdf) |
| Editable HPC Showcase content draft | [`reports/hpc_showcase_hpc_presentation_draft_20260720.md`](../reports/hpc_showcase_hpc_presentation_draft_20260720.md) |
| Project progress and research-gap report | [`output/pdf/01_project_progress_and_research_gap_20260717.pdf`](../output/pdf/01_project_progress_and_research_gap_20260717.pdf) |
| Storage-safe cutover execution report | [`output/pdf/02_storage_safe_cutover_execution_20260717.pdf`](../output/pdf/02_storage_safe_cutover_execution_20260717.pdf) |
| HPC jobs, resources, and capacity audit | [`output/pdf/03_hpc_jobs_resources_and_capacity_audit_20260717.pdf`](../output/pdf/03_hpc_jobs_resources_and_capacity_audit_20260717.pdf) |
| Reproducibility, provenance, and validation report | [`output/pdf/04_reproducibility_provenance_and_validation_20260717.pdf`](../output/pdf/04_reproducibility_provenance_and_validation_20260717.pdf) |
| Next-phase research execution plan | [`output/pdf/05_next_phase_research_execution_plan_20260717.pdf`](../output/pdf/05_next_phase_research_execution_plan_20260717.pdf) |
| Reusable process-report generator | [`scripts/build_process_pdf_reports.py`](../scripts/build_process_pdf_reports.py) |
| Thai-safe slide-image PDF exporter | [`scripts/build_slide_image_pdf.py`](../scripts/build_slide_image_pdf.py) |
| Template-based English HPC Showcase poster (editable PPTX) | [`output/presentations/hpc_showcase_hpc_poster_template_english_20260720.pptx`](../output/presentations/hpc_showcase_hpc_poster_template_english_20260720.pptx) |
| Template-based Thai-English HPC Showcase poster (editable PPTX) | [`output/presentations/hpc_showcase_hpc_poster_template_bilingual_20260720.pptx`](../output/presentations/hpc_showcase_hpc_poster_template_bilingual_20260720.pptx) |
| Template-based English HPC Showcase poster (print PDF) | [`output/pdf/hpc_showcase_hpc_poster_template_english_20260720.pdf`](../output/pdf/hpc_showcase_hpc_poster_template_english_20260720.pdf) |
| Template-based Thai-English HPC Showcase poster (print PDF) | [`output/pdf/hpc_showcase_hpc_poster_template_bilingual_20260720.pdf`](../output/pdf/hpc_showcase_hpc_poster_template_bilingual_20260720.pdf) |
| Template-based poster handoff and confirmation checklist | [`reports/hpc_showcase_hpc_poster_template_handoff_20260720.md`](../reports/hpc_showcase_hpc_poster_template_handoff_20260720.md) |

All 14 presentation slides and both one-page posters passed PPTX rendering and
overflow inspection. All seven pages of each presentation PDF and both poster
PDFs were visually inspected. All 20 pages of the five technical reports passed
Poppler rendering, text extraction, and page-bounds checks. Every artifact
states that the full 150-trial screening remains unsubmitted and does not
present a final investment or scientific conclusion.

## Verified Large-Scale Campaign Evidence

Canonical project progress is **11/37 verified gates (29.7%)**. The active V2 control graph now reports **9 VERIFIED, 1 READY (`screening`), and 25 PENDING** across **35 stages** because `calibration_lineage` consolidates three former gates into one. Native jobs `3540` and `3541` verified vector calibration and determinism. Failed checkpoint-write job `3542` remains quarantined; replacement job `3543` established the checkpoint-writing baseline. The final cutover preserved rollback and lineage evidence, applied the storage-safe release, and used exactly one new job, `3550`, to verify the replacement smoke. V2 has five submissions to date, and the full screening is unsubmitted.

Job `3550` establishes runtime and no-checkpoint admissibility for the 150-run full screening array: all five algorithm projections remain below 72 hours, all required receipts passed, and the observed smoke footprint fell from 275,209,516 to 137,777,145 bytes (about 49.94%). Submission authorization is still closed. The post-smoke decision passed all eight byte/inode checks and its write probe but left only 331,455,656 bytes (0.309 GiB) of modeled personal-group margin; it expired after 15 minutes and must not be reused.

Historical accepted V1 campaign state: **10 VERIFIED, 1 READY (`screening`), and 17 PENDING**, covering **498 planned runs** and **29.44 million requested timesteps**. Exactly five jobs were submitted (`3525`, `3526`, `3528`, `3529`, and `3530`); job `3530` is the accepted calibration, and no screening job was submitted after reaching the five-job ceiling.

| artifact | evidence |
| --- | --- |
| Campaign control-plane report | [`docs/research_campaign_control_plane_report.md`](research_campaign_control_plane_report.md) |
| Large-scale HPC checklist | [`docs/large_scale_hpc_research_checklist.md`](large_scale_hpc_research_checklist.md) |
| Generated live campaign status | [`campaigns/large_scale_thai_drl/reports/campaign_status.md`](../campaigns/large_scale_thai_drl/reports/campaign_status.md) |
| Calibration execution and incident report | [`reports/bistka_calibration_execution_20260715.md`](../reports/bistka_calibration_execution_20260715.md) |
| Post-calibration deployment gap audit | [`reports/post_calibration_deployment_gap_audit_20260715.md`](../reports/post_calibration_deployment_gap_audit_20260715.md) |
| V2 fail-closed control-plane implementation report | [`reports/v2_control_plane_implementation_20260715.md`](../reports/v2_control_plane_implementation_20260715.md) |
| V1-to-V2 calibration migration report | [`reports/v1_to_v2_calibration_migration_20260716.md`](../reports/v1_to_v2_calibration_migration_20260716.md) |
| Screening smoke execution, quota recovery, and full-screening gate | [`reports/screening_smoke_execution_20260716.md`](../reports/screening_smoke_execution_20260716.md) |
| Storage-safe screening candidate technical report | [`reports/screening_storage_safe_release_candidate_20260717.md`](../reports/screening_storage_safe_release_candidate_20260717.md) |
| Applied storage-safe cutover and job-`3550` execution report | [`reports/screening_storage_safe_cutover_execution_20260717.md`](../reports/screening_storage_safe_cutover_execution_20260717.md) |
| Storage-safe projection figure | [`PNG`](../reports/figures/screening_storage_safe_projection_20260717.png) and [`SVG`](../reports/figures/screening_storage_safe_projection_20260717.svg) |
| Generated live V2 campaign status | [`campaigns/large_scale_thai_drl_v2/reports/campaign_status.md`](../campaigns/large_scale_thai_drl_v2/reports/campaign_status.md) |
| V2 live BistKA capacity snapshot | [`reports/bistka_capacity_v2_20260716.txt`](../reports/bistka_capacity_v2_20260716.txt) |
| V2 login-shell capacity, module, and empty-queue snapshot | [`reports/bistka_capacity_v2_20260716_login.txt`](../reports/bistka_capacity_v2_20260716_login.txt) |
| V2 base-release BistKA preflight | [`reports/bistka_calibration_v2_preflight.json`](../reports/bistka_calibration_v2_preflight.json) |
| Archived checksum-bound V1 calibration import receipt (historical migration evidence) | [`campaigns/large_scale_thai_drl_v2/imports/archive/v1_calibration_evidence_before_dependency_lock_20260716.json`](../campaigns/large_scale_thai_drl_v2/imports/archive/v1_calibration_evidence_before_dependency_lock_20260716.json) |
| Archived V1 calibration import SHA-256 sidecar | [`campaigns/large_scale_thai_drl_v2/imports/archive/v1_calibration_evidence_before_dependency_lock_20260716.json.sha256`](../campaigns/large_scale_thai_drl_v2/imports/archive/v1_calibration_evidence_before_dependency_lock_20260716.json.sha256) |
| Native V2 vector benchmark, job `3540` | [`reports/vectorized_training_benchmark_3540.json`](../reports/vectorized_training_benchmark_3540.json) |
| Native V2 vector benchmark trials and receipt, job `3540` | [`reports/vectorized_training_benchmark_3540.csv`](../reports/vectorized_training_benchmark_3540.csv) and [`receipt`](../reports/vectorized_training_benchmark_3540.csv.receipt.json) |
| Native V2 determinism result and receipt, job `3541` | [`reports/determinism_3541.json`](../reports/determinism_3541.json) and [`receipt`](../reports/determinism_3541.json.receipt.json) |
| V2 locked screening configuration (`num_envs: 1`) | [`campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml`](../campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml) |
| V2 locked screening configuration SHA-256 sidecar | [`campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml.sha256`](../campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml.sha256) |
| V2 calibration decision provenance | [`campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.decision.json`](../campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.decision.json) |
| V2 calibration decision SHA-256 sidecar | [`campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.decision.json.sha256`](../campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.decision.json.sha256) |
| V2 cumulative screening manifest | [`campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2_manifest.json`](../campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2_manifest.json) |
| V2 cumulative screening manifest SHA-256 sidecar | [`campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2_manifest.sha256`](../campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2_manifest.sha256) |
| V2 cumulative screening bundle | [`campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2.tar.gz`](../campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2.tar.gz) |
| V2 cumulative screening bundle SHA-256 sidecar | [`campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2.tar.gz.sha256`](../campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2.tar.gz.sha256) |
| V2 screening-release BistKA preflight | [`reports/bistka_screening_v2_preflight.json`](../reports/bistka_screening_v2_preflight.json) |
| Verified five-algorithm screening-smoke outputs, job `3543` | [`results/screening_smoke_3543/`](../results/screening_smoke_3543/) |
| Screening-smoke summary, SHA-256 `4d724b25b430e3574e1e9d8ca1064ece12499a3c6001a98abf522f5aff6c1883` | [`results/screening_smoke_3543/screening_smoke_summary.json`](../results/screening_smoke_3543/screening_smoke_summary.json) |
| Screening-smoke job `3543` accounting | [`reports/slurm_accounting_screening_smoke_3543.txt`](../reports/slurm_accounting_screening_smoke_3543.txt) |
| Screening-smoke job `3543` scheduler output and error logs | [`logs/screening_smoke_3543.out`](../logs/screening_smoke_3543.out) and [`logs/screening_smoke_3543.err`](../logs/screening_smoke_3543.err) |
| Fresh post-smoke group/project quota and empty-queue evidence | [`reports/screening_capacity_post_smoke_20260717.txt`](../reports/screening_capacity_post_smoke_20260717.txt) |
| Checksum-backed full-screening capacity hold | [`reports/screening_capacity_decision_20260717.json`](../reports/screening_capacity_decision_20260717.json) and [`SHA-256 sidecar`](../reports/screening_capacity_decision_20260717.json.sha256) |
| Storage-safe release profile | [`config/bistka_profiles_v2/screening_storage_safe.yaml`](../config/bistka_profiles_v2/screening_storage_safe.yaml) |
| Storage-safe generated deployment configuration | [`campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_storage_safe_v2_deployment.yaml`](../campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_storage_safe_v2_deployment.yaml) |
| Storage-safe candidate manifest, release `16ca55f9264a54e3bc86ecf51d2e4b3ab0c4e85ddb80f22e7dd8585a466a3dc2` | [`manifest`](../campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_storage_safe_v2_manifest.json), SHA-256 `8a5f0f360610c127f828ab9936ca6733fd3a892b612b848a5a3e2d4c01b6a20f` |
| Storage-safe candidate bundle | [`bundle`](../campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_storage_safe_v2.tar.gz), SHA-256 `db1b226dbc57937167453c373da01a8a8dc78c0be7a98e700d3d2e6062d839a8` |
| Checksum-bound storage-safe BistKA quota/write snapshot | [`snapshot`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/quota_snapshot_20260717T125752Z.json) and [`sidecar`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/quota_snapshot_20260717T125752Z.json.sha256) |
| Point-in-time storage-safe capacity decision | [`decision`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/storage_safe_decision_20260717T125752Z.json), [`sidecar`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/storage_safe_decision_20260717T125752Z.json.sha256), and [`operational note`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/README.md) |
| Cutover calibration-lineage receipt, SHA-256 `25cda3f51aeb200353064e495fd58484ade711f36ca575c9633ae231e069c807` | [`receipt`](../campaigns/large_scale_thai_drl_v2/imports/storage_safe_cutover_calibration_lineage.json) and [`sidecar`](../campaigns/large_scale_thai_drl_v2/imports/storage_safe_cutover_calibration_lineage.json.sha256) |
| Pre-cutover rollback archive, SHA-256 `5dd485c41ee5e287c893081f760f1f496fb458e1e1097e691fb64e254e822aee` | [`archive`](../campaigns/large_scale_thai_drl_v2/cutovers/storage_safe_20260717_pre_cutover/pre_cutover_control_and_release.tar.gz) and [`snapshot receipt`](../campaigns/large_scale_thai_drl_v2/cutovers/storage_safe_20260717_pre_cutover/snapshot_receipt.json) |
| Active final storage-safe manifest, release `56358f2a4996f654426814d88e622b7f217ac89bfc02f8bbdacc27e66873b55e` | [`manifest`](../campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_storage_safe_cutover_v2_manifest.json), SHA-256 `439b590bc4a85f03a6d98723ee3d28e40acdc00d95bdfbf48d310f07d867e25c` |
| Active final storage-safe bundle | [`bundle`](../campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_storage_safe_cutover_v2.tar.gz), SHA-256 `a5135665dd14fdfcd0a5dcdbfab70a7e6f88fd78fc75ff946df9851c01a4072e` |
| Passing final cutover BistKA preflight | [`receipt`](../reports/bistka_screening_storage_safe_cutover_v2_preflight.json), SHA-256 `1ccac9827e2f018927d711c0fb0918119c79f8721bd6d82d3d08ed026ab59f0c` |
| Pre-job-`3550` capacity snapshot and decision | [`snapshot`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/quota_snapshot_20260717T144710Z.json) and [`decision`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/storage_safe_decision_20260717T144710Z.json) |
| Verified storage-safe smoke outputs, job `3550` | [`results`](../results/screening_storage_safe_smoke_3550/), [`summary`](../results/screening_storage_safe_smoke_3550/screening_smoke_summary.json), and [`safe gate`](../results/screening_storage_safe_smoke_3550/storage_safe_smoke_validation.json) |
| Job-`3550` strict import and accounting | [`import validation`](../reports/screening_storage_safe_smoke_3550_import_validation.json) and [`SLURM accounting`](../reports/slurm_accounting_screening_storage_safe_smoke_3550.txt) |
| Post-job-`3550` capacity snapshot and expired pass | [`snapshot`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/quota_snapshot_20260717T154236Z.json) and [`decision`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/storage_safe_post_smoke_decision_20260717T154236Z.json) |
| Fail-closed full-screening execution-window gate | [`scripts/prepare_screening_execution_window.py`](../scripts/prepare_screening_execution_window.py) and [`tests/test_prepare_screening_execution_window.py`](../tests/test_prepare_screening_execution_window.py) |
| Consume-once authorized screening renderer (render only; zero submission calls) | [`scripts/render_authorized_screening.py`](../scripts/render_authorized_screening.py) and [`tests/test_render_authorized_screening.py`](../tests/test_render_authorized_screening.py) |
| Reproducible zero-job BistKA execution-window evidence producer | [`scripts/capture_screening_execution_window_evidence.sh`](../scripts/capture_screening_execution_window_evidence.sh) |
| Authoritative 26 July scheduler, empty user/account queues, credit, QOS, partition, and live-release evidence | [`snapshot`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/scheduler_snapshot_20260726T143716Z.txt) and [`sidecar`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/scheduler_snapshot_20260726T143716Z.txt.sha256) |
| Authoritative 26 July group/project quota and direct-child write-probe evidence | [`snapshot`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/quota_snapshot_20260726T143716Z.json) and [`sidecar`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/quota_snapshot_20260726T143716Z.json.sha256) |
| Fresh 26 July read-only 248-file BistKA preflight | [`receipt`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/live_preflight_20260726T143716Z.json) and [`sidecar`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/live_preflight_20260726T143716Z.json.sha256) |
| 26 July execution-window safety hold (0 jobs; 0.308 GiB margin) | [`reports/screening_execution_window_hold_20260726.md`](../reports/screening_execution_window_hold_20260726.md) |
| Quarantined quota incident and exact failed job `3542` evidence | [`campaigns/large_scale_thai_drl_v2/incidents/screening_smoke_3542/`](../campaigns/large_scale_thai_drl_v2/incidents/screening_smoke_3542/) |
| Frozen accepted V1 snapshot | [`campaigns/large_scale_thai_drl/snapshots/v1_accepted_20260715.json`](../campaigns/large_scale_thai_drl/snapshots/v1_accepted_20260715.json) |
| Corrected initial-release preflight | [`reports/bistka_calibration_preflight.json`](../reports/bistka_calibration_preflight.json) |
| Screening-overlay preflight | [`reports/bistka_screening_preflight.json`](../reports/bistka_screening_preflight.json) |
| Accepted vector benchmark summary | [`reports/vectorized_training_benchmark_3530.json`](../reports/vectorized_training_benchmark_3530.json) |
| Accepted vector benchmark trials | [`reports/vectorized_training_benchmark_3530.csv`](../reports/vectorized_training_benchmark_3530.csv) |
| Accepted vector benchmark receipt | [`reports/vectorized_training_benchmark_3530.csv.receipt.json`](../reports/vectorized_training_benchmark_3530.csv.receipt.json) |
| Vector calibration figure | [`reports/figures/vectorized_training_calibration_3530.png`](../reports/figures/vectorized_training_calibration_3530.png) |
| Accepted determinism result | [`reports/determinism_3530.json`](../reports/determinism_3530.json) |
| Accepted determinism receipt | [`reports/determinism_3530.json.receipt.json`](../reports/determinism_3530.json.receipt.json) |
| Locked measured screening configuration (`num_envs: 1`) | [`campaigns/large_scale_thai_drl/derived/locked_large_scale_screening.yaml`](../campaigns/large_scale_thai_drl/derived/locked_large_scale_screening.yaml) |
| Calibration decision receipt | [`campaigns/large_scale_thai_drl/derived/locked_large_scale_screening.decision.json`](../campaigns/large_scale_thai_drl/derived/locked_large_scale_screening.decision.json) |
| Corrected 149-file initial release manifest | [`campaigns/large_scale_thai_drl/deployment/bistka_deployment_manifest.json`](../campaigns/large_scale_thai_drl/deployment/bistka_deployment_manifest.json) |
| Six-file screening overlay manifest | [`campaigns/large_scale_thai_drl/deployment/bistka_screening_overlay_manifest.json`](../campaigns/large_scale_thai_drl/deployment/bistka_screening_overlay_manifest.json) |
| Rejected quota/stale-data evidence quarantine | [`reports/rejected_calibration/20260715_quota_and_data_shadow/`](../reports/rejected_calibration/20260715_quota_and_data_shadow/) |
| Calibration scheduler logs for jobs 3525, 3526, 3528, 3529, and 3530 | [`logs/`](../logs/) |

Native V2 job `3540` independently selected `num_envs: 1`, and job `3541` passed determinism across two independent processes for 549 steps. The resulting locked screening configuration has SHA-256 `d7b6ab71ede33ae4c4106fded3078b9a4eb3c56af345e67960061feea1897c1e`. Historical V1 accepted job `3530` also selected `num_envs: 1`; its median throughput for 1/2/4/8/16 environments was 347.9200/265.9007/217.0773/143.4171/83.6104 samples/second, with maximum process-tree RSS of 602.484/1,641.066/2,661.281/4,690.129/8,768.617 MiB. All accepted evidence is bound to the corrected 2,183,530-byte CSV with SHA-256 `b877b08ed7a022db47739423291df0232ce377869cb7e0de6162ab82291ac6cc`.

The downstream release-closure gap and the storage-safe cutover are implemented. The active 35-stage graph preserves the same 503-run scientific workload while consolidating calibration lineage. Aggregate stages inherit release, preflight, and runtime bindings from `run_source`; strict selections and the protocol lock consume only verified aggregate evidence. The final storage-safe screening release, returned preflight, and job `3550` gate are verified. Full screening remains unsubmitted, and feature ablation remains blocked until observed macro and fundamental CSVs are hash/byte certified in a passing readiness receipt or the preregistered scope is formally amended before test access.

## V2 Control Plane And Verified Base Deployment

| artifact | evidence |
| --- | --- |
| V2 campaign specification | [`config/research_campaign.yaml`](../config/research_campaign.yaml) |
| V2 base deployment configuration | [`config/bistka_deployment_v2.yaml`](../config/bistka_deployment_v2.yaml) |
| Five cumulative release profiles | [`config/bistka_profiles_v2/`](../config/bistka_profiles_v2/) |
| V2 base manifest, release `4b7d1d4b887bf08675bc664683b06f0fb103369016f7233e6caeb594e41ee382` | [`campaigns/large_scale_thai_drl_v2/deployment/bistka_deployment_v2_manifest.json`](../campaigns/large_scale_thai_drl_v2/deployment/bistka_deployment_v2_manifest.json) |
| V2 base bundle | [`campaigns/large_scale_thai_drl_v2/deployment/bistka_release_v2.tar.gz`](../campaigns/large_scale_thai_drl_v2/deployment/bistka_release_v2.tar.gz) |
| Archived V1 calibration import receipt (historical migration evidence) | [`campaigns/large_scale_thai_drl_v2/imports/archive/v1_calibration_evidence_before_dependency_lock_20260716.json`](../campaigns/large_scale_thai_drl_v2/imports/archive/v1_calibration_evidence_before_dependency_lock_20260716.json) |
| Locked V2 screening config, SHA-256 `d7b6ab71ede33ae4c4106fded3078b9a4eb3c56af345e67960061feea1897c1e` | [`campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml`](../campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.yaml) |
| Native V2 screening decision, SHA-256 `bc0f859d20fcf4fe3c09c7492cf0871b982f69adbfe0713a327eb72a78a167ca` | [`campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.decision.json`](../campaigns/large_scale_thai_drl_v2/derived/locked_large_scale_screening.decision.json) |
| Cumulative screening manifest, release `ef89ff5a738c5b25460ae40ad7789239134d761ce511a36665f857e01e3d74b4` | [`campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2_manifest.json`](../campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2_manifest.json) |
| Cumulative screening bundle | [`campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2.tar.gz`](../campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_v2.tar.gz) |
| Live capacity and queue evidence | [`reports/bistka_capacity_v2_20260716.txt`](../reports/bistka_capacity_v2_20260716.txt) |
| Passing BistKA base preflight | [`reports/bistka_calibration_v2_preflight.json`](../reports/bistka_calibration_v2_preflight.json) |
| Passing BistKA screening preflight | [`reports/bistka_screening_v2_preflight.json`](../reports/bistka_screening_v2_preflight.json) |
| Verified job `3543` screening-smoke summary | [`results/screening_smoke_3543/screening_smoke_summary.json`](../results/screening_smoke_3543/screening_smoke_summary.json) |
| Full-screening capacity decision (`eligible_for_submission: false`) | [`reports/screening_capacity_decision_20260717.json`](../reports/screening_capacity_decision_20260717.json) |
| Active storage-safe release `56358f2a4996f654426814d88e622b7f217ac89bfc02f8bbdacc27e66873b55e` | [`manifest`](../campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_storage_safe_cutover_v2_manifest.json) and [`bundle`](../campaigns/large_scale_thai_drl_v2/deployment/bistka_screening_storage_safe_cutover_v2.tar.gz) |
| Passing active storage-safe preflight | [`reports/bistka_screening_storage_safe_cutover_v2_preflight.json`](../reports/bistka_screening_storage_safe_cutover_v2_preflight.json) |
| Verified storage-safe smoke job `3550` | [`results`](../results/screening_storage_safe_smoke_3550/) and [`import validation`](../reports/screening_storage_safe_smoke_3550_import_validation.json) |
| Post-smoke capacity decision (expired historical pass) | [`campaigns/large_scale_thai_drl_v2/operational/screening_capacity/storage_safe_post_smoke_decision_20260717T154236Z.json`](../campaigns/large_scale_thai_drl_v2/operational/screening_capacity/storage_safe_post_smoke_decision_20260717T154236Z.json) |
| Failed job `3542` quota-incident quarantine | [`campaigns/large_scale_thai_drl_v2/incidents/screening_smoke_3542/`](../campaigns/large_scale_thai_drl_v2/incidents/screening_smoke_3542/) |
| Quota-safe atomic applier | [`src/campaigns/atomic_release_apply.py`](../src/campaigns/atomic_release_apply.py) |
| Validating V2 staging script | [`scripts/stage_bistka_deployment_v2.sh`](../scripts/stage_bistka_deployment_v2.sh) |

The live V2 base manifest binds **183 files totaling 5,669,381 bytes** and the certified 2,183,530-byte market CSV. Its manifest SHA-256 is `3083a36024c29b1e48b52fc3af66454ae4feceb9d4477773ffc35b8364cc8872`, bundle SHA-256 is `30a078f5a8e0cc1052339aaccc2d9e7a04f5af37c2b018e838952940a9e6a422`, and returned base-preflight SHA-256 is `82d4fe058b4d64142786764362ab3caa9930224d545733d48dfab8c4ff2493e8`.

The 207-file cumulative checkpoint-writing release and job `3543` remain preserved as the predecessor baseline. The active replacement is the deterministic **248-file, 7,850,129-byte** storage-safe release. Its manifest SHA-256 is `439b590bc4a85f03a6d98723ee3d28e40acdc00d95bdfbf48d310f07d867e25c`, bundle SHA-256 is `a5135665dd14fdfcd0a5dcdbfab70a7e6f88fd78fc75ff946df9851c01a4072e`, and returned preflight SHA-256 is `1ccac9827e2f018927d711c0fb0918119c79f8721bd6d82d3d08ed026ab59f0c`. Job `3550` produced five complete manifests, five retention receipts, and zero checkpoint paths; its summary SHA-256 is `e3095e1149a6cefe9d12cef97ca38b0b1ef1585bfa3e4d20bf66a4ce977b88c3`. `screening` is the sole READY stage, but the full array remains unsubmitted until a newly authorized window captures a fresh capacity pass with safer margin. Feature ablation remains scientifically blocked until observed macro and fundamental inputs are readiness-certified or the preregistered scope is formally amended before test access.

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
