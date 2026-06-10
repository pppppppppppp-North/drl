# BistKA HPC Sync Status

Last verified: 2026-06-09

## Verified Connection

The project can connect to BistKA with the project user account and password-based SSH.

Verified target:

```text
s00658@10.205.100.101
```

Verified hostname:

```text
bistka-master
```

Project directory:

```text
/lustrefs/project/25sfcs03/drl_thai_stock
```

The BistKA project directory exists, but it is not a git repository. Do not run git commands there expecting normal `pull`, `status`, or branch behavior.

## Sync Method That Worked

Direct `scp` and `rsync` attempts were unreliable during the latest sync. The verified method was:

1. Commit and push local changes to GitHub `main`.
2. SSH into BistKA.
3. Download the GitHub `main` branch archive on the cluster.
4. Extract only the required tracked files into the project directory.

Template command to run from the BistKA project directory:

```bash
curl -L https://github.com/pppppppppppp-North/drl/archive/refs/heads/main.tar.gz \
  -o /tmp/drl_main_latest.tgz

tar -xzf /tmp/drl_main_latest.tgz --strip-components=1 \
  drl-main/README.md \
  drl-main/PROGRESS_CHECKLIST.md \
  drl-main/docs/final_artifacts_manifest.md \
  drl-main/docs/data_sources.md \
  drl-main/docs/external_data_handoff.md \
  drl-main/docs/external_data_provider_request.md \
  drl-main/docs/external_data_intake_checklist.md \
  drl-main/docs/full_external_rebuild.md \
  drl-main/hpc/hpc_sync_status.md \
  drl-main/config/real_ohlcv_full_external.yaml \
  drl-main/data/reference/external_data_manifest_template.csv \
  drl-main/src/data/source_readiness.py \
  drl-main/src/data/intake_validation.py \
  drl-main/tests/test_intake_validation.py \
  drl-main/tests/test_source_readiness.py \
  drl-main/data/reference/set50_or_set100_universe_template.csv \
  drl-main/data/reference/sector_mapping_template.csv \
  drl-main/data/reference/official_macro_long_template.csv \
  drl-main/data/reference/historical_news_sentiment_template.csv \
  drl-main/data/reference/fundamentals_template.csv \
  drl-main/comprehensive_introduction_literature_review.tex \
  drl-main/comprehensive_introduction_literature_review.pdf \
  drl-main/final_project_beamer_100_pages.tex \
  drl-main/final_project_beamer_100_pages.pdf
```

Add or remove paths from the `tar` command as needed. Keep the `drl-main/` prefix because it is the top-level directory inside the GitHub archive.

## Latest Synced Artifacts

The latest sync copied the documentation and readiness artifacts needed to continue from the external-data handoff:

- `docs/external_data_handoff.md`
- `src/data/source_readiness.py`
- `tests/test_source_readiness.py`
- required external data templates in `data/reference/`
- `comprehensive_introduction_literature_review.tex`
- `comprehensive_introduction_literature_review.pdf`
- `README.md`
- `PROGRESS_CHECKLIST.md`
- `docs/final_artifacts_manifest.md`
- `docs/data_sources.md`
- `docs/full_external_rebuild.md`
- `config/real_ohlcv_full_external.yaml`
- `docs/external_data_provider_request.md`
- `docs/external_data_intake_checklist.md`
- `data/reference/external_data_manifest_template.csv`
- `src/data/intake_validation.py`
- `tests/test_intake_validation.py`

## Verification Commands

Run these on BistKA after a sync:

```bash
cd /lustrefs/project/25sfcs03/drl_thai_stock

test -f docs/external_data_handoff.md
test -f comprehensive_introduction_literature_review.pdf

python -m src.data.source_readiness --output reports/source_readiness.csv
python -m src.data.intake_validation \
  --output reports/external_intake_validation.csv \
  --markdown-output reports/external_intake_validation.md
```

The source-readiness and intake-validation commands are expected to exit nonzero until the five required external files are provided. That failure is currently a correct readiness signal, not a code failure.

## Current Boundary

All local and HPC-side scaffolding for the next research step is in place. The remaining work is blocked by missing external datasets:

- broad SET50/SET100 ticker universe,
- complete sector membership or sector-index history,
- historical official macro exports,
- historical Thai news or sentiment data,
- licensed fundamentals.

Use `docs/external_data_handoff.md` and `docs/data_sources.md` as the source-of-truth handoff for those files.
