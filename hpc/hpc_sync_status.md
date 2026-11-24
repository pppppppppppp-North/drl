# BistKA HPC Sync Status

Last full remote evidence: 2026-07-26

Latest connection probe: 2026-07-31
Latest connection and code copy: 2026-11-22

Latest job check: 2026-11-24

## 2026-11-22 Connection And Code Copy

- **Connection:** the user signed in once with the password (`s00658@hpc.jsite.me`, reachable over the VPN and as `10.205.100.101`), opening an SSH control-master session that later commands reused. No key is installed on BistKA.
- **State, read-only:** no jobs queued; every partition idle. Service hours: project `25sfcs03` 9,978.72 of 10,000 SHr left (expires 2027-02-28), personal `s00658` 496.25 of 500. Quota: home 23.66 of 50 GB, project 36.64 of 300 GB. Last job: 3624 on 2026-07-22.
- **Environment:** `/lustrefs/project/25sfcs03/envs/drl_env` has Python 3.12.13, torch 2.10.0, Stable-Baselines3 2.8.0, gymnasium 1.2.3, numpy 2.4.6 and pandas 3.0.3. The development Mac has Python 3.11.13, torch 2.15.0, SB3 2.9.2, gymnasium 1.4.0, numpy 2.2.6 and pandas 2.3.3. pandas 3.0 changes copy and string semantics, so the test suite must pass on BistKA before any training run.
- **Code copy:** commit `5c4ac24` (branch `point-in-time-universe`) was copied to the new folder `/lustrefs/project/25sfcs03/drl_thai_stock_20261122`: a `git archive` without `data/holdout_sealed/`, plus the untracked inputs under `data/raw/extended/` (the SET100 price files, the price map, the SET50 ETF, the macro proxies and the five-stock prices). 788 files, 25 MB.
  - **Verified:** every file's SHA-256 on BistKA matched the local file. `COPY_MANIFEST.sha256` and `COPY_RECEIPT.json` sit in the folder; `sha256sum -c COPY_MANIFEST.sha256` re-checks it.
  - **Left alone:** the July deployment in `/lustrefs/project/25sfcs03/drl_thai_stock` was not changed.
- **Test jobs (operator-approved):** `slurm/test_suite_cpu.sbatch` on `compute-devel`, run from the new folder. It was copied there after the checksum check, so it is not in `COPY_MANIFEST.sha256`.
  - **Job `4406`** (58 s): pytest stopped at collection, because `scipy` and `pdfplumber` are missing from the environment. No test ran.
  - **Job `4407`** (2 min 46 s, with `--continue-on-collection-errors`): 573 passed, 60 failed, 1 skipped, 2 collection errors. 54 failures are the V3-campaign tests that also fail on the Mac. Five V2 deployment and fetch-tool tests refuse the non-July project root by design, and one LSTM-forecast test needs `scipy`. Every test of the training code passed. Report: `logs/tests_4407.txt` in the folder.
  - **Cost:** 0.14 SHr in all (project balance 9,978.58 SHr).
- **scipy installed (operator-approved):** `scipy==1.17.1`, the Mac's version, added to `/lustrefs/project/25sfcs03/envs/drl_env` with pip on the login node. A dry run showed it changed no other package; `pip check` finds no broken requirement. The package lists before and after are in `logs/drl_env_freeze_{before,after}_scipy_20261122.txt` in the new folder. It imports, and pandas' Spearman correlation gives the same value as on the Mac. `pdfplumber` is still missing; only the constituent-list ingestion needs it.
- **Job `4408`** (operator-approved, 38 s): the two test files that needed scipy, `tests/test_pit_heuristic_baselines.py` and `tests/test_pit_lstm_forecast_baseline.py`, selected with the script's new `TESTS` variable. 8 of 8 passed, as on the Mac. Project balance 9,978.57 SHr.
- **Job `4409`, training pool (operator-approved, submitted 21:00):** one job instead of about 210 array elements, as the operator asked. `slurm/walk_forward_pool.sbatch` with `POOL=slurm/pools/bistka_20261122.tasks` on `compute-long`, 32 CPUs. 210 tasks at 300k timesteps, seeds 42–46: SAC, TD3, DDPG, A2C and PPO on the five starter stocks, and the shared-weight point-in-time design. The smoke stage passed for all six setups by 21:03 and the pool started at 21:02:52. Estimate 12–13 hours, about 420 SHr.
- **Job `4409` progress, 2026-11-23:** still running with no failures, but no task had finished after 13 hours, so the estimate above was far too low.
  - **Measured speed:** each SAC worker writes about 5.1 steps per second to its replay buffer (5.07–5.44 over 10 minutes, measured with `hpc/buffer_rate.py`, which reads the processes' page tables and changes nothing in the job). At 10:10 every SAC run was about 228,000 of 300,000 steps in, so one SAC task takes about 17 hours here, against about 3 hours on the Mac.
  - **Why:** the job's 32 CPUs are 16 physical cores of an AMD EPYC 7542 (up to 2.9 GHz) with two hardware threads each, so two workers share each core. Asking for whole cores would double the billed CPUs for less than double the speed.
  - **Cost:** the job bills 44 SHr per hour, not 32, because `compute-long` charges 0.1 SHr per GB of requested memory per hour and the job asked for 128 GB (it uses about 49 GB). By 12:00 the project had used about 650 SHr in all (`sshare` raw usage); about 9,350 were left.
  - **Projection:** the first 32 SAC tasks finish at about 14:00 on 2026-11-23. With TD3 and DDPG near SAC's time and the point-in-time task at 8.5–18 hours (the Mac's 1.5–3 hours scaled by the same factor), the pool ends 67–86 hours after it started, against its 72-hour limit, using about 3,000–3,200 SHr in all. If it reaches the limit, the point-in-time tasks still running are lost and a follow-up job must finish them (about 200–600 SHr more).
  - **First tasks finished, 2026-11-23:** the first 32 SAC tasks ended between 14:03 and 14:33, each after 61,236–62,140 s of training (`timing.csv`), with no failures; each wrote its validation curves, metrics and significance tests. At 16:20 the last 3 SAC tasks ran at about 5.5 steps per second and the 29 TD3 tasks at about 8.4, so a TD3 task takes about 10 hours. With DDPG assumed close to TD3, the pool now ends 54–73 hours after its start for about 2,400–3,200 SHr; it stays inside the 72-hour limit unless a point-in-time task takes more than about 17 hours.
  - **Progress 2026-11-24 18:40:** 93 of 210 tasks done (35 SAC, 35 TD3, 23 DDPG), no failures. A DDPG task takes about 13.4 hours (12.9–13.7 h over 23 tasks), slower than TD3. The 20 point-in-time tasks started 14:18–14:57 beside the last 12 DDPG tasks (running until about 03:30 on 2026-11-25).
  - **Point-in-time speed:** PPO keeps no replay buffer, so `hpc/buffer_rate.py` cannot see its progress. The new `hpc/rollout_period.py` reads `/proc/<pid>/maps` five times a second: each of the 20 minibatch updates in a PPO iteration (n_steps 128, batch 64, 10 epochs) maps and frees about 230 MB, and a gap of about 1.2 s with no such mapping marks the collection of the next 128 steps. Over 150 s, every one of the 20 tasks ran one iteration per 11.0–13.3 s (mean 12.6 s), so 2,344 iterations (300k steps) take 7.2–8.7 hours, at the low end of the 8.5–18 hours assumed. The first point-in-time tasks finish about 22:30–23:00.
  - **Re-projection:** the pool ends about 07:00–10:00 on 2026-11-25, 11–14 hours before its limit at 21:03, for about 2,550–2,700 SHr in all. PPO (`plan_main_weekly_lstm`) and A2C are still estimates (1.7–2.7 h and 1.05–1.9 h a task). No follow-up job is needed.
- **Job `4411`, helper pool (operator-approved, "Submit helper job"; cancelled before it started):** submitted at 11:57 on 2026-11-23 to work the same task list backwards beside job 4409 (`slurm/pools/bistka_20261122_reverse.tasks`, `KEEP_LOCKS=1`, `SMOKE_TIMESTEPS=0`, 32 CPUs, 96 GB). It stayed pending with reason `QOSGrpCpuLimit`: the project's QOS `25sfcs03` allows 32 CPUs at once across all its jobs, and job 4409 holds all 32. Cancelled at 11:58; it never ran and cost nothing. The personal account `s00658` has its own 32-CPU QOS, but only about 496 SHr, less than the helper's projected 1,350.

## 2026-07-31 Connection And Sync Boundary

- The VPN path reached the BistKA SSH endpoint at `10.205.100.101`.
- The bounded key-only probe was rejected because no accepted public key is configured for `s00658`.
- A password-authenticated refresh was not performed without a separate explicit stored-credential approval.
- Zero jobs were submitted and no remote file was changed.
- The new official SET constituent archive and isolated V3 membership-development capsule remain local and have not been synchronized or deployed.
- The authoritative remote scheduler/quota/release evidence therefore remains the checksum-bound 2026-07-26 snapshot documented in `PROGRESS_CHECKLIST.md`.

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

The local project now contains official point-in-time SET50/SET100 membership and sector history for 2021–2024. It is not part of the active BistKA V2 release. The remaining broad-study data blockers are:

- licensed or explicitly authorized OHLCV and corporate actions for the 138-ticker acquisition union,
- licensed trading-status/corporate-action rules that extend the implemented local membership panel into final investability, plus an author-approved locked V3 release,
- point-in-time official macro vintages,
- historical Thai news or sentiment data,
- licensed fundamentals.

Use `docs/external_data_handoff.md` and `docs/data_sources.md` as the source-of-truth handoff for those files.
