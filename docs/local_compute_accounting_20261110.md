# Local Compute Accounting (RQ6, local part)

Date: 2026-11-10; the run trained with the no-trade band and the two source arms added 2026-11-11; the cash-capped run added 2026-11-20
Scope: every walk-forward run since 2026-11-06 ran on the development Mac (8 cores: 4 performance and 4 efficiency; 8 GB of memory). No HPC job has been submitted. RQ6 asks for HPC usage in service hours (SHr), throughput and the speed-up from parallel arrays. This note answers the local part from the runs' own timing files. The BistKA part still needs a submitted job.

Made with `scripts/local_compute_accounting.py`. Output (ignored by git): `results/local_compute_accounting_20261120.csv`, regenerated after the cash-capped run (`..._20261111.csv` after the macro-proxy arm, `..._20261110.csv` before the band run).

## How it is measured

- **Task time:** each task is one model, trained for 100k timesteps and then validated on one CPU core. Its time is recorded in `fold_<Y>/<task>/timing.csv`.
  - **Wall time per task:** it includes any time the Mac slept, ran on battery in Low Power Mode, or held the task paused when memory ran short. It is an upper bound on CPU time.
- **Wall time of a run:** from the first task's `splits.json` to the last task's validation curve, from the file times.
- **Speed-up:** task time divided by wall time, the average number of tasks that ran at once.

## Results

| Run | Tasks | Task-hours | Median minutes per task (range) | Timesteps per second (median) | Wall hours | Speed-up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Five-stock walk-forward, 2 designs × 7 folds × 5 seeds | 70 | 10.2 | 6.1 (1.8–25.8) | 520 | 2.1 | 4.9 |
| Source ablation, 2 designs × 7 folds × 3 seeds | 42 | 38.8 | 8.6 (5.3–212.5) | 193 | 6.6 | 5.9 |
| Algorithm comparison, A2C × 7 folds × 3 seeds | 21 | 1.2 | 3.7 (3.0–3.8) | 450 | 0.2 | 6.8 |
| Point-in-time SET50, first run (7 workers) | 21 | 25.3 | 67.6 (57.0–90.4) | 25 | 3.8 | 6.7 |
| Point-in-time SET50, sticky slots (3–4 workers) | 21 | 31.1 | 70.5 (28.5–225.2) | 24 | 8.7 | 3.6 |
| Point-in-time SET50, shared-weight actor (3 workers) | 21 | 29.0 | 62.0 (38.6–214.8) | 27 | 9.9 | 2.9 |
| Point-in-time SET50, trained with the band (3 workers, overnight on AC power) | 21 | 10.5 | 28.9 (27.9–34.4) | 58 | 3.5 | 3.0 |
| Point-in-time SET50, + BOT policy rate (3 workers, overnight) | 21 | 10.8 | 29.5 (29.0–36.9) | 57 | 3.6 | 3.0 |
| Point-in-time SET50, + macro proxies (3 workers, early morning) | 21 | 12.7 | 31.5 (29.8–57.4) | 53 | 4.3 | 3.0 |
| Point-in-time SET50, cash capped at 20% (3 workers, evening on AC power) | 21 | 12.8 | 33.1 (32.4–48.2) | 50 | 4.3 | 3.0 |
| **Total** | **280** | **182.3** | | | | |

## What it shows

- **The point-in-time universe costs about 10–20 times more per timestep.**
  - **Five stocks:** the LSTM agent trained at about 520 timesteps a second.
  - **Point-in-time SET50:** about 25 a second during the day and 58 overnight on an idle machine, because the encoder runs over 50 slots.
  - **Per model:** a 100k-timestep point-in-time model takes about an hour of one core during the day, beside other work.
  - **Idle machine:** overnight on AC power, with nothing else running, the same shared-weight model (the run trained with the band) took 29 minutes, at 58 timesteps a second. The two source arms that followed took 30 and 32 minutes per model, so the extra inputs barely change the cost. The cash-capped run took 33 minutes per model in the evening, with light work on the Mac. During the day, contention and Low Power Mode, not the model, set the rate.
- **Memory, not cores, limits local parallelism.**
  - **Per worker:** a point-in-time worker needs up to about 750 MB, so only three fit beside the usual desktop apps on the 8 GB Mac.
  - **Wall time:** a 21-task run takes 9–10 hours during the day and 3.5–4.3 hours overnight, early in the morning or in the evening, with an average of about three tasks at once.
  - **With seven workers:** the first point-in-time run ran 6.7 tasks at once but swapped heavily.
- **The long tails are sleep, battery and contention, not training.** The slowest tasks (3.5 hours) overlapped periods with the Mac asleep, on battery in Low Power Mode, or sharing the CPU with other heavy work.
- **On BistKA:** a SLURM array gives each task its own CPU and about 1 GB (`slurm/walk_forward_reward_array.sbatch`).
  - **Projection:** a 21-task point-in-time run would take about one task's time, 30–60 minutes instead of 3.5–10 hours, for about 10–30 core-hours.
  - **Not measured:** this is a projection from the local per-task times. The HPC measurement (SHr, queue time, actual speed-up) needs a submitted job, which needs the operator's approval.
