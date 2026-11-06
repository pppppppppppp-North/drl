# Screening Smoke Job 3542 Incident Evidence

This directory quarantines the exact artifacts, scheduler logs, and accounting
for failed BistKA job `3542`.

- Campaign: `large_scale_thai_drl_v2`
- Stage: `screening_smoke`
- Attempt: `screening_smoke_20260716T160537Z_132d2047`
- Disposition: rejected operational evidence; never admissible as completed
  screening-smoke evidence
- Failure class: Lustre group-quota enforcement (`EDQUOT`) during the terminal
  PPO checkpoint write

The evidence is intentionally outside the project-level `results/`,
`logs/`, and `reports/` paths searched by campaign gates. A later successful
retry must use a new immutable campaign attempt and a new SLURM job ID.
