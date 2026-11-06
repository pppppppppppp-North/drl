# Screening Capacity Evidence

This directory contains checksum-bound, point-in-time evidence for the V2 storage-safe screening candidate, applied cutover, and verified smoke.

- `scheduler_snapshot_20260726T143716Z.txt` is the authoritative 26 July BistKA scheduler, credit, QOS, empty user/account queue, and live deployed-release snapshot. The `T142444Z` pair is retained as the preliminary capture.
- `quota_snapshot_20260726T143716Z.json` is the matching group/project quota and direct-child write-probe snapshot.
- `live_preflight_20260726T143716Z.json` is the matching read-only remote preflight: all 248 deployed files and the observed-data/runtime bindings were reverified.
- The 26 July execution-window gate recomputed a 6,475,756,231-byte requirement against 6,806,358,016 bytes of group soft headroom. The resulting 330,601,785-byte (0.308 GiB) margin was below the 1 GiB authorization minimum, so the gate returned `HOLD` and submitted zero jobs. See [`reports/screening_execution_window_hold_20260726.md`](../../../../reports/screening_execution_window_hold_20260726.md).
- `quota_snapshot_20260717T144710Z.json` is the fresh post-cutover snapshot captured immediately before storage-safe smoke job `3550`.
- `storage_safe_decision_20260717T144710Z.json` is the matching `pass` decision: all eight group/project byte and inode checks passed, the direct-child write probe fsynced and was removed, and the maximum projected task runtime remained below 72 hours.
- `quota_snapshot_20260717T154236Z.json` is the fresh snapshot captured after job `3550` completed and the queue returned to zero jobs.
- `storage_safe_post_smoke_decision_20260717T154236Z.json` is the matching post-smoke `pass`: all eight checks and the write probe passed, but the modeled byte margin narrowed to 331,455,656 bytes (0.309 GiB).
- `quota_snapshot_20260717T125752Z.json` records exact group/project byte and inode quotas plus a bounded, fsynced direct-child write probe on BistKA.
- `storage_safe_decision_20260717T125752Z.json` recomputes the five-algorithm runtime and checkpoint-free storage projection from the verified job `3543` smoke summary.
- Every JSON file has an exact sibling `.sha256` sidecar.

Every decision is historical after its configured 15-minute freshness window. A recorded `candidate_capacity_eligible: true` is not submission authorization and must never be reused as a live gate. Immediately before any future job submission, capture a new snapshot on the BistKA login node, assess it locally or in the exact deployed release, and preserve both new sidecars.

The `2026-07-26T15:43:21Z` follow-up reached the SSH service through the reconnected VPN, but the locally available key-only authentication was rejected. It executed only the remote no-op `true`, produced no new evidence, and submitted no job. Restore approved SSH authentication before attempting another capture; do not treat VPN reachability alone as an authenticated execution window.

The standalone execution-window gate in `scripts/prepare_screening_execution_window.py` adds scheduler/account-queue, conservative-credit, live-release, fresh 248-file remote-preflight, campaign-state, exact-SLURM, minimum-TTL, and 1 GiB margin checks that the older point-in-time storage decisions do not contain. It binds the canonical job-`3550` smoke evidence, verifies the complete local release inventory, writes an immutable checksum-bound receipt only on full authorization, and never invokes `sbatch`. SHA-256 sidecars detect corruption and local drift; authenticity comes from capturing and transferring the evidence through the trusted VPN/SSH session, not from the unkeyed sidecars themselves.

Keep new scheduler snapshots in this operational directory, not under the broad historical `reports/bistka_capacity_*` campaign glob. Run the gate from the project root with fresh paths:

```bash
python -m scripts.prepare_screening_execution_window \
  --project-root . \
  --quota-snapshot campaigns/large_scale_thai_drl_v2/operational/screening_capacity/quota_snapshot_<UTC>.json \
  --scheduler-snapshot campaigns/large_scale_thai_drl_v2/operational/screening_capacity/scheduler_snapshot_<UTC>.txt \
  --live-remote-preflight campaigns/large_scale_thai_drl_v2/operational/screening_capacity/live_preflight_<UTC>.json \
  --output reports/screening_execution_windows/screening_execution_window_<UTC>.json
```

Exit code `0` means the sealed, checksum-bound render-only receipt was written. Exit code `2` means `HOLD`; no receipt, attempt, command, or job is created. Production safety floors cannot be lowered by CLI flags or API arguments.

If and only if the gate writes an `authorized: true` receipt, use the supported consume-once renderer instead of calling the campaign controller directly:

```bash
python -m scripts.render_authorized_screening \
  --project-root . \
  --receipt reports/screening_execution_windows/screening_execution_window_<UTC>.json
```

The consumer verifies the receipt sidecar and invariants, requires its filename stamp to match all three evidence files, reruns the complete live gate on its own wall clock, and rechecks the receipt-specific TTL immediately before and after rendering. It claims the evidence identity globally, acquires a permanent stage-wide render lock, requires exactly one new immutable attempt with a valid self-hash and matching command exports, then writes a checksum-bound consumption record. A partial failure is preserved as an in-doubt record and cannot auto-retry. It does **not** execute the rendered script or submit a job. Direct controller rendering and direct execution of the generated script remain outside the supported screening workflow until mandatory receipt/TTL enforcement at submission can be included in a newly built and remotely preflighted release without drifting the active deployment.

Static or dangling symlinks across the controller's command, attempt, status-report, and event-log outputs are rejected before rendering. The next manifest-locked controller release must also publish its mutable text/append outputs through no-follow, directory-bound writers to close the remaining same-user path-race class before eventual script execution is allowed.

For a future window, use `scripts/capture_screening_execution_window_evidence.sh` on `bistka-master` with a UTC `YYYYMMDDTHHMMSSZ` stamp. It stages only six checksum-bound evidence files under `/tmp`, performs one bounded 1 MiB write probe that is fsynced and removed, and submits no SLURM job. Copy the three inputs and their sidecars into this directory before running the local gate.

The pre-smoke snapshot passed with 6,923,721,728 bytes of personal-group soft headroom against the conservatively modeled 6,475,623,256-byte requirement, a margin of 448,098,472 bytes (0.417 GiB). It authorized only the single replacement smoke job after both release `56358f2a4996f654426814d88e622b7f217ac89bfc02f8bbdacc27e66873b55e` and its preflight had been verified.

The post-smoke snapshot passed with 6,807,078,912 bytes of personal-group soft headroom, leaving 331,455,656 bytes (0.309 GiB) above the same modeled requirement. That receipt is now expired and does **not** authorize the full screening. The 150-trial array remains unsubmitted. In a separate authorized window, verify the active marker/preflight, capture and assess another fresh snapshot and write probe, and prefer materially more headroom before submission.
