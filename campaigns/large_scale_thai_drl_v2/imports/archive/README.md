# Archived calibration imports

The V1-to-V2 calibration import was accepted before the V2 dependency
specifications were tightened to exact Optuna and TensorBoard versions.

On 2026-07-16, the migration validator correctly rejected reuse because
`requirements.txt` and `environment.yml` no longer matched the frozen V1
scientific-input inventory. The receipt and its original checksum are retained
here as historical evidence, but they are intentionally outside the active
campaign import path. Fresh native V2 calibration receipts must satisfy the
current exact release and preflight.
