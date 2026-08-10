# Full-140 Historical Checkpoint Audit

This experiment evaluates the retained checkpoints from the two-layer PPO and
SAO skip-observation PPO runs on the same 140 ALFWorld in-distribution tasks
used by the dual-critic run.

Checkpoints are evaluated at steps 25, 50, 75, 100, 125, and 150. Evaluation
uses eight GPUs, validation batches of 20, environment seed 0, at most 50
environment steps, temperature 0.4, and sampling enabled. The indexed
validation patch verifies coverage of 140 unique ALFWorld game files.

Run a 20-case smoke test:

```bash
VAL_DATA_SIZE=20 VAL_BATCH_SIZE=20 RUN_TAG=smoke \
  bash experiments/full140_checkpoint_audit_20260810/run_one.sh critic2l 25
```

Launch the full sequential audit:

```bash
bash experiments/full140_checkpoint_audit_20260810/launch_tmux.sh
```

Outputs are written under:

```text
/home/dataset-local/cjj/RL/runs/full140_checkpoint_audit_20260810/
```

`results.csv` and `results.json` are regenerated after every successful
checkpoint, so an interrupted audit retains all completed measurements.
