# Luna ALFWorld Residual Ablation

This experiment isolates Luna's Actor advantage composition while keeping the
unified two-layer, two-head critic and all other ALFWorld settings fixed.

```text
Luna residual: A_actor(t,i) = A_turn(t) + alpha * (A_token(t,i) - mean_i A_token(t,i))
Direct mix:    A_actor(t,i) = A_turn(t) + alpha * A_token(t,i)
Token only:    A_actor(t,i) = A_token(t,i)
Turn only:     A_actor(t,i) = A_turn(t)
```

Both value heads are still trained with the same token and turn targets. Only
the advantage passed to the Actor changes, with `alpha=1`.

## Protocol

- Actor: Qwen2.5-1.5B-Instruct.
- Critic: Qwen embedding plus the first two Transformer blocks, shared by two
  scalar value heads.
- Training tasks per update: 128 (`env.rollout.n=1`).
- Full seen validation: all 140 indexed ALFWorld tasks, in batches of 20.
- Maximum environment turns: 50.
- Training updates: 150.
- Validation and checkpoint refresh: every 5 updates.
- Checkpoints: named `best` and `latest` slots only.
- Best metric: full `val/success_rate` over 140 tasks.
- W&B finalization is explicit, so the final step is flushed before exit.

## Remote Layout

```text
worktree:    /home/dataset-local/cjj/RL/SmartCritic_ALFWORLD_DIRECT
branch:      luna-alfworld-direct-ablation-20260823
runs:        /home/dataset-local/cjj/RL/runs/luna_alfworld_advantage_ablation
checkpoints: /home/dataset-local/cjj/RL/checkpoints/luna_alfworld_advantage_ablation
```

Run the full experiment with:

```bash
bash experiments/luna_alfworld_advantage_ablation_20260823/launch_full150.sh
```

If the GPUs are occupied by the WebShop ablation, submit it safely with:

```bash
bash experiments/luna_alfworld_advantage_ablation_20260823/launch_after_webshop.sh
```

On SAIDS, `submit_saids.sh` submits a one-GPU environment probe followed by an
eight-GPU A100 training job with an `afterok` dependency:

```bash
bash experiments/luna_alfworld_advantage_ablation_20260823/submit_saids.sh
```

SAIDS records W&B offline because its compute network cannot reach
`api.wandb.ai`. The run directory lives under the shared run root and can be
synced from the eight-GPU server or a connected workstation after training.

After the shared environment probe has passed, submit the token-only and
turn-only jobs independently so Slurm can run them concurrently when two nodes
are available:

```bash
bash experiments/luna_alfworld_advantage_ablation_20260823/submit_saids_credit_only.sh
```

For a fresh cluster deployment, `submit_saids_retry.sh` first runs an exact
one-update integration smoke test and releases all three formal jobs only when
that test succeeds. It reuses prepared indexed parquet files and exposes the
shared filesystem through an ASCII-only path so Hydra never receives the
Chinese group-directory component:

```bash
bash experiments/luna_alfworld_advantage_ablation_20260823/submit_saids_retry.sh
```

`submit_saids_a800_smoke.sh` tests the same global training semantics on four
A800 GPUs. It keeps the 128-task rollout batch, PPO mini-batch 256, micro-batch
one per GPU, and full seen-140 validation unchanged; only the FSDP world size
and resource-only Ray CPU allocation differ:

```bash
bash experiments/luna_alfworld_advantage_ablation_20260823/submit_saids_a800_smoke.sh
```
