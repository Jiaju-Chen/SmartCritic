# Luna ALFWorld Residual Ablation

This experiment removes Luna's action-centered token residual while keeping
the unified two-layer, two-head critic and all other ALFWorld settings fixed.

```text
Luna residual: A_actor(t,i) = A_turn(t) + alpha * (A_token(t,i) - mean_i A_token(t,i))
Direct mix:    A_actor(t,i) = A_turn(t) + alpha * A_token(t,i)
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
