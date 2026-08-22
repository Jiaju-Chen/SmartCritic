# Luna WebShop Advantage Ablation

This isolated experiment compares four actor-credit formulas while keeping the
shared two-layer, two-head critic and all optimization settings fixed.

```text
residual:   A_turn + alpha * (A_token - action_mean(A_token))
direct:     A_turn + alpha * A_token
token_only: A_token
turn_only:  A_turn broadcast to every action token
```

The completed unified Luna run is the `residual` baseline with `alpha=1`. This
directory runs the remaining modes in the order `direct`, `token_only`, and
`turn_only`. Both critic heads continue to train in every mode so that the only
experimental variable is the advantage supplied to the Actor.

## Protocol

- Actor: Qwen2.5-1.5B-Instruct.
- Critic: Qwen embedding plus the first two Transformer blocks, shared by two
  scalar value heads.
- Training interactions per update: 128 (`env.rollout.n=1`).
- Fixed monitoring evaluation: 128 goals from the synthetic small-product
  WebShop test pool `[0, 500)`, evaluated in batches of 16.
- Maximum environment turns: 15.
- Training updates: 150.
- Validation and checkpoint refresh: every 5 updates.
- Checkpoints: named `best` and `latest` slots only.
- `best` metric: continuous WebShop task score.
- Actor loss aggregation remains `token-mean` to match the completed residual
  baseline. Action-mean aggregation is a separate ablation.

## Remote Layout

```text
worktree:    /home/dataset-local/cjj/RL/GiGPO_PVF_LUNA_WEBSHOP_ABLATIONS
branch:      luna-webshop-advantage-ablation-20260822
runs:        /home/dataset-local/cjj/RL/runs/luna_webshop_advantage_ablation
checkpoints: /home/dataset-local/cjj/RL/checkpoints/luna_webshop_advantage_ablation
```

Run `launch_smoke_direct.sh` for a one-update integration test. After it passes,
`launch_sequence.sh` runs the three formal experiments sequentially on eight
GPUs.
