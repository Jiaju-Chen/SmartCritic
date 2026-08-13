# Dual-Critic PPO on WebShop

This directory defines the controlled WebShop comparison between two critic
depths:

- `2 + 2`: a two-layer token critic and a two-layer turn-boundary critic.
- `28 + 28`: two full Qwen2.5-1.5B critics.

The actor, reward, dual-credit estimator, optimizer settings, data sizes,
validation stream, seed, and checkpoint policy are held fixed. Only
`critic.model.override_config.num_hidden_layers` and
`turn_critic.model.override_config.num_hidden_layers` change.

## Protocol

- Actor: Qwen2.5-1.5B-Instruct.
- Training interactions per update: 128 (`env.rollout.n=1`).
- Official monitoring validation: 128 WebShop interactions every 5 updates.
- WebShop validation pool: goal indexes `[0, 500)`.
- Maximum environment turns: 15.
- Training updates: 150.
- Checkpoints: two named slots only, `best` and `latest`.
- `latest` is refreshed every 5 updates.
- `best` uses the continuous WebShop task score, not exact-match success.
- A separate fixed 500-goal evaluation is planned for `best` and `latest` and
  does not participate in checkpoint selection.

The standard WebShop resources are reused read-only from
`/home/dataset-local/xiax/verl-agent`. `prepare_resources.sh` creates only
symbolic links inside the isolated worktree; it does not copy or edit those
resources.

## Remote layout

```text
worktree:   /home/dataset-local/cjj/RL/GiGPO_PVF_WebShop
branch:     dual-critic-webshop-20260813
runs:       /home/dataset-local/cjj/RL/runs/dual_critic_webshop
checkpoints:/home/dataset-local/cjj/RL/checkpoints/dual_critic_webshop
```

Run `bash prepare_resources.sh`, then `bash probe_webshop_env.sh`. Use
`bash launch_smoke.sh` for the one-update integration test. After it passes,
start the formal lightweight run with `bash launch_two_layer.sh`.
