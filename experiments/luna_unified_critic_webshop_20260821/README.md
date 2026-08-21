# Unified Luna Critic on WebShop

This experiment ports the validated unified Luna critic to WebShop without
changing Luna's advantage construction:

```text
shared Qwen2.5 2-layer critic
  |- token value head: cross-turn response-token GAE
  `- turn value head: pre-response state-boundary GAE

A_luna(t, i) = A_turn(t) + beta * (A_token(t, i) - mean_i A_token(t, i))
```

The comparison is controlled against the previous WebShop `2 + 2` dual-critic
run. Actor, reward, training size, validation stream, seed, optimizer settings,
and training horizon are unchanged. The only algorithmic/model difference is
that the two critics are replaced by one shared two-layer critic with two scalar
heads.

## Protocol

- Actor: Qwen2.5-1.5B-Instruct.
- Critic: Qwen embedding plus the first two Transformer blocks, shared by two
  scalar value heads.
- Training interactions per update: 128 (`env.rollout.n=1`).
- Fixed monitoring validation: 128 WebShop goals, evaluated in batches of 16.
- Maximum environment turns: 15.
- Training updates: 150.
- Validation and checkpoint refresh: every 5 updates.
- Checkpoints: named `best` and `latest` slots only.
- `best` metric: continuous WebShop task score.

## Remote Layout

```text
worktree:    /home/dataset-local/cjj/RL/GiGPO_PVF_LUNA_WEBSHOP
branch:      luna-unified-webshop-20260821
runs:        /home/dataset-local/cjj/RL/runs/luna_unified_webshop
checkpoints: /home/dataset-local/cjj/RL/checkpoints/luna_unified_webshop
```

Run the one-update integration test with `launch_smoke1.sh`. After it succeeds,
start the formal run with `launch_full150.sh`.
