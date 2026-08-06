# Dual-Critic Hybrid PPO on ALFWorld

This experiment keeps the existing SAO-style skip-observation token critic and
adds an independent turn-boundary critic. The actor advantage is

```text
turn advantage + token residual around the per-turn token mean
```

Both critics use the Qwen2.5-1.5B embedding and first two transformer layers,
followed by an independent scalar value head. Existing PPO and SAO experiment
directories are not modified.

The full run uses 128 training cases per update and evaluates all 140
in-distribution ALFWorld cases every five updates.

Remote branch: `dual-critic-hybrid-alfworld-20260806`

Run the smoke test with `bash launch_smoke.sh`. After it passes, launch the full
run with `bash launch_tmux.sh`.
