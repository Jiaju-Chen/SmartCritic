# Full-Critic Dual-Critic Ablation on ALFWorld

This is a controlled ablation of
`experiments/dual_critic_hybrid_alfworld_20260806`.

The actor, skip-observation token GAE, turn-level GAE, residual fusion, reward,
data, optimizer hyperparameters, validation protocol, and seed are unchanged.
The only model change is that both critics use the original PPO critic:
the complete 28-layer Qwen2.5-1.5B token-classification model with one scalar
value head. No `num_hidden_layers` override is passed.

The formal run trains on 128 ALFWorld cases per update for 150 updates. It
evaluates all 140 in-distribution cases every five updates in batches of 20 and
keeps at most two checkpoint slots: `best` and `latest`.

Run `bash launch_smoke.sh` first. After it completes one actor update and both
critic updates without an out-of-memory error, run `bash launch_tmux.sh`.
