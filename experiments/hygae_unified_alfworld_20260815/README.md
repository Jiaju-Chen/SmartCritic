# HyGAE Unified Critic on ALFWorld

This experiment implements the core HyGAE estimator on top of the existing
ALFWorld PPO pipeline:

- one two-layer Qwen2.5-1.5B critic;
- skip-observation token GAE;
- turn GAE from the same critic's final valid token in each environment turn;
- `0.5 * turn advantage + 0.5 * token advantage` for the actor;
- the matching mixed return for the unified critic.

The five-step pilot keeps the mature dual-critic experiment's actor, data,
rollout, optimizer, validation, and checkpoint settings. It uses 128 training
cases per update and all 140 in-distribution ALFWorld cases at step 5.

Remote worktree: `/home/dataset-local/cjj/RL/GiGPO_PVF_HYGAE`

Remote branch: `hygae-alfworld-20260815`

Launch with `bash experiments/hygae_unified_alfworld_20260815/launch_pilot5.sh`.

