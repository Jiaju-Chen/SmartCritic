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

Launch the 150-step run with
`bash experiments/hygae_unified_alfworld_20260815/launch_full150.sh`.

## Pilot Result

- Run: `ppo_qwen25_15b_hygae_unified_critic2l_t128_v140_vb20_8gpu_seed0_pilot5_20260815`
- W&B: `https://wandb.ai/cjj01-ustc/verl_agent_alfworld_critic_ablation/runs/hygae2lp5_0815`
- Status: completed normally (`status=0`)
- Wall time: about 1 hour 30 minutes, including the full 140-case validation
- Step-5 validation success: `10 / 140 = 0.0714286`
- Step-5 validation score: `0.2869097`
- Validation coverage: 140 unique indexed ALFWorld gamefiles
- Checkpoints: exactly `best` and `latest`; both point to step 5 in this pilot

Training rollout success rates for steps 1-5 were `0.0391`, `0.0781`,
`0.1094`, `0.0469`, and `0.0625`. These five updates establish execution
correctness and numerical stability; they are not enough to compare final
sample efficiency or policy quality with the dual-critic method.
