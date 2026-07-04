# Progress-Value PPO on ALFWorld

First-version test of replacing PPO's learned critic with a lightweight progress-value advantage estimator.

Hypothesis: in long-horizon sparse-reward ALFWorld, an action-level value target based on final success reward, trajectory length, and remaining steps can provide a denser and cheaper advantage signal than PPO's scalar value head.

Default run:

- model: Qwen2.5-1.5B-Instruct local snapshot
- train tasks: 16
- rollout group: 8 per train task
- total train env batch: 128
- validation tasks: 140 seen tasks
- GPUs: 8
- epochs: 150
- advantage estimator: `progress_value`
- critic worker: disabled by estimator selection

Launch:

```bash
cd /home/dataset-local/cjj/RL/GiGPO_PVF
bash experiments/progress_value_ppo_alfworld_20260704/launch_tmux.sh
```

Important paths are printed by `launch_tmux.sh`. The training log is `$RUN_DIR/logs/train.tmux.log`.
