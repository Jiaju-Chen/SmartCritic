# Two-Layer Qwen Critic PPO on ALFWorld

This experiment changes only the PPO critic depth relative to the existing full-Qwen PPO baseline.

- actor: Qwen2.5-1.5B-Instruct
- critic: Qwen2.5 token embedding + first 2 Transformer layers + scalar token-classification head
- advantage estimator: GAE
- train tasks: 128
- validation tasks: 32 seen tasks
- rollout count: 1 per task
- GPUs: 8
- epochs: 150
- validation frequency: 5
- actor learning rate: `1e-6`
- critic learning rate: `1e-5`
- WandB: online with a stable run ID

The critic remains fully trainable. Rewards, PPO clipping, value clipping, GAE, actor optimization, and ALFWorld behavior are unchanged.

Launch:

```bash
cd /home/dataset-local/cjj/RL/GiGPO_PVF
bash experiments/truncated_qwen_critic_alfworld_20260731/launch_tmux.sh
```

One-epoch smoke test:

```bash
RUN_NAME=ppo_qwen25_15b_critic2l_smoke_20260731 \
SESSION=ppo_critic2l_smoke_20260731 \
WANDB_RUN_ID=c2lsmoke0731 \
TRAIN_DATA_SIZE=8 \
VAL_DATA_SIZE=4 \
PPO_MINI_BATCH_SIZE=8 \
TOTAL_EPOCHS=1 \
TEST_FREQ=1 \
SAVE_FREQ=-1 \
bash experiments/truncated_qwen_critic_alfworld_20260731/launch_tmux.sh
```
