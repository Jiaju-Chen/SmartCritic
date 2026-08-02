# Unseen Evaluation

Both checkpoints were evaluated on the 134-task ALFWorld
`eval_out_of_distribution` split with eight GPUs, seed 0, a maximum of 50
environment steps, sampling enabled, and validation temperature 0.4.

| PPO critic | Checkpoint | Success rate | Test score |
| --- | ---: | ---: | ---: |
| Full Qwen2.5-1.5B critic | 150 | 0.4552238806 | 2.3351532969 |
| Two-layer Qwen critic | 150 | 0.5671641791 | 3.3409873708 |

The two-layer result is 76/134 successes versus 61/134 for the full critic, a
gain of 15 tasks or 11.19 percentage points. This is a single stochastic
evaluation. Treat the performance difference as promising rather than
conclusive until it is repeated with multiple evaluation seeds.

## Training Cost Through Step 150

| Metric | Full critic | Two-layer critic | Change |
| --- | ---: | ---: | ---: |
| Critic parameters | 1.54B | 326.97M | -78.77% |
| Wall-clock training time | 42:50:52 | 32:00:55 | -25.28% |
| Mean value forward time/step | 51.54 s | 5.16 s | -89.99% |
| Mean critic update time/step | 240.76 s | 26.88 s | -88.83% |
| Mean non-validation step time | 978.10 s | 711.50 s | -27.26% |
| Peak allocated GPU memory | 37.20 GB | 35.50 GB | -4.57% |

The causal compute comparison is the value forward and critic update cost.
Generation, actor update, and validation timings also depend on policy behavior,
trajectory lengths, and system conditions.

## Artifacts

- Two-layer evaluation log:
  `/home/dataset-local/cjj/RL/runs/truncated_qwen_critic_alfworld/ppo_qwen25_15b_critic2l_step150_unseen134_20260802/logs/eval.tmux.log`
- Full-critic baseline evaluation log:
  `/home/dataset-local/cjj/RL/runs/baselines_alfworld_qwen25_15b_paper/ppo_eval150_unseen134_tmux/logs/eval.tmux.log`
