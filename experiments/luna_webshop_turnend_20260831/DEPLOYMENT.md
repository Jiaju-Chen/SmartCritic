# Deployment and verification, 2026-08-31

## Locations

- Server: yun-my8card-vscode, 8 NVIDIA A100-SXM4-80GB GPUs.
- Worktree: `/home/dataset-local/cjj/RL/SmartCritic-webshop-turnend`.
- Branch: `luna-webshop-turnend-20260831`.
- Training source commit: `cac14415457aba28e887f5d97b8ac17b79315146`.
- Formal tmux: `luna_webshop_turnend_formal_0831`.
- Formal start: 2026-08-31 21:28:24 +08:00.
- Run: `luna_unified2h_webshop_actionend_t128_v128_8gpu_seed0_20260831`.
- W&B: https://wandb.ai/cjj01-ustc/verl_agent_webshop_critic_ablation/runs/lunawsend0831
- Launch log: `/home/dataset-local/cjj/RL/runs/luna_webshop_turnend/launch_formal_20260831/train.log`.
- Checkpoints: `/home/dataset-local/cjj/RL/checkpoints/luna_webshop_turnend/luna_unified2h_webshop_actionend_t128_v128_8gpu_seed0_20260831`.

## Verification

1. 23 tests passed on the server using the existing WebShop Python environment.
2. First smoke failed before parameter updates on an object-dtype success array
   in the newly added diagnostic. The failing unit test reproduced the exact
   error; explicit boolean conversion fixed it. Original failed log retained.
3. Fresh `smoke2` exited 0 after one actor/critic update and 8-case validation.
   Both best and latest contain 8 nonempty model, optimizer, and extra-state
   shards for each of actor/critic, plus dataloader state. Latest step is 1.
4. Formal training started from pretrained weights, not the smoke checkpoint.
   Its initial validation completed all 128 tasks. Step 1 completed and the
   following iteration is running; no training exception or OOM observed.
5. W&B API confirmed the formal run under `cjj01-ustc`, state `running`.

## Fairness audit

Compared live W&B configs with the original unified residual WebShop run
`lunauniws_0821`. Ignoring output/run paths, the only config differences were:

- New `algorithm.hybrid_advantage.turn_value_position=action_end`.
- Corresponding `critic.turn_value_position=action_end`.
- Explicit `composition_mode=residual`; this was implicit in the old run.

Train and validation parquet tables are elementwise identical (128 rows each,
checked with PyArrow, ignoring file metadata). Final advantage whitening remains
ON. The actor, critic depth, residual coefficient, loss coefficient, optimizer,
batch sizes, token/turn discounts, validation sampling, and save policy match.

## First-step timing

These are diagnostic observations, not a statistically established speed result.

| Component | Original unified Luna, step 1 | Luna-End, step 1 |
| --- | ---: | ---: |
| Rollout | 92.278 s | 94.316 s |
| Value forward | 2.790 s | 2.899 s |
| Critic update | 12.667 s | 12.888 s |
| Actor update | 130.144 s | 130.497 s |
| Total training step | 315.300 s | 318.136 s |

Luna-End step 1: token value loss 5.742, turn value loss 1.308,
raw turn advantage RMS 1.540; values are console-rounded. Initial validation
success is 7/128, before any training, and is not evidence of ablation quality.

The original run's 150 step timers total 46,532 seconds (12.93 hours), including
30 validations. A cautious estimate for this run is 13-17 hours from launch,
approximately September 1, 10:30-14:30 China time, excluding final WebShop-500
evaluation and all SearchQA evaluation. Future trajectory length can change
runtime, so this is not a guaranteed completion time.

The W&B SDK buffers explicit-step history until the next step or `finish()`;
console output through Ray can also lag. The native W&B `files/output.log`
confirmed the complete step-1 metrics while the outer tmux log was still buffered.
Normal training exit calls `logger.finish()`, as verified by smoke2.
