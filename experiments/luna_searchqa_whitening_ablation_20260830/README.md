# SearchQA Luna Unified: final-advantage whitening ablation

Reference run: `lunauni8formal0830`, completed at update 200 with exit status 0.
Candidate run: `lunauni8white0830`. This is a fresh training run, not a resume.

## Only algorithmic change

`algorithm.hybrid_advantage.whiten_advantages=False -> True`.
The existing implementation whitens the final mixed actor advantages across all
valid generated tokens in the rollout batch. Token and turn critic targets,
within-action token residuals, and alpha=1 are unchanged. No training source file
is modified by this experiment.

## Fixed settings

- Actor: Qwen2.5-1.5B-Instruct, full 28 layers.
- Critic: shared first two Qwen layers and two value heads.
- Eight A100 GPUs, 256 tasks x 5 rollouts = 1,280 trajectories per update.
- Token gamma/lambda: 1/1. Turn gamma/lambda: 0.95/0.95.
- Actor/critic learning rates: 1e-6/1e-5; actor warmup ratio 0.1.
- PPO minibatch 512; microbatch 4 per GPU. KL coefficient 0.001.
- Maximum prompt/response lengths 4096/512; maximum environment turns 4.
- Same retriever URL, top-3 retrieval, timeout 30 seconds, fail-on-error enabled.
- Same fixed stratified 512-case parquet, validation every 50 updates, greedy
  validation. These are monitoring results, not the 51,713-case final benchmark.
- 200 updates, best/latest checkpoint slots, checkpointing every 50 updates.
- W&B entity/project: cjj01-ustc/verl_agent_searchqa_critic_ablation.

`reference_config.json` is the completed run's W&B configuration, extracted on
2026-08-30. `verify_config.py` resolves the candidate Hydra configuration and
rejects any differences except whitening and experiment/output/Ray identifiers.
Two scheduler fields populated by the trainer at runtime are normalized before
comparison. Missing keys also count as differences.

## Start on the eight-GPU server

```bash
cd /home/dataset-local/cjj/RL/SmartCritic-searchqa
bash experiments/luna_searchqa_whitening_ablation_20260830/launch_tmux.sh
```

The launcher refuses existing logs, populated checkpoint directories, duplicate
tmux sessions, and GPUs occupied by compute processes. Logs and exit status are
written to the new run directory. The existing retriever tunnel supervisor is
reused and not restarted. No older experiment, log, or checkpoint is deleted.

Existing W&B diagnostics include final actor-advantage mean/min/max, raw turn
advantage mean/RMS, token residual RMS, actor KL/clip fraction/gradient norm, critic
value losses and explained variance. They support initial scale/stability checks;
raw mixed-advantage standard deviation and sign-flip fraction are not added here
to keep the training code identical to the reference.

Compare update-matched monitoring curves and the predetermined update-200 endpoint.
Do not attribute a difference between this run and an adaptive-alpha run solely
to whitening. Audit the deployment source hashes and data files before launch.

## Deployment audit

Started in tmux `searchqa_luna2l_whiten_20260830` on 2026-08-30. The preflight
comparison passed with exactly four changed fields: whitening, experiment name,
checkpoint directory, and Ray temporary directory. `deployment_record.json`
records the deployed experiment commit, runtime code hashes, and input file hashes.
The remote checkout retains its pre-existing search-transport working-tree patch;
that file and all algorithm files were deliberately not overwritten.

Reference monitoring: update 150 = 201/512 (39.2578125%), update 200 = 199/512
(38.8671875%). Reference best/latest slots are updates 150/200. All 200 updates
completed; summed step timings were 20.58 hours. No timeout, connection-error,
exhausted-search-retry, traceback, or OOM strings were found in its training log.
This log check is not a guarantee about every internal request at the retriever.
