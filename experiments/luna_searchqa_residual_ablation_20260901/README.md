# SearchQA Luna residual ablation

This experiment is the direct-mix counterpart of the completed two-layer,
whitened SearchQA Luna run `lunauni8white0830`.

## Only algorithmic change

Reference residual composition followed by final whitening:

```text
A_actor = Whiten(A_turn + (A_token - mean_within_action(A_token)))
```

Candidate direct composition followed by the same final whitening:

```text
A_actor = Whiten(A_turn + A_token)
```

Both value heads, both critic targets and losses, token/turn GAE, alpha=1, and
all PPO settings remain unchanged. Final mixed-advantage whitening is enabled
in both runs.

## Fixed protocol

- Actor: Qwen2.5-1.5B-Instruct, full 28 layers.
- Critic: shared first two Qwen layers with token and turn value heads.
- Eight A100 GPUs; 256 tasks x 5 rollouts = 1,280 trajectories per update.
- Token gamma/lambda: 1/1; turn gamma/lambda: 0.95/0.95.
- Actor/critic learning rates: 1e-6/1e-5; KL coefficient: 0.001.
- Fixed source-stratified 512-case monitoring set, greedy validation every 50
  updates.
- 200 updates; only `best` and `latest` checkpoint slots are retained.
- Search top-k: 3; maximum turns: 4; retrieval errors terminate training.
- W&B project: `cjj01-ustc/verl_agent_searchqa_critic_ablation`.

`verify_config.py` compares the resolved candidate configuration with the
completed non-whitened residual reference config. It requires exactly the two
intended algorithmic settings (`composition_mode=direct` and
`whiten_advantages=True`) and refuses any other hyperparameter drift. The
completed whitened residual run was independently audited against the same
reference with whitening as its only algorithmic difference; therefore this
candidate differs from it only in residual versus direct composition.

## Launch

```bash
cd /home/dataset-local/cjj/RL/SmartCritic-searchqa
bash experiments/luna_searchqa_residual_ablation_20260901/launch_tmux.sh
```

The launcher refuses to overwrite an existing log or populated checkpoint
directory. It does not modify or delete any earlier run.
