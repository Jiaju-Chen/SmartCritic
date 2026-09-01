# SearchQA Full Evaluation (2026-09-01)

This directory evaluates the two completed 2-layer unified Luna runs on the
official processed SearchQA test set. The evaluations are intentionally run
sequentially so that both checkpoints see the same retriever capacity.

## Checkpoints

- `no_whiten`: step-200 latest checkpoint from `lunauni8formal0830`
- `whiten`: step-200 latest checkpoint from `lunauni8white0830`

## Protocol

- Test set: `searchR1_official/processed/test.parquet` (51,713 cases)
- Validation temperature: 0
- Validation trajectories per case: 1
- Maximum turns: 4
- Retriever: top-3, 30-second timeout, fail on any retrieval error
- GPUs: 8
- W&B project: `verl_agent_searchqa_full_eval`

The evaluation checkpoints are read-only. Output, logs, and W&B files are
written below `/home/dataset-local/cjj/RL/runs/luna_searchqa_full_eval_20260901`.

## Launch

```bash
bash experiments/luna_searchqa_full_eval_20260901/launch_tmux.sh
```

The launcher starts `searchqa_full51713_both_0901`; `no_whiten` runs first and
`whiten` starts only after the first evaluation exits successfully.
