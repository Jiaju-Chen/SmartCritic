# Unified Luna SearchQA smoke

This experiment connects the existing Search-R1 environment to Unified Luna
without changing the critic or advantage implementation.

## Method

- actor: full Qwen2.5-1.5B-Instruct;
- critic: token embedding, first two Qwen Transformer blocks, and two value
  heads;
- token credit: action tokens are chained across search turns;
- turn credit: one value target at the action-start boundary of every turn;
- actor credit: turn advantage plus the within-action centered token residual;
- no gradient-adaptive gate and no additional advantage whitening.

## Smoke assets

The official Search-R1 setup requires the Wikipedia 2018 corpus, E5 index, and
an E5 retrieval model. Those assets are not currently installed on my8card.
The smoke run therefore uses the same SearchEnv and `/retrieve` HTTP contract
with a compact lexical retriever over factual passages. Its only purpose is to
verify the complete rollout, reward, critic, actor, validation, checkpoint, and
W&B path. It is not a benchmark result.

## Commands

Environment-only probe:

```bash
bash experiments/luna_unified_critic_searchqa_20260825/run_env_probe.sh
```

One-step eight-GPU training smoke:

```bash
bash experiments/luna_unified_critic_searchqa_20260825/run_smoke.sh
```

Wait for the current WebShop run to release all GPUs, then launch the smoke:

```bash
bash experiments/luna_unified_critic_searchqa_20260825/wait_for_webshop_then_smoke.sh
```

For a full Search-R1 experiment, use the upstream preprocessing script and E5
retrieval service documented in `docs/UPSTREAM_VERL_AGENT_README.md`, then pass
the resulting files and endpoint through `TRAIN_DATA`, `VAL_DATA`, and
`SEARCH_URL`.
