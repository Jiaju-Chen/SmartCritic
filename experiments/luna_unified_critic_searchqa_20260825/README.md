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

The same smoke can run on one A100 80GB without changing the global training
batch or algorithm:

```bash
NUM_GPUS=1 TENSOR_PARALLEL_SIZE=1 \
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

## Official Search-R1 run

The formal scripts preserve the upstream Search-R1 environment protocol:

- NQ and HotpotQA mixed training data from `PeterJinGo/nq_hotpotqa_train`;
- the upstream multi-dataset test parquet;
- Wikipedia-2018 corpus and the official E5 flat index;
- E5-base-v2 retrieval with `topk=3`;
- four search turns, 4096 prompt tokens, and 512 response tokens;
- batch size 256, validation batch size 512, and one dataset epoch.

Prepare the QA parquet files on the eight-GPU training host:

```bash
bash experiments/luna_unified_critic_searchqa_20260825/prepare_official_data.sh
```

Prepare and start the retriever on a host with at least 160 GB free disk space
and one A100-80GB:

```bash
bash experiments/luna_unified_critic_searchqa_20260825/prepare_official_retriever.sh
bash experiments/luna_unified_critic_searchqa_20260825/run_official_retriever.sh
```

Forward that server to local port 18000, then launch formal training:

```bash
SEARCH_URL=http://127.0.0.1:18000/retrieve \
  bash experiments/luna_unified_critic_searchqa_20260825/run_formal.sh
```

When the retriever runs on `yun-my1card`, the persistent launcher creates the
SSH tunnel, waits for retrieval readiness, and starts training automatically:

```bash
bash experiments/luna_unified_critic_searchqa_20260825/launch_formal_via_my1card.sh
```

The formal Luna run uses five rollouts per question, matching GiGPO's group
size and total sampled environment interactions. Luna does not use group
normalization; the five trajectories are ordinary PPO samples consumed by its
critic. Dataset, retriever, environment horizon, prompts, response limit, and
evaluation data are also aligned with GiGPO. The actor remains
Qwen2.5-1.5B-Instruct to match the existing Luna ablations; model-scale
comparisons against the published 3B/7B Search-R1 table must therefore be
reported separately.
