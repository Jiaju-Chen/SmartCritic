# SearchQA Luna 28-layer unified critic

This is an isolated capacity-control experiment for the existing SearchQA
Luna run. It keeps the actor, data, retriever, reward, advantage composition,
rollout count, optimization settings, validation schedule, and checkpoint
policy unchanged. The only algorithmic/model change is:

```text
critic.model.override_config.num_hidden_layers: 2 -> 28
```

The actor remains Qwen2.5-1.5B-Instruct. The critic still has two value heads
for token-level and turn-level targets, and the actor still consumes the Luna
residual hybrid advantage. This is therefore a critic-capacity ablation, not a
new reward or advantage method.

Expected output directories on 8card:

```text
/home/dataset-local/cjj/RL/runs/luna_unified_searchqa/<run-name>
/home/dataset-local/cjj/RL/checkpoints/luna_unified_searchqa/<run-name>
```

The checkpoint directory uses only `best` and `latest`. Validation and saving
remain every 50 training steps, with `val/success_rate` selecting `best`; the
fixed-budget comparison point is `latest` at step 200.
