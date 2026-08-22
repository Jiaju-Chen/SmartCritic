# SmartCritic

SmartCritic is a research codebase for building **accurate and efficient critics
for long-horizon language agents**. The current implementation studies how PPO
can assign credit at both token and turn timescales without requiring a second
full-size language model as the critic.

The repository currently contains ALFWorld and WebShop experiments with
Qwen2.5-1.5B actors, lightweight critics, turn-aware advantage estimation, and
shared-backbone dual-head critics.

## Relationship to GiGPO

SmartCritic is **built on the `verl-agent` codebase released with GiGPO**, which
is itself built on veRL. We reuse its multi-turn rollout system, environment
integration, Ray/FSDP/vLLM workers, PPO trainer, and experiment infrastructure.

SmartCritic is **not an implementation of the GiGPO optimization algorithm**.
The current method remains actor-critic PPO and modifies the critic architecture
and credit-assignment path. In particular, it does not use GiGPO's group-in-group
advantage as its defining objective.

See [Repository Lineage](docs/REPOSITORY_LINEAGE.md) for the exact boundary and
[the archived upstream README](docs/UPSTREAM_VERL_AGENT_README.md) for the
original GiGPO/verl-agent documentation.

## Current Method

The current SmartCritic line, previously called **Luna** internally, combines:

1. **Token-scale credit:** a value is estimated for each generated action token.
2. **Turn-scale credit:** one value estimates the consequence of the complete
   environment action at a turn boundary.
3. **Residual fusion:** the turn advantage carries cross-turn progress, while a
   centered token residual distinguishes tokens within the action.
4. **Lightweight critic:** Qwen token embeddings and the first two Transformer
   blocks are reused, followed by value heads. This preserves language-aware
   representations while removing most critic layers.
5. **Unified critic:** the latest variant shares one two-layer text backbone and
   uses separate token-value and turn-value heads, so the text is encoded once.

For action token `j` in turn `t`, the residual hybrid advantage is

```math
A_{t,j}^{\mathrm{smart}}
= A_t^{\mathrm{turn}}
+ \alpha\left(
    A_{t,j}^{\mathrm{token}}
    - \frac{1}{|a_t|}\sum_{u \in a_t} A_{t,u}^{\mathrm{token}}
  \right).
```

The centering term prevents the token branch from simply duplicating the
turn-level signal. Whether a fixed or adaptive `alpha` is best remains an active
experimental question.

## Research Questions

- Does turn-aware credit improve PPO on sparse-reward, long-horizon agents?
- Is the token residual necessary once turn-level credit is available?
- Can a two-layer critic match or outperform a full 28-layer critic?
- Can a shared dual-head critic retain the benefit of two timescales while
  reducing training time?
- How should token and turn advantages be balanced across environments?

## Repository Layout

| Path | Purpose |
| --- | --- |
| `agent_system/` | Multi-turn rollout and environment interfaces inherited from verl-agent |
| `verl/` | PPO trainer, workers, value/advantage computation, and SmartCritic changes |
| `experiments/` | Reproducible launch scripts and experiment-specific notes |
| `tests/` | Focused tests for truncated, dual, and unified critics |
| `recipe/` and `gigpo/` | Preserved upstream algorithms and baselines |
| `docs/` | Method lineage, development rules, and upstream documentation |

## Important Branches

| Branch | Scope |
| --- | --- |
| `main` | Latest integrated SmartCritic code and WebShop advantage ablations |
| `structured-critic-ppo` | Two-layer Qwen critic PPO baseline |
| `sao-skip-observation-alfworld-20260804` | Cross-turn action-token value chain |
| `dual-critic-hybrid-alfworld-20260806` | Separate token and turn critics with residual fusion |
| `luna-unified-critic-20260816` | Shared two-layer backbone with two value heads |
| `luna-unified-webshop-20260821` | Unified critic port to WebShop |
| `luna-webshop-advantage-ablation-20260822` | Residual/direct/token-only/turn-only ablations |
| `hygae-turn-end-fix-20260817` | HyGAE turn-end reproduction and diagnostic fix |

Each major experiment remains on a named branch so completed runs can be traced
to their exact implementation. New work should use a new branch or Git worktree
instead of overwriting a completed experiment directory.

## Data and Checkpoints

Model weights, datasets, environments, W&B state, logs, and checkpoints are not
stored in Git. Keep them in server-local storage and record their paths in the
run configuration. See [Development and Synchronization](docs/DEVELOPMENT.md)
for the two-server workflow.

## License and Attribution

This repository retains the upstream Apache License 2.0 and attribution files.
SmartCritic modifications are developed on top of:

- [GiGPO / verl-agent](https://github.com/langfengQ/verl-agent)
- [veRL](https://github.com/volcengine/verl)

If you use the inherited GiGPO implementation, cite the upstream GiGPO work as
described in the [original README](docs/UPSTREAM_VERL_AGENT_README.md).
