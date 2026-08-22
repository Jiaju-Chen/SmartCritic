# Repository Lineage and Algorithm Boundary

## Short Answer

SmartCritic uses GiGPO's released `verl-agent` repository as its engineering
foundation, but the current SmartCritic method is not GiGPO. It is a PPO
actor-critic method that changes the critic and the advantage construction.

## What Is Inherited

The following components come from the upstream verl-agent/GiGPO codebase:

- veRL-based distributed training and configuration;
- Ray workers and FSDP model sharding;
- vLLM rollout generation and actor weight synchronization;
- multi-turn agent rollout orchestration;
- ALFWorld and WebShop environment integration;
- PPO, GRPO, and GiGPO baseline trainers and recipes;
- checkpointing, logging, and W&B integration.

These components are infrastructure and baseline implementations. Keeping their
history makes upstream fixes auditable and avoids presenting inherited code as a
new contribution.

## What SmartCritic Changes

SmartCritic's research changes are concentrated in the PPO value and advantage
path:

- a Qwen critic truncated to the embedding plus the first two Transformer blocks;
- token values computed along the action-token sequence across environment turns;
- turn-boundary values for macro-action credit assignment;
- residual fusion of token and turn advantages;
- separate dual-critic and shared-backbone dual-head variants;
- ALFWorld and WebShop evaluation/indexing utilities and ablations.

The actor remains a full Qwen2.5-1.5B language model in the current experiments.
The environment reward definition and PPO clipped policy objective are not
replaced by GiGPO's group-in-group objective.

## Difference from GiGPO

GiGPO obtains additional credit signals by grouping matching states within the
current rollout batch and comparing returns inside these state-conditioned
groups. Its core question is how to exploit repeated states and trajectory
groups without a learned critic.

SmartCritic instead asks how to learn a compact value estimator that represents
both local token decisions and long-horizon turn progress. Its core object is a
learned critic, and its current objective is PPO. Therefore:

| Dimension | GiGPO | SmartCritic |
| --- | --- | --- |
| Base optimizer | Group-relative policy optimization | PPO actor-critic |
| Learned critic | Not the defining component | Central component |
| Credit source | Outcome groups and matching states | Learned token and turn values |
| Granularity | Trajectory/state-group comparisons | Token and environment-turn timescales |
| Efficiency target | Better use of grouped rollouts | Smaller and shared critic computation |

## Git Lineage

The repository intentionally retains the upstream commit history. The main
SmartCritic sequence is:

```text
verl-agent / GiGPO base
  -> progress-value PPO exploration
  -> two-layer Qwen critic PPO
  -> cross-turn action-token chain
  -> token/turn dual critic with residual fusion
  -> shared two-layer dual-head critic
  -> ALFWorld and WebShop ablations
```

Reproduction-only branches, including the HyGAE diagnostic, remain separate from
the SmartCritic main line.

## Attribution Rule

Papers and documentation should use wording equivalent to:

> We implement SmartCritic on top of the open-source verl-agent framework
> released with GiGPO, which is built on veRL. We retain its multi-turn rollout
> and distributed PPO infrastructure, while replacing the critic architecture
> and multi-timescale advantage estimator.

Do not describe the inherited environment, distributed trainer, or baseline
implementations as SmartCritic contributions.

