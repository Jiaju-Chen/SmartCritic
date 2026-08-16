# Luna Unified Critic

This experiment keeps Luna's actor advantage unchanged while replacing its two
independent critics with one shared two-layer Qwen critic and two scalar heads.

For the hidden state before action token `i` in turn `t`:

```text
h(t,i)       = shared_qwen_2l(prompt_t, action_t,<i)
V_token(t,i) = w_token^T h(t,i)
V_turn(t)    = w_turn^T h(t,1)
```

The token head uses cross-turn action-token GAE. The turn head uses state-boundary
turn GAE. The actor still receives Luna's zero-mean residual decomposition:

```text
A_luna(t,i) = A_turn(t) + beta * (A_token(t,i) - mean_i A_token(t,i))
```

The shared critic is optimized once per PPO update:

```text
L_critic = L_token + eta * L_turn
```

Defaults are `beta=1` and `eta=1`. Both heads are rows of the same
`AutoModelForTokenClassification` output matrix, so there is one Transformer
forward, one optimizer, and one critic checkpoint.

Remote worktree:

```text
/home/dataset-local/cjj/RL/GiGPO_PVF_LUNA_UNIFIED
branch: luna-unified-critic-20260816
```

The 5-step smoke command is:

```bash
bash experiments/luna_unified_critic_alfworld_20260816/launch_smoke5.sh
```

The 150-step command is:

```bash
bash experiments/luna_unified_critic_alfworld_20260816/launch_full150.sh
```

Both use train size 128, full seen validation size 140 with validation batch
size 20, eight GPUs, and `best`/`latest` checkpoint slots.
