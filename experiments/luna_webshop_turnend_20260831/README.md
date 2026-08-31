# Luna-End: WebShop turn-value readout ablation

Baseline: unified two-layer Luna, residual actor credit, WebShop 20260821.
Only the turn-value readout changes: prompt boundary -> after the last valid
response token (including EOS if it is part of the existing response mask).
The token head still reads prefix states before each generated token.
Both heads share the pretrained embedding, first two Qwen blocks, and final norm.

The turn GAE uses endpoint values consistently for current and next turns.
It still broadcasts to the SAME current response. Its loss mask and target
move to the last valid response slot. This is intentionally a position
ablation, NOT a Q-minus-V algorithm or an official HyGAE reproduction.
An action-conditioned baseline can cancel useful action credit; diagnose this
risk rather than assuming the endpoint is a valid state-value baseline.

## Fixed training configuration

- Actor: complete Qwen2.5-1.5B-Instruct; critic: shared 2 layers, 2 value heads.
- 8 A100 GPUs; 128 training tasks x 1 rollout; PPO mini-batch 64, micro-batch 1/GPU.
- Token gamma/lambda 1/1; turn gamma/lambda 1/0.95; residual coefficient 1.
- Final mixed-advantage whitening ON, matching the historical WebShop baseline.
- Actor/critic learning rates 1e-6/1e-5; KL loss coefficient 0.01.
- Prompt/response limits 4096/512; 15 environment turns; 2-turn history; seed 0.
- 150 optimizer iterations, 128 monitoring tasks in batches of 16 every 5 steps.
- Validation temperature 0.4, sampling ON; no new full-500 evaluation in training.
- Best/latest slots every 5 steps, best by WebShop task score, not success rate.
- Formal run starts from pretrained weights, not the smoke checkpoint.

## Diagnostics

Existing hybrid metrics remain. `turn_diag/*` weights each action equally:
raw pre-whitening turn-advantage RMS and absolute mean, old value/target MSE,
success/failure-conditioned advantage means and counts. These are descriptive
training diagnostics, not independent estimates of true value accuracy.

## Run

```bash
bash experiments/luna_webshop_turnend_20260831/launch.sh smoke2
bash experiments/luna_webshop_turnend_20260831/launch.sh formal
```

Launch the formal job only after smoke exits zero, saves best/latest, and GPUs
are released. Scripts refuse duplicate launches and keep separate logs and
checkpoint roots. The original repository, runs, and SearchQA tunnel are untouched.

Formal W&B: `cjj01-ustc/verl_agent_webshop_critic_ablation/lunawsend0831`.
Smoke W&B: `cjj01-ustc/verl_agent_webshop_critic_ablation/lunawsendprobe0831`.

The first smoke stopped before updates because its diagnostic did not accept
the rollout's object-dtype outcome array. A regression test reproduces the
failure; explicit boolean conversion fixes it. `smoke2` uses a fresh run,
`lunawsendprobe20831`, keeping the first log intact.
