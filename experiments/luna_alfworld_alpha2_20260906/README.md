# ALFWorld Luna alpha=2 matched run

This is a one-variable comparison against `lunauni2150_0820`, the Qwen2.5-1.5B
shared two-layer dual-head residual Luna run that reached 90.0% on seen-140 at
step 150 and 82.09% on unseen-134.

The intended change is only:

```text
algorithm.hybrid_advantage.token_residual_scale: 1.0 -> 2.0
```

The formal run preserves residual composition, final advantage whitening,
token gamma/lambda 1.0/1.0, turn gamma/lambda 1.0/0.95, train size 128,
seen validation size 140, validation batch size 20, 150 updates, validation and
checkpoint frequency 5, seed 0, actor LR 1e-6, critic LR 1e-5, and the shared
two-layer/two-value-head critic. W&B is recorded offline on lzj.

`run_sequence_lzj.sh` first executes a two-update smoke with separate output and
checkpoint paths. The formal run starts only if the smoke exits successfully.
