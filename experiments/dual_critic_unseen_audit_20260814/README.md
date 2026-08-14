# ALFWorld Dual-Critic Unseen Audit

This audit evaluates the retained `best` and `latest` actor checkpoints from
the two ALFWorld dual-critic runs on all 134 out-of-distribution tasks.

The four evaluations are:

| Method | Slot | Training step |
|---|---|---:|
| 2+2-layer dual critic | best | 145 |
| 2+2-layer dual critic | latest | 150 |
| 28+28-layer dual critic | best | 145 |
| 28+28-layer dual critic | latest | 150 |

Evaluation is actor-only: critics are not needed to execute a frozen policy.
All four runs therefore use the same evaluation process and differ only in the
actor checkpoint loaded. The source checkpoints are never modified. A
read-only symbolic link named `global_step_<step>` is created inside each
evaluation directory because the trainer's explicit resume interface requires
that checkpoint naming convention.

Protocol:

- evaluation split: `eval_out_of_distribution`
- cases: 134
- validation batch size: 20
- environment seed: 0
- maximum environment steps: 50
- temperature: 0.4
- sampling: enabled
- GPUs: 8

Launch the sequential queue with:

```bash
bash experiments/dual_critic_unseen_audit_20260814/launch_tmux.sh
```

Outputs are written to:

```text
/home/dataset-local/cjj/RL/runs/dual_critic_unseen_audit_20260814
```

`results.csv` and `results.json` are regenerated after every completed run.
