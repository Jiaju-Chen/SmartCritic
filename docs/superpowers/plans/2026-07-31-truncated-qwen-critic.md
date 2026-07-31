# Truncated Qwen Critic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a fair ALFWorld PPO experiment whose critic uses the Qwen2.5-1.5B token embedding, the first two Transformer layers, final normalization, and one scalar token-classification head.

**Architecture:** Reuse the existing Hugging Face `AutoModelForTokenClassification` critic and make the existing `critic.model.override_config` interface effective. The experiment sets only `num_hidden_layers=2`; actor, rewards, generalized advantage estimation, PPO losses, validation, and rollout settings remain identical to the existing full-critic PPO baseline.

**Tech Stack:** Python 3.10, PyTorch, Hugging Face Transformers, FSDP, Hydra/OmegaConf, Ray, verl, ALFWorld, WandB.

---

### Task 1: Apply Critic Model Overrides

**Files:**
- Modify: `verl/workers/fsdp_workers.py`
- Test: `tests/workers/critic/test_critic_override.py`

- [ ] **Step 1: Write the failing test**

Create a source-level regression test that locates the critic-specific `AutoConfig.from_pretrained` call and verifies that `update_model_config(critic_model_config, override_config_kwargs=override_config_kwargs)` is called before `AutoModelForTokenClassification.from_pretrained`.

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
/home/dataset-local/cjj/RL/envs/gigpo-baselines/bin/python tests/workers/critic/test_critic_override.py
```

Expected: `FAIL`, because the critic currently prints overrides without applying them.

- [ ] **Step 3: Apply the override through the existing helper**

Import `update_model_config` next to `print_model_size`, then call:

```python
update_model_config(critic_model_config, override_config_kwargs=override_config_kwargs)
```

immediately after loading the critic config and before model construction. Log the resulting layer count on rank zero.

- [ ] **Step 4: Run the regression test**

Run the same command. Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add verl/workers/fsdp_workers.py tests/workers/critic/test_critic_override.py
git commit -m "fix: apply critic model config overrides"
```

### Task 2: Verify Two-Layer Model Construction

**Files:**
- Create: `tests/workers/critic/smoke_truncated_qwen_critic.py`

- [ ] **Step 1: Load the local Qwen config and apply the experiment override**

Use the local Qwen2.5-1.5B snapshot, set `num_labels=1`, apply `num_hidden_layers=2`, and instantiate `AutoModelForTokenClassification` in bfloat16.

- [ ] **Step 2: Assert the architecture**

Verify:

```python
assert len(model.model.layers) == 2
assert model.score.out_features == 1
```

Also print total parameters and the ratio relative to the 28-layer baseline.

- [ ] **Step 3: Run one forward pass**

Tokenize a short ALFWorld-style observation and assert logits have shape `(1, sequence_length, 1)`.

- [ ] **Step 4: Commit**

```bash
git add tests/workers/critic/smoke_truncated_qwen_critic.py
git commit -m "test: verify two-layer qwen critic construction"
```

### Task 3: Add Reproducible Experiment Launchers

**Files:**
- Create: `experiments/truncated_qwen_critic_alfworld_20260731/README.md`
- Create: `experiments/truncated_qwen_critic_alfworld_20260731/run_train.sh`
- Create: `experiments/truncated_qwen_critic_alfworld_20260731/tmux_entry.sh`
- Create: `experiments/truncated_qwen_critic_alfworld_20260731/launch_tmux.sh`

- [ ] **Step 1: Encode the fair PPO baseline settings**

Set Qwen2.5-1.5B, train size 128, validation size 32, rollout count 1, eight GPUs, 150 epochs, test frequency 5, actor learning rate `1e-6`, critic learning rate `1e-5`, and `algorithm.adv_estimator=gae`.

- [ ] **Step 2: Enable only the two-layer critic difference**

Pass:

```bash
+critic.model.override_config.num_hidden_layers=2
```

Do not alter PPO value loss, generalized advantage estimation, rewards, actor loss, or environment behavior.

- [ ] **Step 3: Configure persistent outputs and WandB**

Use separate run, checkpoint, temporary, and WandB directories. Set `WANDB_MODE=online`, a stable `WANDB_RUN_ID`, and `WANDB_RESUME=allow`.

- [ ] **Step 4: Validate shell syntax and Hydra composition**

Run `bash -n` on all scripts, then a one-epoch smoke run with train size 8, validation size 4, and mini-batch size 8.

- [ ] **Step 5: Commit**

```bash
git add experiments/truncated_qwen_critic_alfworld_20260731
git commit -m "exp: add two-layer qwen critic ppo run"
```

### Task 4: Launch and Monitor the Full Experiment

**Files:**
- Runtime log: `/home/dataset-local/cjj/RL/runs/truncated_qwen_critic_alfworld/<run-name>/logs/train.tmux.log`
- Checkpoints: `/home/dataset-local/cjj/RL/checkpoints/truncated_qwen_critic_alfworld/<run-name>`

- [ ] **Step 1: Launch the full 150-epoch run in tmux**

Use a unique tmux session, run name, and stable WandB run ID.

- [ ] **Step 2: Verify initialization**

Confirm the log reports `num_hidden_layers: 2`, the critic parameter count, Ray startup, ALFWorld workers, and online WandB initialization.

- [ ] **Step 3: Verify resource health**

Confirm all eight GPUs are occupied without out-of-memory or Ray resource errors.

- [ ] **Step 4: Verify training has begun**

Confirm rollout generation or the first PPO step is active, then report the session, log, checkpoint directory, WandB run, and exact hyperparameters.
