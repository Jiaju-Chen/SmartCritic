# ALFWorld Progress-Abstraction Diagnostics

Date: 2026-06-29/30

This directory contains isolated research diagnostics for progress-equivalent state/action abstractions on ALFWorld. It does not change GraphGPO's default algorithm. A small optional validation trajectory dump hook was added to the remote GraphGPO trainer and is only active when `+trainer.val_trajectory_dump_dir=...` is passed.

## Files

- `analyze_alfworld_structure.py`: offline expert-trajectory analysis over ALFWorld `valid_seen` and `valid_unseen`.
- `analyze_rollout_dump.py`: analysis for GraphGPO step-level rollout dumps.
- `graphgpo_ray_trainer.remote_current.py`: remote trainer before the optional dump hook.
- `graphgpo_ray_trainer.remote_before_dump.py`: remote trainer with the optional dump hook.
- Remote result: `/home/dataset-local/cjj/RL/GiGPO/experiments/alfworld_progress_abstraction_20260629/results_valid_seen_unseen.json`.
- Remote result: `/home/dataset-local/cjj/RL/GiGPO/experiments/alfworld_progress_abstraction_20260629/results_rollout_step140_unseen20.json`.
- Remote rollout dump: `/home/dataset-local/cjj/RL/GiGPO/experiments/alfworld_progress_abstraction_20260629/traj_dump_step140_unseen20/step_140_batch_0.jsonl`.

## Commands Run

Expert trajectory structure:

```bash
cd /home/dataset-local/cjj/RL/GiGPO
/home/dataset-local/conda/envs/verl/bin/python experiments/alfworld_progress_abstraction_20260629/analyze_alfworld_structure.py \
  --alfworld-root /home/dataset-local/cjj/RL/alfworld_data \
  --splits valid_seen valid_unseen \
  --eval-log /home/dataset-local/cjj/RL/runs/baselines_alfworld_qwen25_15b_paper/graphgpo_eval140_unseen134_tmux/logs/eval.tmux.log \
  --eval-log /home/dataset-local/cjj/RL/runs/baselines_alfworld_qwen25_15b_paper/grpo_eval150_unseen134_tmux/logs/eval.tmux.log \
  --eval-log /home/dataset-local/cjj/RL/runs/baselines_alfworld_qwen25_15b_paper/ppo_eval150_unseen134_tmux/logs/eval.tmux.log \
  --out experiments/alfworld_progress_abstraction_20260629/results_valid_seen_unseen.json
```

GraphGPO rollout dump sample:

```bash
cd /home/dataset-local/cjj/RL/GiGPO
experiments/alfworld_progress_abstraction_20260629/run_dump_sample.sh
```

Rollout dump analysis:

```bash
cd /home/dataset-local/cjj/RL/GiGPO
/home/dataset-local/cjj/RL/envs/gigpo-baselines/bin/python experiments/alfworld_progress_abstraction_20260629/analyze_rollout_dump.py \
  --dump experiments/alfworld_progress_abstraction_20260629/traj_dump_step140_unseen20/step_140_batch_0.jsonl \
  --out experiments/alfworld_progress_abstraction_20260629/results_rollout_step140_unseen20.json
```

## Key Findings

### 1. Expert trajectories show reusable progress structure beyond raw histories

Across ALFWorld valid_seen + valid_unseen expert trajectories:

- Trajectories: 506.
- High-level expert steps: mean 7.31, median 7, p90 12.
- Raw history-prefix grouping compression: 1.79x.
- Medium progress abstraction compression: 2.57x.
- Coarse progress abstraction compression: 5.78x.

Per-task medium abstraction gains over raw prefix:

- `look_at_obj_in_light`: 2.19x -> 4.88x.
- `pick_and_place_simple`: 1.96x -> 3.17x.
- `pick_clean_then_place_in_recep`: 2.13x -> 3.15x.
- `pick_two_obj_and_place`: 1.44x -> 1.91x.

Interpretation: even a conservative progress abstraction increases group density. The biggest gains occur where task progress has clear subgoal phases. Two-object tasks need a more careful abstraction because first/second object identity matters.

### 2. Real GraphGPO rollout has sparse raw observation groups but denser progress groups

A 20-task unseen rollout from `global_step_140` produced:

- Trajectories: 20.
- Steps: 241.
- Success rate: 0.90.
- Mean steps: 12.05.
- Successful trajectory mean steps: 7.83.
- Failed trajectory mean steps: 50.0.

Group density on actual model rollout:

| grouping | compression | repeated item fraction |
| --- | ---: | ---: |
| raw observation | 1.62x | 0.473 |
| raw observation + semantic action | 1.43x | 0.419 |
| progress abstraction | 3.01x | 0.838 |
| progress abstraction + action schema | 2.56x | 0.751 |
| progress abstraction + semantic action | 1.75x | 0.581 |

Interpretation: progress-equivalent grouping roughly doubles the available group density compared with raw observation grouping on actual model rollouts. This supports a variance-reduction story for progress-equivalent credit assignment.

### 3. Failure mode is inefficient repeated search/navigation, not invalid action syntax

In the 20-task rollout:

- Invalid action fraction: 0.0.
- Go actions: 180 / 241 steps.
- Repeated-go fraction: 0.678 among go actions.
- Both failures were `pick_heat_then_place_in_recep`, each hitting max 50 steps.

Failure examples:

- `heat some mug and put it in cabinet`: the policy repeatedly cycles through countertops/shelves/coffeemachine and never commits to the heat/place subgoal.
- `heat some apple and put it in fridge`: similar repeated navigation between countertop/fridge/shelf locations, reaching max 50 steps.

Interpretation: the model already produces valid ALFWorld actions, but credit is poorly aligned for search termination and subgoal commitment. This is exactly where a belief-progress state with an information-gain/novelty term and a repeated-search cost could improve step efficiency.

## Research Implication

The useful paper direction is not simply better string matching. The empirical phenomenon is:

> Raw observation graphs provide sparse and sometimes misleading credit groups in partially observable ALFWorld tasks. Progress-equivalent abstractions preserve task-relevant subgoal state, increase reusable group density, and expose repeated-search failure modes that outcome reward and raw graph distance under-penalize.

This can motivate a method with three parts:

1. Task-progress belief abstraction: location, inventory/holding, goal object/receptacle, picked/placed/transformed/search progress, explored-location count.
2. Action semantic abstraction: navigation/search/pick/place/transform/redundant-search.
3. Confidence-calibrated credit: mix abstract group advantage with episode-level advantage according to group size/evidence, and penalize repeated low-information navigation.

## Next Experiments

1. Run a larger rollout dump on the full 134 unseen tasks and full 140 seen tasks for GraphGPO step 140.
2. Produce the same rollout dump for PPO and GRPO best/final checkpoints to compare failure modes.
3. Implement an offline credit diagnostic: compare GraphGPO shortest-path credit against progress abstraction credit on the same dumped trajectories.
4. If diagnostics remain strong, implement a training variant that replaces raw `anchor_obs` with progress abstraction for grouping and adds a repeated-search/information-gain term. Start with small train_data_size=16, rollout.n=8, 8 GPUs, validation 140.
