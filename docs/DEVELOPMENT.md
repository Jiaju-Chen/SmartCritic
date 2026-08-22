# Development and Synchronization

## Source of Truth

The private GitHub repository is the source of truth:

```text
git@github.com:Jiaju-Chen/SmartCritic.git
```

Use these remotes on every machine:

```text
origin    git@github.com:Jiaju-Chen/SmartCritic.git
upstream  git@github.com:langfengQ/verl-agent.git
```

On SAIDS, the host-specific GitHub alias is `github-smartcritic`, so its origin
URL is:

```text
git@github-smartcritic:Jiaju-Chen/SmartCritic.git
```

## Server Layout

Current code locations:

```text
yun-my8card: /home/dataset-local/cjj/RL/GiGPO_PVF
SAIDS:       /data2/group_何向南/chenjiaju/luna/repo
```

The SAIDS code path is under `data2` because home storage is not intended for
models, datasets, environments, or checkpoints.

## Branch and Worktree Rule

Use one branch and one worktree per experiment family. Do not edit a completed
run's implementation in place.

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git worktree add ../SmartCritic-<experiment> -b <experiment-branch> main
```

Before launching a run:

1. Commit the code and launch configuration.
2. Record the commit hash in the run metadata or W&B config.
3. Use a unique run name and output directory.
4. Keep `best` and `latest` checkpoints according to the experiment policy.
5. Push the experiment branch before or immediately after launch.

## Two-Server Workflow

Never synchronize source code by copying a dirty working directory between
servers. Synchronize through GitHub:

```bash
git status --short
git push origin HEAD
```

On the other server:

```bash
git fetch origin
git switch <experiment-branch>
git pull --ff-only origin <experiment-branch>
```

If both servers need simultaneous development, assign different branches. Merge
only tested changes into `main`.

## Storage Policy

Do not commit any of the following:

- model weights or Hugging Face caches;
- ALFWorld/WebShop datasets and environment installations;
- checkpoints, optimizer state, rollout caches, or Ray state;
- W&B offline runs or API credentials;
- console logs and generated evaluation output.

Use server-local storage directories such as:

```text
<storage-root>/models
<storage-root>/datasets
<storage-root>/envs
<storage-root>/checkpoints
<storage-root>/runs
<storage-root>/wandb
```

Store only scripts, configuration, compact metric summaries, and documentation
in Git. A run should remain reproducible from its commit, configuration, random
seed, and documented external asset versions.

## Credential Policy

Credentials must remain in the user's SSH agent, `~/.ssh`, `~/.netrc`, or an
environment variable. Never add API keys, private keys, `.netrc`, or `.env`
files to this repository.

