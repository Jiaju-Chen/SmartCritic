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

## Offline Asset Transfer

SAIDS does not need direct Internet access. Run the following command on the Mac
that can SSH to both `yun-my8card-vscode` and `SAIDS`:

```bash
bash scripts/transfer_core_assets_via_ssh.sh
```

The script streams data from yun-my8card through the Mac directly into SAIDS;
it does not create a local copy. It transfers approximately 10.5 GB:

```text
Qwen2.5-1.5B-Instruct  2.9 GB
ALFWorld data          1.7 GB
WebShop resources      5.9 GB
```

The destination layout is:

```text
/data2/group_何向南/chenjiaju/luna/shared/models/Qwen2.5-1.5B-Instruct
/data2/group_何向南/chenjiaju/luna/shared/datasets/alfworld_data
/data2/group_何向南/chenjiaju/luna/shared/datasets/webshop
```

The WebShop source directory contains a host-specific private key that is not a
dataset dependency. The transfer script explicitly excludes it and verifies
that it is absent on SAIDS. The Hugging Face snapshot uses symlinks, so the
script dereferences them and verifies the transferred model weight checksum.

When SAIDS is reachable only through a Codex-created SSH multiplexing tunnel,
the script automatically discovers an active
`~/.ssh/codex-saids-control-*` socket. A specific socket can be selected with:

```bash
DEST_CONTROL_PATH="$HOME/.ssh/codex-saids-control-4" \
  bash scripts/transfer_core_assets_via_ssh.sh
```

This script transfers model and environment data only. Python/Conda environments
must be packed separately with `conda-pack` or rebuilt from an offline package
bundle; copying a Conda directory to a different absolute path is not reliable.

## Credential Policy

Credentials must remain in the user's SSH agent, `~/.ssh`, `~/.netrc`, or an
environment variable. Never add API keys, private keys, `.netrc`, or `.env`
files to this repository.
