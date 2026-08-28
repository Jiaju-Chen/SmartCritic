# Stable SearchQA Luna reruns

This directory contains isolated launchers for rerunning the two SearchQA Luna
experiments after the previous SSH forwarding failure.

The official GPU retriever remains on the dedicated one-card host.  The
launcher supervises the SSH forward and reconnects it when the connection
drops.  Search failures are configured to abort training instead of becoming
ordinary observations, so a broken retrieval path cannot silently corrupt a
run.

The two experiments use separate run and checkpoint directories:

- `run_2l.sh`: two-layer critic
- `run_28l.sh`: twenty-eight-layer critic

Both preserve the previous formal settings: train batch 256, rollout group
size 5, maximum four turns, 200 training iterations, validation and checkpoint
events every 50 iterations, token discount 1.0, turn discount 0.95, and online
Weights & Biases logging.
