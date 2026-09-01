# WebShop Luna gamma/alpha study

This directory contains the isolated WebShop experiments prepared for the
`lzj` eight-A100 server.  The sequence intentionally runs only G0 and G1; G2
is selected after the first two 500-case evaluations are available.

| Run | Turn gamma | Turn lambda | Token gamma | Token lambda | Alpha |
| --- | ---: | ---: | ---: | ---: | ---: |
| G0 | 0.95 | 0.95 | 1.0 | 1.0 | 1.0 |
| G1 | 0.95 | 0.95 | 1.0 | 1.0 | 3.0 |

Both runs use the shared two-layer/two-head Luna critic, residual composition,
actor-advantage whitening, 128 training cases, 128 validation cases, 150
training steps, validation/checkpointing every five steps, and `best_latest`
checkpoint slots.  W&B is always offline on `lzj`.

The verified Python environment is deployed under
`/root/vepfs-data/chenjiaju/conda_envs/smartcritic-webshop-20260902`; no files
are installed into the server owner's environment or root filesystem.

The managed sequence is:

1. G0 train to step 150.
2. Evaluate G0 latest on the fixed 500-case WebShop set.
3. G1 train to step 150.
4. Evaluate G1 latest on the same fixed 500-case set.

`run_sequence.sh` is fail-fast.  It will not start a later phase unless the
previous phase has produced the expected completion marker.

The portable small WebShop resource bundle contains the two 1k-product files,
`items_human_ins.json` (loaded unconditionally by the upstream environment),
and the four small Lucene index aliases used by the environment code.
