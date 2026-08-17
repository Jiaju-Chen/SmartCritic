# HyGAE turn-end alignment diagnostic

This isolated experiment fixes the causal alignment of the HyGAE turn value.
The token critic values remain aligned before each generated token, while the
turn value is read after the final valid response token and is supervised at
the final response position.

Remote worktree: `/home/dataset-local/cjj/RL/GiGPO_PVF_HYGAE_TURNEND`

Remote branch: `hygae-turn-end-fix-20260817`

The one-step smoke uses 8 training tasks and 4 validation tasks. The bounded
diagnostic uses the previous ALFWorld settings with 128 training tasks, full
140-task validation, validation batch size 20, and 10 updates. Both retain only
`best` and `latest` checkpoint slots.
