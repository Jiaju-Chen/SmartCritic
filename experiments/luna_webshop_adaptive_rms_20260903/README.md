# WebShop Luna adaptive RMS study

This experiment keeps the completed WebShop Luna setup unchanged except for
the residual coefficient. Instead of a fixed alpha, it tracks the token-masked
EMA second moments of the turn advantage and within-turn token residual:

```text
alpha_k = rho * EMA_RMS(turn_advantage) / EMA_RMS(token_residual)
```

The first batch initializes both moments directly. Subsequent batches use
`ema_beta=0.9`. The target weighted residual-to-turn RMS ratio is `rho=0.5`,
and alpha is clipped to `[0, 10]` only for numerical protection. The composed
advantage is whitened exactly as in the fixed-alpha WebShop runs.

The run records the following diagnostics in addition to the existing Luna
metrics:

- `hybrid/adaptive_residual_scale`
- `hybrid/adaptive_residual_raw_scale`
- `hybrid/adaptive_target_ratio`
- `hybrid/adaptive_batch_weighted_ratio`
- `hybrid/adaptive_ema_weighted_ratio`
- `hybrid/adaptive_ema_turn_rms`
- `hybrid/adaptive_ema_residual_rms`
- `hybrid/adaptive_scale_clipped`

EMA state is stored as `hybrid_advantage_scale_state.json` in both `latest`
and `best` checkpoints so resumed training preserves the controller history.
All W&B logging remains offline on `lzj`.
