"""Read-only diagnostics with one equally weighted observation per action."""

import numpy as np
import torch


def turn_boundary_diagnostics(advantages, values, returns, boundary_mask, episode_success=None):
    selected = boundary_mask.bool()
    if not selected.any():
        return {}
    raw_advantages = advantages[selected].float()
    value_errors = (returns - values)[selected].float()
    metrics = {
        "turn_diag/raw_advantage_rms": raw_advantages.square().mean().sqrt().item(),
        "turn_diag/raw_advantage_abs_mean": raw_advantages.abs().mean().item(),
        "turn_diag/old_value_target_mse": value_errors.square().mean().item(),
        "turn_diag/action_count": selected.sum().item(),
    }
    if episode_success is not None:
        success = torch.as_tensor(np.asarray(episode_success, dtype=np.bool_), device=values.device)
        if success.ndim != 1 or success.numel() != selected.size(0):
            raise ValueError("episode_success must contain one outcome per action row")
        for name, condition in (("success", success), ("failure", ~success)):
            mask = selected & condition.unsqueeze(-1)
            metrics[f"turn_diag/{name}_action_count"] = mask.sum().item()
            if mask.any():
                metrics[f"turn_diag/{name}_raw_advantage_mean"] = advantages[mask].float().mean().item()
    return metrics
