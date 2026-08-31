"""Align the two value heads without changing the token critic's prefix states."""

import torch


def align_luna_value_logits(logits, response_length, turn_value_position="prompt_end"):
    if turn_value_position not in ("prompt_end", "action_end"):
        raise ValueError(f"Unknown turn_value_position: {turn_value_position}")
    if response_length < 1 or logits.size(1) <= response_length:
        raise ValueError("Value alignment requires a prompt boundary and a nonempty response")
    prefix_values = logits[:, -response_length - 1 : -1]
    if turn_value_position == "prompt_end":
        return prefix_values.squeeze(-1)
    if logits.ndim != 3 or logits.size(-1) != 2:
        raise ValueError("action_end readout requires a unified two-head critic")
    # Slot j holds the turn-head value AFTER response token j; GAE selects
    # the last valid slot. The token head remains BEFORE token j.
    return torch.stack((prefix_values[..., 0], logits[:, -response_length:, 1]), dim=-1)
