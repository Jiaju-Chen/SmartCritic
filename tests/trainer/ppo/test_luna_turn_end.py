from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import compute_dual_critic_hybrid_gae, compute_luna_unified_value_loss
from verl.trainer.ppo.luna_diagnostics import turn_boundary_diagnostics
from verl.workers.critic.dp_critic import DataParallelPPOCritic
from verl.workers.critic.luna_value_alignment import align_luna_value_logits


class CausalSumCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def forward(self, input_ids, **kwargs):
        states = input_ids.float().cumsum(-1)
        return SimpleNamespace(logits=states.unsqueeze(-1) * self.weight)


def make_critic(module, position):
    config = OmegaConf.create({
        "model": {"use_remove_padding": False, "num_value_heads": 2},
        "turn_value_position": position,
    })
    critic = DataParallelPPOCritic(config, module, torch.optim.SGD(module.parameters(), lr=0.1))
    critic.device_name = "cpu"
    return critic


def gae(rewards, values, mask, position, **kwargs):
    return compute_dual_critic_hybrid_gae(
        token_level_rewards=rewards, token_values=torch.zeros_like(rewards),
        turn_values=values, response_mask=mask,
        traj_index=np.array(["a"] * len(rewards)), step_id=np.arange(len(rewards)),
        token_gamma=1.0, token_lam=1.0, turn_gamma=1.0, turn_lam=0.95,
        turn_value_position=position, whiten_advantages=False, **kwargs,
    )


@pytest.mark.parametrize("length", [1, 2, 4])
def test_real_padded_layout_reads_after_last_token_and_keeps_token_head(length):
    # Left-padded prompt, then right-padded response; no next observation.
    response = [7, 11, 13, 17][:length] + [0] * (4 - length)
    ids = torch.tensor([[0, 3, 5] + response])
    micro = {
        "input_ids": ids, "responses": ids[:, -4:],
        "attention_mask": (ids != 0).long(),
        "position_ids": torch.arange(7).unsqueeze(0),
    }
    model = CausalSumCritic()
    before = make_critic(model, "prompt_end")._forward_micro_batch(micro)
    after = make_critic(model, "action_end")._forward_micro_batch(micro)
    torch.testing.assert_close(before[..., 0], after[..., 0])
    assert before[0, 0, 1].item() == 16.0
    assert after[0, length - 1, 1].item() == 2 * (8 + sum(response))
    assert after[0, length - 1, 1] != before[0, length - 1, 1]
    changed = dict(micro, input_ids=ids.clone())
    changed["input_ids"][0, 3 + length - 1] += 19
    altered = make_critic(model, "action_end")._forward_micro_batch(changed)
    assert altered[0, length - 1, 1] - after[0, length - 1, 1] == 38
    torch.testing.assert_close(altered[:, :length, 0], after[:, :length, 0], rtol=0, atol=0)


@pytest.mark.parametrize("position, slots, expected", [
    ("prompt_end", [0, 0], [0.0875, 0.25]),
    ("action_end", [1, 2], [0.25, 0.0]),
])
def test_turn_gae_bootstrap_target_and_loss_mask_move_together(position, slots, expected):
    rewards = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    values = torch.tensor([[0.9, 0.75, 99.0], [0.75, 88.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    result = gae(rewards, values, mask, position)
    _, _, turn_returns, turn_mask, turn_advantages, _ = result
    assert turn_mask.nonzero().tolist() == [[0, slots[0]], [1, slots[1]]]
    for row, slot in enumerate(slots):
        torch.testing.assert_close(turn_advantages[row][mask[row].bool()],
                                   torch.full((int(mask[row].sum()),), expected[row]))
        torch.testing.assert_close(turn_returns[row, slot], values[row, slot] + expected[row])


def test_endpoint_value_loss_backpropagates_through_actual_terminal_logit():
    logits = torch.zeros((2, 7, 2), requires_grad=True)
    aligned = align_luna_value_logits(logits, 4, "action_end")
    turn_mask = torch.tensor([[0., 1., 0., 0.], [0., 0., 0., 1.]])
    token_mask = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 1.]])
    zeros = torch.zeros_like(turn_mask)
    loss, *_ = compute_luna_unified_value_loss(
        token_vpreds=aligned[..., 0], turn_vpreds=aligned[..., 1],
        token_values=zeros, turn_values=zeros, token_returns=zeros,
        turn_returns=turn_mask, token_mask=token_mask, turn_mask=turn_mask,
        cliprange_value=0.5,
    )
    loss.backward()
    assert logits.grad[..., 1].nonzero().tolist() == [[0, 4], [1, 6]]
    assert torch.count_nonzero(logits.grad[..., 0]) == 0


def test_perfect_action_conditioned_terminal_value_cancels_turn_signal_only():
    rewards = torch.tensor([[0.0, 1.0]])
    result = gae(rewards, rewards.clone(), torch.ones_like(rewards), "action_end")
    torch.testing.assert_close(result[4], torch.zeros_like(rewards))
    torch.testing.assert_close(result[1], torch.ones_like(rewards))


def test_diagnostics_use_one_action_not_response_length_and_ignore_missing_class():
    advantages = torch.tensor([[2., 2., 0.], [-1., -1., -1.]])
    mask = torch.tensor([[0., 1., 0.], [0., 0., 1.]])
    metrics = turn_boundary_diagnostics(advantages, torch.zeros_like(mask), advantages, mask, [True, False])
    assert metrics["turn_diag/raw_advantage_rms"] == pytest.approx(2.5 ** 0.5)
    assert metrics["turn_diag/success_raw_advantage_mean"] == 2.0
    assert metrics["turn_diag/failure_raw_advantage_mean"] == -1.0
    assert metrics["turn_diag/action_count"] == 2
    metrics = turn_boundary_diagnostics(advantages[:1], mask[:1], mask[:1], mask[:1], [True])
    assert "turn_diag/failure_raw_advantage_mean" not in metrics


def test_reject_mismatched_head_and_unknown_position():
    with pytest.raises(ValueError, match="two-head"):
        align_luna_value_logits(torch.zeros(1, 4, 1), 2, "action_end")
    with pytest.raises(ValueError, match="Unknown"):
        align_luna_value_logits(torch.zeros(1, 4, 2), 2, "last")


def test_diagnostics_accept_rollout_object_array_outcomes():
    mask = torch.tensor([[0., 1.], [0., 1.]])
    advantages = torch.tensor([[2., 2.], [-1., -1.]])
    outcomes = np.array([True, False], dtype=object)
    metrics = turn_boundary_diagnostics(advantages, mask, mask, mask, outcomes)
    assert metrics["turn_diag/success_raw_advantage_mean"] == 2.0
    assert metrics["turn_diag/failure_raw_advantage_mean"] == -1.0
