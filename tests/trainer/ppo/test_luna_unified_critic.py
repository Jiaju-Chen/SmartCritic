import torch
from omegaconf import OmegaConf
from types import SimpleNamespace

from verl.trainer.ppo.core_algos import compute_luna_unified_value_loss
from verl.workers.critic.dp_critic import DataParallelPPOCritic


class TwoHeadDummyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, **_kwargs):
        positions = torch.arange(input_ids.size(1), device=input_ids.device, dtype=torch.float32)
        logits = torch.stack((positions, positions + 10.0), dim=-1)
        logits = logits.unsqueeze(0).expand(input_ids.size(0), -1, -1) * self.scale
        return SimpleNamespace(logits=logits)


def test_unified_value_loss_uses_token_and_turn_masks_independently():
    token_vpreds = torch.zeros((1, 3), requires_grad=True)
    turn_vpreds = torch.zeros((1, 3), requires_grad=True)
    old_values = torch.zeros((1, 3))
    token_returns = torch.tensor([[1.0, 2.0, 999.0]])
    turn_returns = torch.tensor([[3.0, 999.0, 999.0]])
    token_mask = torch.tensor([[1.0, 1.0, 0.0]])
    turn_mask = torch.tensor([[1.0, 0.0, 0.0]])

    total, token_loss, token_clipfrac, turn_loss, turn_clipfrac = (
        compute_luna_unified_value_loss(
            token_vpreds=token_vpreds,
            turn_vpreds=turn_vpreds,
            token_values=old_values,
            turn_values=old_values,
            token_returns=token_returns,
            turn_returns=turn_returns,
            token_mask=token_mask,
            turn_mask=turn_mask,
            cliprange_value=1000.0,
            turn_loss_coef=0.25,
        )
    )

    torch.testing.assert_close(token_loss, torch.tensor(2.5))
    torch.testing.assert_close(turn_loss, torch.tensor(9.0))
    torch.testing.assert_close(total, torch.tensor(4.75))
    torch.testing.assert_close(token_clipfrac, torch.tensor(0.0))
    torch.testing.assert_close(turn_clipfrac, torch.tensor(0.0))

    total.backward()
    torch.testing.assert_close(token_vpreds.grad, torch.tensor([[-1.0, -2.0, 0.0]]))
    torch.testing.assert_close(turn_vpreds.grad, torch.tensor([[-1.5, 0.0, 0.0]]))


def test_two_head_critic_returns_both_values_from_one_forward():
    module = TwoHeadDummyCritic()
    config = OmegaConf.create(
        {
            "model": {"use_remove_padding": False, "num_value_heads": 2},
            "unified_luna_turn_loss_coef": 1.0,
            "ulysses_sequence_parallel_size": 1,
        }
    )
    critic = DataParallelPPOCritic(
        config=config,
        critic_module=module,
        critic_optimizer=torch.optim.SGD(module.parameters(), lr=0.1),
    )
    critic.device_name = "cpu"

    values = critic._forward_micro_batch(
        {
            "input_ids": torch.ones((1, 5), dtype=torch.long),
            "responses": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "position_ids": torch.arange(5).unsqueeze(0),
        }
    )

    assert values.shape == (1, 2, 2)
    torch.testing.assert_close(
        values.float(),
        torch.tensor([[[2.0, 12.0], [3.0, 13.0]]]),
    )
