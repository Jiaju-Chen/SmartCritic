import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_hygae_unified_gae


def compute(
    rewards,
    values,
    mask,
    *,
    traj_index=None,
    step_id=None,
    token_gamma=1.0,
    token_lam=0.0,
    turn_lam=0.0,
    alpha=0.5,
    length_matched_turn_gamma=True,
):
    batch_size = rewards.shape[0]
    if traj_index is None:
        traj_index = np.asarray(["trajectory-0"] * batch_size, dtype=object)
    if step_id is None:
        step_id = np.arange(batch_size, dtype=np.int32)
    return compute_hygae_unified_gae(
        token_level_rewards=rewards,
        values=values,
        response_mask=mask,
        traj_index=traj_index,
        step_id=step_id,
        token_gamma=token_gamma,
        token_lam=token_lam,
        turn_lam=turn_lam,
        alpha=alpha,
        length_matched_turn_gamma=length_matched_turn_gamma,
        whiten_advantages=False,
    )


def test_uses_final_token_value_and_mixes_advantage_and_return():
    rewards = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[0.2, 0.7]])
    mask = torch.ones_like(rewards)

    advantages, returns, turn_advantages, token_advantages, turn_returns = compute(
        rewards, values, mask
    )

    torch.testing.assert_close(token_advantages, torch.tensor([[0.5, 0.3]]))
    torch.testing.assert_close(turn_advantages, torch.tensor([[0.3, 0.3]]))
    torch.testing.assert_close(turn_returns, torch.tensor([[2.0, 1.0]]))
    torch.testing.assert_close(advantages, torch.tensor([[0.4, 0.3]]))
    torch.testing.assert_close(returns, torch.tensor([[1.35, 1.0]]))


def test_turn_value_is_not_first_token_or_turn_mean():
    rewards = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[9.0, 0.25]])
    mask = torch.ones_like(rewards)

    advantages, _, turn_advantages, _, _ = compute(
        rewards, values, mask, alpha=1.0
    )

    expected = torch.tensor([[0.75, 0.75]])
    torch.testing.assert_close(turn_advantages, expected)
    torch.testing.assert_close(advantages, expected)


def test_terminal_reward_propagates_across_environment_turns():
    rewards = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    values = torch.zeros_like(rewards)
    mask = torch.ones_like(rewards)

    advantages, _, turn_advantages, token_advantages, _ = compute(
        rewards,
        values,
        mask,
        token_lam=1.0,
        turn_lam=1.0,
    )

    expected = torch.ones_like(rewards)
    torch.testing.assert_close(turn_advantages, expected)
    torch.testing.assert_close(token_advantages, expected)
    torch.testing.assert_close(advantages, expected)


def test_turn_discount_matches_generated_token_length():
    rewards = torch.zeros((2, 2))
    values = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    _, _, turn_advantages, _, _ = compute(
        rewards,
        values,
        mask,
        token_gamma=0.5,
        alpha=1.0,
    )

    torch.testing.assert_close(turn_advantages[0], torch.tensor([-0.5, -0.5]))
    torch.testing.assert_close(turn_advantages[1], torch.tensor([-2.0, 0.0]))


def test_rejects_invalid_mixing_coefficient():
    rewards = torch.zeros((1, 1))
    values = torch.zeros_like(rewards)
    mask = torch.ones_like(rewards)

    try:
        compute(rewards, values, mask, alpha=1.1)
    except ValueError as error:
        assert "alpha must be in" in str(error)
    else:
        raise AssertionError("invalid HyGAE alpha was accepted")


if __name__ == "__main__":
    test_uses_final_token_value_and_mixes_advantage_and_return()
    test_turn_value_is_not_first_token_or_turn_mean()
    test_terminal_reward_propagates_across_environment_turns()
    test_turn_discount_matches_generated_token_length()
    test_rejects_invalid_mixing_coefficient()
    print("HyGAE unified GAE tests passed")
