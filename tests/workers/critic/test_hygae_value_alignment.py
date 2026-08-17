import torch

from verl.workers.critic.dp_critic import (
    extract_response_value_views,
    replace_last_valid_response_value,
)


def test_extracts_pre_token_values_and_post_turn_value():
    full_values = torch.tensor(
        [
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
        ]
    )

    token_values, turn_end_values = extract_response_value_views(
        full_values=full_values,
        attention_mask=attention_mask,
        response_length=3,
    )

    torch.testing.assert_close(
        token_values,
        torch.tensor([[12.0, 13.0, 14.0], [22.0, 23.0, 0.0]]),
    )
    torch.testing.assert_close(turn_end_values, torch.tensor([15.0, 24.0]))


def test_replaces_only_last_valid_position_for_critic_training():
    token_values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
    turn_end_values = torch.tensor([30.0, 50.0])
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    critic_values, turn_end_mask = replace_last_valid_response_value(
        token_values=token_values,
        turn_end_values=turn_end_values,
        response_mask=response_mask,
    )

    torch.testing.assert_close(
        critic_values,
        torch.tensor([[1.0, 2.0, 30.0], [4.0, 50.0, 0.0]]),
    )
    torch.testing.assert_close(
        turn_end_mask,
        torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.bool),
    )


if __name__ == "__main__":
    test_extracts_pre_token_values_and_post_turn_value()
    test_replaces_only_last_valid_position_for_critic_training()
    print("HyGAE critic value alignment test passed")
