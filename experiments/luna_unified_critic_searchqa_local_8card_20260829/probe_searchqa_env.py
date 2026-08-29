#!/usr/bin/env python3
"""Exercise a real SearchEnv search turn and a rewarded answer turn."""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.env import SearchEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-url", default="http://127.0.0.1:18002/retrieve")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    env = SearchEnv(
        OmegaConf.create(
            {
                "search_url": args.search_url,
                "topk": 3,
                "timeout": args.timeout,
                "log_requests": False,
                "fail_on_error": True,
            }
        )
    )
    try:
        env.reset(
            {
                "ground_truth": {"target": ["Paris"]},
                "max_turns": 4,
                "data_source": "searchR1_local_acceptance",
            }
        )

        search_step = env.step("<think>I should retrieve evidence.</think><search>Eiffel Tower city</search>")
        observation = search_step["observations"][0]["content"]
        assert search_step["reward"] == 0
        assert not search_step["done"]
        assert "Paris" in observation

        answer_step = env.step("<think>The evidence identifies the city.</think><answer>Paris</answer>")
        assert answer_step["done"]
        assert answer_step["reward"] == 1
        print("SEARCH_ENV_ACCEPTANCE_OK reward=1 turns=2 observation_contains=Paris")
    finally:
        env.close()


if __name__ == "__main__":
    main()
