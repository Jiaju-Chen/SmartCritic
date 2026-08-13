import os

import ray

from agent_system.environments.env_package.webshop.envs import build_webshop_envs


def main() -> None:
    project_root = os.environ.get(
        "PROJECT_ROOT", "/home/dataset-local/cjj/RL/GiGPO_PVF_WebShop"
    )
    package_root = os.path.join(
        project_root, "agent_system/environments/env_package/webshop/webshop"
    )
    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": False,
        "file_path": os.path.join(package_root, "data/items_shuffle_1000.json"),
        "attr_path": os.path.join(package_root, "data/items_ins_v2_1000.json"),
    }
    resources = {"num_cpus": 0.25}
    train_count = int(os.environ.get("TRAIN_ENV_COUNT", "128"))
    val_count = int(os.environ.get("VAL_ENV_COUNT", "16"))
    val_total = int(os.environ.get("VAL_TOTAL_COUNT", "128"))
    if val_total % val_count:
        raise ValueError("VAL_TOTAL_COUNT must be divisible by VAL_ENV_COUNT")

    ray.init(
        num_cpus=96,
        include_dashboard=False,
        _temp_dir=os.environ.get(
            "RAY_TMP_ROOT", "/home/dataset-local/cjj/dcw_probe_ray"
        ),
    )
    train_envs = build_webshop_envs(
        seed=0,
        env_num=train_count,
        group_n=1,
        is_train=True,
        env_kwargs=env_kwargs,
        resources_per_worker=resources,
    )
    val_envs = build_webshop_envs(
        seed=1000,
        env_num=val_count,
        group_n=1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker=resources,
    )
    train_obs, _ = train_envs.reset()
    seen_goal_indices = []
    for start in range(0, val_total, val_count):
        goal_indices = list(range(start, start + val_count))
        val_obs, val_infos = val_envs.reset(goal_indices=goal_indices)
        assert len(val_obs) == val_count
        seen_goal_indices.extend(info["goal_index"] for info in val_infos)
    assert seen_goal_indices == list(range(val_total))
    print(
        f"PASS train_envs={len(train_obs)} val_envs={val_count} "
        f"validation_interactions={len(seen_goal_indices)} "
        f"unique_validation_goals={len(set(seen_goal_indices))}"
    )
    train_envs.close()
    val_envs.close()
    ray.shutdown()


if __name__ == "__main__":
    main()
