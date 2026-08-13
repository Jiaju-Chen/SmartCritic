#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/GiGPO_PVF_WebShop}
PYTHON=${PYTHON:-/home/dataset-local/conda/envs/verl-agent-webshop/bin/python}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAVA_HOME=${JAVA_HOME:-/home/dataset-local/conda/envs/verl-agent-webshop}
export PATH="$JAVA_HOME/bin:$PATH"

"$PYTHON" - <<'PY'
import os
import sys

project_root = os.getcwd()
webshop_root = os.path.join(
    project_root,
    "agent_system/environments/env_package/webshop/webshop",
)
sys.path.insert(0, webshop_root)

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

data_root = os.path.join(webshop_root, "data")
env = WebAgentTextEnv(
    observation_mode="text",
    num_products=None,
    human_goals=False,
    seed=0,
    file_path=os.path.join(data_root, "items_shuffle_1000.json"),
    attr_path=os.path.join(data_root, "items_ins_v2_1000.json"),
)
try:
    observation, info = env.reset(session=0)
    assert len(env.server.goals) > 500
    assert "Instruction:" in observation
    assert env.get_available_actions()["has_search_bar"]
    print(f"PASS goals={len(env.server.goals)} session=0")
    print(observation[:240].replace("\n", " | "))
finally:
    env.close()
PY
