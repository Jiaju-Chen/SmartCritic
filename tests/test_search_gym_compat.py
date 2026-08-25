import importlib.util
import sys
from pathlib import Path


def test_search_env_falls_back_to_gymnasium(monkeypatch):
    monkeypatch.setitem(sys.modules, "gym", None)
    module_path = (
        Path(__file__).parents[1]
        / "agent_system"
        / "environments"
        / "env_package"
        / "search"
        / "envs.py"
    )
    spec = importlib.util.spec_from_file_location("search_envs_gym_compat", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.gym.__name__ == "gymnasium"

