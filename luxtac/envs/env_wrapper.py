"""Single-environment interface to the LuxTac task for the scripts in this repo.

`IsaacEnvWrapper(cfg)` builds the task named in the hydra config and exposes numpy
actions in, observation dict out. The underlying IsaacGymEnvs task object is available
as `.envs`. The task classes are deployed into the isaacgymenvs clone by
scripts/deploy.sh.
"""
import importlib

import isaacgym                      # noqa: F401  must be imported before torch
import isaacgymenvs
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_seed

import numpy as np
import torch

# task name in the yaml -> (module, class)
TASKS = {
    "LuxTacTaskInsertion": ("isaacgymenvs.tasks.tacsl.luxtac_task_insertion", "LuxTacTaskInsertion"),
}


def _task_class(name):
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    module, attr = TASKS[name]
    try:
        return getattr(importlib.import_module(module), attr)
    except ImportError as e:
        raise ImportError(
            f"task {name} is not deployed in {isaacgymenvs.__path__[0]} "
            f"({e.name or e}); run scripts/deploy.sh on that clone") from e


class IsaacEnvWrapper:
    def __init__(self, cfg):
        self.cfg = cfg
        task_cfg = omegaconf_to_dict(cfg["task"])
        print(f"[luxtac] isaacgymenvs from {isaacgymenvs.__path__[0]}")
        self.envs = _task_class(task_cfg["name"])(
            cfg=task_cfg,
            rl_device=cfg.rl_device,
            sim_device=cfg.sim_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless,
            virtual_screen_capture=cfg.capture_video,
            force_render=cfg.force_render,
        )

    observation_space = property(lambda self: self.envs.observation_space)
    action_space = property(lambda self: self.envs.action_space)
    num_acts = property(lambda self: self.envs.num_actions)
    num_obs = property(lambda self: self.envs.num_observations)
    num_envs = property(lambda self: self.envs.num_environments)

    def seed(self, seed=None):
        """Seed python, numpy and torch. None draws a seed at random."""
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = set_seed(seed, torch_deterministic=self.cfg.torch_deterministic, rank=0)

    def reset(self):
        self.envs.reset_idx(torch.arange(self.num_envs))
        self.envs.compute_observations()
        return self.envs.reset()["obs"]

    def step(self, actions):
        """actions: numpy array for the single environment."""
        act = torch.from_numpy(actions).to(dtype=torch.float32).unsqueeze(0)
        obs, reward, done, info = self.envs.step(act)
        return obs["obs"], reward, done, info

    def render(self, mode="rgb_array"):
        if self.envs.virtual_display and mode == "rgb_array":
            return np.array(self.envs.virtual_display.grab())

    def close(self):
        pass
