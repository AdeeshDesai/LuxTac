"""LuxTac normal-force tactile sensor for IsaacGym / TacSL.

Deployed into isaacgymenvs/tacsl_sensors/luxtac by scripts/deploy.sh.
"""

from .luxtac_cfg import (
    LuxTacSensorCfg,
    UPSTREAM_CFG_ALOHA,
    UPSTREAM_CFG_DEFAULT,
)

__all__ = [
    "LuxTacSensorCfg",
    "UPSTREAM_CFG_ALOHA",
    "UPSTREAM_CFG_DEFAULT",
    "LuxTacFieldSensor",
]


def __getattr__(name):
    # Deferred so `import luxtac` works without isaacgym present (the config and
    # viz modules are pure numpy/torch and useful on their own).
    if name == "LuxTacFieldSensor":
        from .luxtac_sensor import LuxTacFieldSensor
        return LuxTacFieldSensor
    raise AttributeError(name)
