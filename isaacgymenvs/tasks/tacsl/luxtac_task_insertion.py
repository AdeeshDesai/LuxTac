"""Insertion task sensed through LuxTac pads.

LuxTacSensors sits before TacSLTaskInsertion in the MRO, so the LuxTac force law and
sensor setup replace the GelSight ones and the rest of the task (observations, actions,
controller, resets, cameras) is inherited.

Adds these observation keys, each present only if listed in obsDims:

    luxtac_left / luxtac_right             : (num_envs, rows, cols) float32
        The taxel reading. For the photodiode response this is the LuxTac driver's
        normalized s in [0, 1] -- (1 - raw/S0) divided by that taxel's R/b and gated
        at 3 sigma -- the same quantity pad.read()['s'] gives on the real sensor
        after a re-zero. s = 1 is about 5.8 N on one taxel. NOTE each 4x4 block is
        the TRANSPOSE of the driver's grid; see photodiode.py.
    luxtac_depth_left / luxtac_depth_right : (num_envs, sample_rows, sample_cols)
        Depth in metres at every sample point. Saved so the curve can be reapplied at
        training time without collecting again.

Deployed to isaacgymenvs/tasks/tacsl/ by scripts/deploy.sh.
"""

from isaacgymenvs.tacsl_sensors.luxtac.luxtac_env_mixin import LuxTacSensors
from isaacgymenvs.tasks.tacsl.tacsl_task_insertion import TacSLTaskInsertion


class LuxTacTaskInsertion(LuxTacSensors, TacSLTaskInsertion):
    """TacSL insertion task, LuxTac pads instead of GelSight."""

    def _create_sensors(self):
        # Parent creates cameras and honours the disabled GelSight flags.
        super()._create_sensors()
        self.setup_luxtac_sensors()

    def get_tactile_force_field_tensors_dict(self):
        """Same accessor name the task layer uses for the GelSight force field."""
        return self.get_luxtac_tensors_dict()

    def compute_observations(self):
        # Parent returns obs_buf or, in dict-obs mode (the teleop path), obs_dict.
        ret = super().compute_observations()
        luxtac = self.get_luxtac_tensors_dict()
        for name in ('luxtac_left', 'luxtac_right'):
            if name not in luxtac:
                continue
            if name in self.obs_dict:
                self.obs_dict[name][:] = luxtac[name]['grid']
            depth_key = name.replace('luxtac_', 'luxtac_depth_')
            if depth_key in self.obs_dict:
                self.obs_dict[depth_key][:] = luxtac[name]['depth_grid']
        return ret
