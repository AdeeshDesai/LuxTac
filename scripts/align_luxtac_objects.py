#!/usr/bin/env python
"""Close the sim gripper on a LuxTac alignment object and read the pad at each width.

The object is selected through hydra overrides and the asset table
luxtac_asset_info_objects.yaml, so the task yaml is not edited.

The real gripper was commanded closed past the object and stalled on its force limit,
so its recorded width cannot be replayed. The DOF target is swept instead and the
achieved penetration is reported:

    DOF = (T - 2*indent + 2*proud) / 2      T = object extent along the grip axis

with proud = 1.0mm for the gel45 URDF (gap = 2*DOF - 2.0mm).

One object per process: IsaacGym cannot create a second sim in the same process.
scripts/run_sweep.sh and scripts/run_holds.sh loop over the objects.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CONFIG = str(_HERE.parent / "luxtac" / "config")

SHAPES = ("cuboid", "cylinder", "ellipse", "peg")
ORIENTS = ("vert", "horz", "deg45", "deg135")
# extent along the grip axis, mm
THICK = {"cuboid": 20.0, "cylinder": 25.0, "ellipse": 20.0, "peg": 15.0}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--object", default="all",
                help="'all', a shape (cuboid/cylinder/ellipse/peg), or one "
                     "shape_orient key such as peg_deg45")
ap.add_argument("--indent", type=float, nargs="+",
                default=[0.3, 0.5, 0.7, 0.9, 1.2, 1.5],
                help="target indentations in mm, swept per object")
ap.add_argument("--proud-mm", type=float, default=1.0,
                help="how far the gel stands proud of the blade (gel45 URDF: 1.0)")
ap.add_argument("--urdf", default="luxtac_franka_gel45.urdf")
ap.add_argument("--response", default="photodiode",
                choices=["photodiode", "linear_peak", "spring"])
ap.add_argument("--k", type=float, default=3100.0, help="gel compliance_stiffness, N/m")
ap.add_argument("--cols", type=int, default=4,
                help="taxels along the finger (4 = the calibrated block)")
ap.add_argument("--settle", type=int, default=60, help="sim steps to settle at each width")
ap.add_argument("--out", default="/tmp/lux_sim.npz")
ap.add_argument("--diag", action="store_true",
                help="print where the object and the pad actually are, then exit. "
                     "Verifies the staging instead of trusting that the pose "
                     "randomisation in the task yaml is zeroed.")
# Hydra also parses sys.argv, so take only our flags and hand the rest on.
_A, _REST = ap.parse_known_args()
sys.argv = [sys.argv[0]] + _REST


def selected():
    if _A.object == "all":
        return [(s, o) for s in SHAPES for o in ORIENTS]
    if _A.object in SHAPES:
        return [(_A.object, o) for o in ORIENTS]
    for s in SHAPES:
        for o in ORIENTS:
            if _A.object == f"{s}_{o}":
                return [(s, o)]
    raise SystemExit(f"unknown --object {_A.object!r}")


def set_compliance(t_, k, kd, links=("luxtac_gel_left", "luxtac_gel_right")):
    """Push the gel compliance onto the live actor.

    The yaml value is applied at asset load; this re-applies it per run so a swept
    value takes effect without rebuilding the env.
    """
    from isaacgym import gymapi
    gym, env0 = t_.gym, t_.env_ptrs[0]
    h = t_.actor_handles["franka"]
    for env_ptr in t_.env_ptrs:
        props = gym.get_actor_rigid_shape_properties(env_ptr, h)
        names = [gym.get_actor_rigid_body_name(env_ptr, h, i)
                 for i in range(gym.get_actor_rigid_body_count(env_ptr, h))]
        for i, p in enumerate(props):
            bi = gym.get_actor_rigid_body_index(env_ptr, h, 0, gymapi.DOMAIN_ACTOR)
            p.compliance = k
        gym.set_actor_rigid_shape_properties(env_ptr, h, props)
    return k, kd


def _save(results, path):
    pack = {}
    for sub, rows in results.items():
        for key in ("target_mm", "dof_mm", "dof_act_mm", "F", "d_max", "d_mean"):
            pack[f"{sub}__{key}"] = np.array([r[key] for r in rows])
        pack[f"{sub}__s"] = np.stack([r["s"] for r in rows])
        pack[f"{sub}__depth"] = np.stack([r["depth"] for r in rows])
    pack["subassemblies"] = np.array(sorted(results))
    np.savez_compressed(path, **pack)


def main():
    import hydra
    from omegaconf import DictConfig
    from isaacgym import gymapi
    from luxtac.envs.env_wrapper import IsaacEnvWrapper   # isaacgym before torch
    import torch
    from luxtac import luxtac_viz

    jobs = selected()
    results = {}

    @hydra.main(version_base="1.1", config_path=_CONFIG,
                config_name="isaacgym_config_luxtac")
    def run(cfg: DictConfig):
        cfg.headless = True
        cfg.num_envs = 1
        e = cfg.task.env
        e.asset_info_filename = "luxtac_asset_info_objects.yaml"
        e.franka_urdf_file = _A.urdf
        e.luxtac.num_cols = int(_A.cols)
        # obsDims is declared SEPARATELY from env.luxtac and the task allocates the
        # observation buffers from it, so changing num_cols alone throws
        #   "expanded size of the tensor (8) must match the existing size (4)"
        # out of luxtac_task_insertion.compute_observations.
        _nr, _ss = int(e.luxtac.num_rows), int(e.luxtac.subsamples)
        for _side in ("left", "right"):
            e.obsDims[f"luxtac_{_side}"] = [_nr, int(_A.cols)]
            e.obsDims[f"luxtac_depth_{_side}"] = [_nr * _ss, int(_A.cols) * _ss]
        e.luxtac.response = _A.response
        e.luxtac.compliance_stiffness = float(_A.k)
        e.compliance_stiffness = float(_A.k)

        for shape, orient in jobs:
            sub = f"lux_{shape}_{orient}"
            T = THICK[shape]
            mass = 0.03                       # contact damping
            kd = float(2 * np.sqrt(_A.k * mass))
            e.desired_subassemblies = [sub]
            e.luxtac.compliant_damping = kd
            e.compliant_damping = kd

            print(f"\n=== {sub}  (grip-axis extent {T:.1f} mm, gel proud "
                  f"{_A.proud_mm} mm, k={_A.k:.0f}, kd={kd:.1f}) ===")
            env = IsaacEnvWrapper(cfg)
            env.seed(0)
            env.reset()
            t_ = env.envs
            gym = t_.gym
            zero = np.zeros(env.action_space.shape[-1], dtype=np.float32)
            for _ in range(4):
                env.step(zero)

            gel_id = gym.find_actor_rigid_body_index(
                t_.env_ptrs[0], t_.actor_handles["franka"],
                "luxtac_gel_left", gymapi.DOMAIN_ENV)
            assert gel_id > -1, "luxtac_gel_left not found"

            if _A.diag:
                # Close first.
                T0 = (T - 2.0 * _A.indent[0] + 2.0 * _A.proud_mm) / 2.0
                t_.cfg_task.env.franka_close_gripper_width = float(T0 * 1e-3)
                for _ in range(_A.settle):
                    env.step(zero)
                gym.refresh_rigid_body_state_tensor(t_.sim)
                gel_p = t_.body_pos[0, gel_id].cpu().numpy()
                print(f"  object origin - fingertip midpoint, mm: "
                      f"{np.round((t_.plug_pos[0].cpu().numpy() - t_.fingertip_midpoint_pos[0].cpu().numpy())*1e3, 2)}")
                pads = t_.get_luxtac_tensors_dict()
                dep = pads["luxtac_left"]["depth_grid"][0].cpu().numpy() * 1e3
                print(f"  depth_grid {dep.shape}  nonzero {int((dep>0).sum())}/{dep.size}  max {dep.max():.4f} mm")
                print("  sample-point footprint (. = no contact):")
                for r in range(dep.shape[0]):
                    print("    " + "".join("." if v <= 0 else
                          ("#" if v > 0.75*dep.max() else ("+" if v > 0.35*dep.max() else "-"))
                          for v in dep[r]))
                cy = (dep > 0).any(axis=1).nonzero()[0]
                cx = (dep > 0).any(axis=0).nonzero()[0]
                print(f"  contact spans rows {cy.min()}-{cy.max()} of 0-19, cols {cx.min()}-{cx.max()} of 0-19")
                print(f"  centroid (row, col) = "
                      f"{np.round(np.array(np.nonzero(dep>0)).mean(axis=1), 2)}  (centre would be 9.5, 9.5)")
                return 0
            rows = []
            print(f"{'tgt mm':>7} {'DOF mm':>7} {'DOFact':>7} {'F_gel N':>8} "
                  f"{'d_max':>7} {'d_mean':>7} {'mean s':>8} {'lit':>7}")
            for ind in _A.indent:
                dof_mm = (T - 2.0 * ind + 2.0 * _A.proud_mm) / 2.0
                t_.cfg_task.env.franka_close_gripper_width = float(dof_mm * 1e-3)
                for _ in range(_A.settle):
                    env.step(zero)
                gym.refresh_net_contact_force_tensor(t_.sim)
                f = float(torch.norm(t_.contact_force_net[0, gel_id]))
                dof_act = float(t_.dof_pos[0, 7]) * 1e3
                pads = t_.get_luxtac_tensors_dict()
                gl = pads["luxtac_left"]["grid"][0].cpu().numpy()
                dmm = luxtac_viz.pool_subsamples(
                    pads["luxtac_left"]["depth_grid"][0].cpu().numpy(),
                    *gl.shape) * 1e3
                rows.append(dict(target_mm=ind, dof_mm=dof_mm, dof_act_mm=dof_act,
                                 F=f, d_max=float(dmm.max()),
                                 d_mean=float(dmm[dmm > 0].mean()) if (dmm > 0).any() else 0.0,
                                 s=gl.copy(), depth=dmm.copy()))
                print(f"{ind:7.2f} {dof_mm:7.3f} {dof_act:7.3f} {f:8.3f} "
                      f"{dmm.max():7.3f} "
                      f"{(dmm[dmm>0].mean() if (dmm>0).any() else 0):7.3f} "
                      f"{gl.mean():8.4f} {int((gl>0).sum()):3d}/{gl.size}")
            results[sub] = rows
            _save(results, _A.out)          # checkpoint: a later crash keeps this one

        print(f"\nsaved -> {_A.out}")
        return 0

    return run()


if __name__ == "__main__":
    sys.exit(main() or 0)
