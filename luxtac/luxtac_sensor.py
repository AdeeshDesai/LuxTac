"""
LuxTac normal-force tactile sensor for IsaacGym / TacSL.

Implements an SDF penalty model on TacSL's existing SDF query machinery:

    penetration = max(-sdf, 0)
    fn = clamp(stiffness * penetration, 0, max_force)
    if normalize_forces: fn = fn / max_force

Normal only: the LuxTac normal cartridge reads indentation and has no tangential
channel.

A mixin, like TactileFieldSensor. It subclasses TactileFieldSensor and overrides only
the grid construction and the force law. Compose it the same way:

    class MySensors(LuxTacFieldSensor, CameraSensor):
        ...
"""

import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym import torch_utils as tu
from isaacgym.torch_utils import tf_combine

from isaacgymenvs.tacsl_sensors.tacsl_sensors import TactileFieldSensor
from . import photodiode
from .luxtac_cfg import LuxTacSensorCfg


class LuxTacFieldSensor(TactileFieldSensor):
    """Penalty-based normal-force tactile field, LuxTac pad geometry."""

    def __init__(self, *args, luxtac_cfg=None, **kwargs):
        # Transparent pass-through: this class sits early in task-env MROs whose
        # __init__ chain carries (cfg, rl_device, sim_device, ...) positionally, so
        # everything except our own kwarg must flow through untouched.
        self.luxtac_cfg = luxtac_cfg if luxtac_cfg is not None else LuxTacSensorCfg()
        for w in self.luxtac_cfg.validate():
            print(f"[LuxTac] config warning: {w}")
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ taxel grid

    def generate_luxtac_points(self, cfg=None, visualize=False):
        """Build the taxel lattice analytically.

        The grid comes from (num_rows, num_cols, point_distance, normal_axis,
        normal_offset) in the patch frame. TacSL's generate_tactile_points() derives
        the pitch from the elastomer mesh bounds instead, which would not give the
        pad's pitch.

        With subsamples > 1 each taxel becomes an n x n block of points, in row-major
        order over the (sample_rows, sample_cols) grid.

        Returns:
            (pos_local, quat_local): (N, 3) and (N, 4) tensors, N = cfg.num_points.
        """
        cfg = cfg or self.luxtac_cfg

        rows = photodiode.sample_axis(cfg.num_rows, cfg.subsamples, cfg.point_distance)
        cols = photodiode.sample_axis(cfg.num_cols, cfg.subsamples, cfg.point_distance)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")

        # In-plane axes are whichever two are not the normal axis, in ascending order.
        in_plane = [a for a in range(3) if a != cfg.normal_axis]
        pts = np.zeros((cfg.sample_rows, cfg.sample_cols, 3), dtype=np.float32)
        pts[..., cfg.normal_axis] = cfg.normal_offset
        pts[..., in_plane[0]] = rr
        pts[..., in_plane[1]] = cc
        pts = pts.reshape(-1, 3)

        pos_local = torch.tensor(pts, dtype=torch.float32, device=self.device)
        quat_local = torch.zeros((pos_local.shape[0], 4), dtype=torch.float32,
                                 device=self.device)
        quat_local[:, 3] = 1.0  # identity, (x, y, z, w) as used throughout TacSL

        if visualize:
            print(f"[LuxTac] {self.luxtac_cfg.describe()}")
            extent = pts.max(axis=0) - pts.min(axis=0)
            print(f"[LuxTac] lattice extent (mm): {np.round(extent * 1e3, 2)}")

        return pos_local, quat_local

    def query_distance_and_normal_with_physx(self, query_points, visualize=False):
        """Serialize the shared SDF view tensor between pads.

        refresh_sdf_view_tensor() fills ONE buffer -- self.sdf_tensor, allocated once
        in setup_tactile_force_field -- and the parent reads it immediately with no
        synchronisation, returning views rather than copies. With both pads active the
        two refresh/read pairs race over that one buffer, and in a few percent of
        frames one pad reads the other's results.

        Synchronising after the refresh orders the read, and cloning means the next
        pad's refresh cannot overwrite values this one still holds.
        """
        # sdf_tensor is allocated for ALL pads at once. PhysX validates the query
        # shape against that capacity and REJECTS a short one -- logging to stderr
        # and leaving the buffer untouched, so a single-pad query would silently read
        # stale values rather than raise. Pad short queries up to capacity (the
        # filler points are discarded below) so every call size works.
        # A mismatched SHAPE count is the other silent failure: PhysX logs to stderr
        # and leaves the buffer stale, so the frame reads last frame's distances.
        assert query_points.shape[-3] == self.sdf_tensor.shape[-3], (
            f"{query_points.shape[-3]} shapes queried against an sdf_tensor "
            f"allocated for {self.sdf_tensor.shape[-3]}; setup_luxtac_sensors "
            "must acquire with num_shapes_per_env = len(luxtac_indenters())")
        n = query_points.shape[-2]
        cap = self.sdf_tensor.shape[-2]
        if n < cap:
            filler = query_points[..., -1:, :].expand(*query_points.shape[:-2], cap - n, 3)
            query_points = torch.cat([query_points, filler], dim=-2).contiguous()
        elif n > cap:
            raise ValueError(f"{n} query points exceeds sdf_tensor capacity {cap}; "
                             "setup_luxtac_sensors must allocate cols * n_pads")

        self.gym.refresh_sdf_view_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.sdf_shape_global_ids_per_env),
            gymtorch.unwrap_tensor(query_points))
        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            torch.cuda.synchronize()
        distance = -self.sdf_tensor[:, :, :n, 3].clone()
        normal = self.sdf_tensor[:, :, :n, :3].clone()
        return distance > 0.0, distance, normal

    def get_vt_bias(self, y_angvel):
        """Neutralize the parent's tangential-velocity bias correction.

        query_collision() subtracts a per-taxel vt bias loaded from a pickle sized
        for the GelSight 10x14 grid (140 points); with any other taxel count the
        subtraction shape-errors. The pad has no tangential channel, so the bias is
        zero.
        """
        return np.zeros((self.luxtac_cfg.num_points, 3), dtype=np.float32)

    def generate_tactile_points(self, *args, **kwargs):
        """Override of the TacSL raycast generator, so the parent's
        initialize_penalty_based_tactile() builds the analytic lattice.

        The lattice lives in the pad frame, so every raycast argument is ignored.
        """
        return self.generate_luxtac_points(self.luxtac_cfg)

    def setup_luxtac_points(self, indenter_link_id, cfg=None, visualize=False):
        """Register the LuxTac lattice for one indenter, mirroring how the parent
        class populates tactile_pos_local_dict / tactile_quat_local_dict."""
        pos_local, quat_local = self.generate_luxtac_points(cfg, visualize=visualize)
        if not hasattr(self, "tactile_pos_local_dict") or self.tactile_pos_local_dict is None:
            self.tactile_pos_local_dict = {}
            self.tactile_quat_local_dict = {}
        self.tactile_pos_local_dict[indenter_link_id] = pos_local
        self.tactile_quat_local_dict[indenter_link_id] = quat_local
        self.tactile_pos_local, self.tactile_quat_local = pos_local, quat_local
        return pos_local, quat_local

    # ------------------------------------------------------------------ force model

    def compute_luxtac_normal_force(self, penetration, cfg=None):
        """The penalty law, isolated so it can be unit-tested without a sim.

        Args:
            penetration: (..., N) tensor of penetration depths in metres, >= 0.
        Returns:
            (fn_newtons, fn_normalized) -- the second is fn/max_force if
            normalize_forces else a copy of the first.
        """
        cfg = cfg or self.luxtac_cfg
        fn = (cfg.stiffness * penetration).clamp(0.0, cfg.max_force)
        if cfg.normalize_forces:
            return fn, fn / cfg.max_force
        return fn, fn.clone()

    def read_luxtac_pad(self, penetration, cfg=None):
        """Turn one pad's sample-point depths into its readout.

        Args:
            penetration: (envs, cfg.num_points) depths in metres, >= 0.
        Returns:
            dict with penetration (m), depth_grid (envs, sample_rows, sample_cols),
            fn (N per taxel), drop, normalized, grid (envs, rows, cols), and counts.
            grid is what the policy sees: the driver's normalized s in [0, 1] for the
            photodiode response (photodiode.normalize), fn/max(fn) over the pad for
            linear_peak (photodiode.normalize_peak), and fn/max_force for the spring.
            drop is the un-normalized fractional drop, photodiode only.
        """
        cfg = cfg or self.luxtac_cfg
        shape = (cfg.num_rows, cfg.num_cols, cfg.subsamples)
        if cfg.response == "photodiode":
            if getattr(self, "_luxtac_curves", None) is None:
                self._luxtac_curves = {
                    k: torch.tensor(v, dtype=torch.float32, device=penetration.device)
                    for k, v in photodiode.curve_maps(
                        cfg.num_rows, cfg.num_cols, cfg.per_taxel_curves).items()}
            drop, fn = photodiode.read_pad(penetration, *shape, self._luxtac_curves)
            grid = photodiode.normalize(drop, self._luxtac_curves)
            # Sample points past the depth clip. Note s saturates EARLIER
            # than the clip: it reaches 1 at 1.67-1.92mm depending on the taxel, so
            # num_clipped is the count that matters for the observation.
            num_saturated = int((penetration * 1e3 >= photodiode.DEPTH_CLIP_MM).sum().item())
        elif cfg.response == "linear_peak":
            if getattr(self, "_luxtac_lin_curves", None) is None:
                self._luxtac_lin_curves = {
                    k: torch.tensor(v, dtype=torch.float32, device=penetration.device)
                    for k, v in photodiode.linear_curve_maps(
                        cfg.num_rows, cfg.num_cols, cfg.per_taxel_curves).items()}
            c = self._luxtac_lin_curves
            fn, d_tax = photodiode.read_pad_linear(penetration, *shape, c)
            # Per-pad normalisation. When cfg.peak_scope == "frame" and more than one
            # pad is read, get_luxtac_all_pads OVERWRITES this with a single divisor
            # taken across every pad; see renormalize_pads_together. This stays here so
            # a lone pad still gets a usable grid.
            grid = photodiode.normalize_peak(
                fn, cfg.peak_floor_force, cfg.peak_floor_scale,
                d_tax if cfg.peak_gate else None, c if cfg.peak_gate else None)
            drop = None
            num_saturated = int((penetration * 1e3 >= photodiode.DEPTH_CLIP_MM).sum().item())
        else:
            fn_s, norm_s = self.compute_luxtac_normal_force(penetration, cfg)
            fn = photodiode.pool_to_taxels(fn_s, *shape)
            grid = photodiode.pool_to_taxels(norm_s, *shape)
            drop = None
            d_tax = None
            num_saturated = int((fn_s >= cfg.max_force - 1e-9).sum().item())
        if cfg.response == "photodiode":
            d_tax = None
        return {
            "penetration": penetration,
            "depth_grid": penetration.view(-1, cfg.sample_rows, cfg.sample_cols),
            "fn": fn,
            "depth_taxel": d_tax,
            "drop": drop,
            "normalized": grid,
            "grid": grid,
            "num_contacting": int((penetration > 0).sum().item()),
            "num_saturated": num_saturated,
            "num_clipped": int((grid >= 1.0).sum().item()),
        }

    def get_luxtac_normal_forces(self, indenter_link_id, elastomer_link_id, cfg=None):
        """Full per-frame readout for one pad.

        Structured to parallel TactileFieldSensor.get_penalty_based_tactile_forces(),
        minus everything tangential: vt is not requested from query_collision(), and no
        friction cone is applied.

        Returns:
            dict with penetration (m), fn (N), normalized [0,1], normal_world,
            and the (rows, cols) reshaped normalized grid.
        """
        cfg = cfg or self.luxtac_cfg

        self.tactile_pos_local = self.tactile_pos_local_dict[indenter_link_id]
        self.tactile_quat_local = self.tactile_quat_local_dict[indenter_link_id]
        self.sdf = self.sdf_dict[indenter_link_id]
        self.indenter_mesh = self.indenter_mesh_dict[indenter_link_id]
        self.indenter_mesh_local_tf = self.indenter_mesh_local_tf_dict[indenter_link_id]
        self.sdf_shape_global_ids_per_env = self.sdf_shape_global_ids_per_env_dict[indenter_link_id]

        sdf_tf = tf_combine(self.body_quat[:, indenter_link_id],
                            self.body_pos[:, indenter_link_id],
                            self.indenter_mesh_local_tf[0].expand(self.num_envs, 4),
                            self.indenter_mesh_local_tf[1].expand(self.num_envs, 3))
        sdf_angvel_world = self.body_angvel[:, indenter_link_id]
        sdf_linvel_world = self.body_linvel[:, indenter_link_id]

        self.tactile_pos_world, self.tactile_quat_world = self.get_tactile_points_in_world(
            self.tactile_pos_local, self.tactile_quat_local, elastomer_link_id)

        # Velocities are still passed because query_collision()'s signature requires
        # them, but only `depth` and `normal_world` are consumed below.
        tactile_velocity_world = self.get_tactile_points_velocities(elastomer_link_id)
        depth, _depth_dot, normal_world, _vt_world = self.query_collision(
            self.sdf, sdf_tf, sdf_linvel_world, sdf_angvel_world,
            self.tactile_pos_world, tactile_velocity_world)

        return dict(self.read_luxtac_pad(depth.clamp(min=0.0), cfg),
                    normal_world=normal_world)

    def get_luxtac_all_pads(self, pads, cfg=None):
        """Query every pad in ONE SDF refresh, and split the result.

        Both pads probe the same indenter, so their query points can be concatenated
        into a single refresh_sdf_view_tensor() call. That removes the alternation
        that races over the shared sdf_tensor buffer -- measured at ~8% of queries
        returning the other pad's data, and not fixable by CUDA-stream
        synchronisation because PhysX does not order against the torch stream.

        Requires sdf_tensor to have been allocated for len(pads) * cfg.num_points
        queries; setup_luxtac_sensors does that.

        Args:
            pads: ordered {name: {'indenter_link_rb_id', 'pad_link_rb_id'}}
        Returns:
            {name: result dict}, same shape as get_luxtac_normal_forces returns.
        """
        cfg = cfg or self.luxtac_cfg
        names = list(pads)
        ind_ids = self._luxtac_indenter_ids
        ref = ind_ids[0]

        # The lattice is a property of the PAD, not of what it touches -- our
        # generate_tactile_points ignores the indenter entirely -- so indenter 0's
        # entry is the lattice for all of them.
        self.tactile_pos_local = self.tactile_pos_local_dict[ref]
        self.tactile_quat_local = self.tactile_quat_local_dict[ref]

        n_pts = self.tactile_pos_local.shape[0]
        world = []
        for n in names:
            pw, _ = self.get_tactile_points_in_world(
                self.tactile_pos_local, self.tactile_quat_local, pads[n]['pad_link_rb_id'])
            world.append(pw)
        all_world = torch.cat(world, dim=1)                      # (envs, n_pads*N, 3)
        total = all_world.shape[1]

        # The SAME world points, re-expressed in each indenter's own SDF frame. One
        # slot of the query buffer per indenter; PhysX takes shape_ids of
        # (envs, n_shapes) and query_points of (envs, n_shapes, n_queries, 3).
        per_shape = []
        for ind_id in ind_ids:
            mesh_tf = self.indenter_mesh_local_tf_dict[ind_id]
            sdf_tf = tf_combine(self.body_quat[:, ind_id], self.body_pos[:, ind_id],
                                mesh_tf[0].expand(self.num_envs, 4),
                                mesh_tf[1].expand(self.num_envs, 3))
            tf_inv = tu.tf_inverse(sdf_tf[0], sdf_tf[1])
            per_shape.append(tu.tf_apply(
                tf_inv[0].unsqueeze(1).expand(self.num_envs, total, 4),
                tf_inv[1].unsqueeze(1).expand(self.num_envs, total, 3),
                all_world))
        pts_sdf = torch.stack(per_shape, dim=1).contiguous()     # (envs, n_shapes, total, 3)
        self.sdf_shape_global_ids_per_env = self._luxtac_sdf_ids

        _, distance, _ = self.query_distance_and_normal_with_physx(pts_sdf)
        depth_per = distance.clamp(min=0.0)                      # (envs, n_shapes, total)

        # The union, exactly: the SDF of a union of rigid bodies is min(sdf_A, sdf_B),
        # so penetration is max(-sdf_A, -sdf_B), and clamping at zero commutes with
        # max. Summing instead would double-count a taxel touched by two objects, and
        # the readout is nonlinear in depth, so depths are not additive.
        depth_all = depth_per.max(dim=1).values if len(ind_ids) > 1 else depth_per[:, 0]

        out = {}
        for i, n in enumerate(names):
            out[n] = self.read_luxtac_pad(depth_all[:, i * n_pts:(i + 1) * n_pts], cfg)
        out = self.renormalize_pads_together(out, cfg)
        if len(ind_ids) > 1:
            # Invariant: only one object is in the gripper, so no sample point may be
            # inside two of them at once.
            both = ((depth_per > 0).sum(dim=1) > 1).sum()
            out['num_multi_indenter_points'] = int(both.item())
        return out

    def renormalize_pads_together(self, out, cfg=None):
        """linear_peak, peak_scope="frame": one divisor across every pad.

        Two pads gripping one object carry near-equal and opposite normal force, so
        the ratio between their readings is physical. Per-pad normalisation pins each
        pad's max to 1.0 and destroys it. Taking the peak over the whole frame keeps
        the ratio and leaves the pattern within each pad untouched.

        No-op for every other response, for peak_scope="pad", and for a single pad.
        """
        cfg = cfg or self.luxtac_cfg
        names = list(out)
        if (cfg.response != "linear_peak" or getattr(cfg, "peak_scope", "frame") != "frame"
                or len(names) < 2):
            return out
        fn = torch.stack([out[n]["fn"] for n in names], dim=1)          # (envs, pads, r, c)
        c = self._luxtac_lin_curves
        d_tax = (torch.stack([out[n]["depth_taxel"] for n in names], dim=1)
                 if cfg.peak_gate else None)
        grid = photodiode.normalize_peak(
            fn, cfg.peak_floor_force, cfg.peak_floor_scale,
            d_tax, c if cfg.peak_gate else None)
        for i, n in enumerate(names):
            g = grid[:, i]
            out[n]["normalized"] = out[n]["grid"] = g
            out[n]["num_clipped"] = int((g >= 1.0).sum().item())
        return out

    def get_luxtac_force_fields(self, cfg=None):
        """Read every configured pad, keyed like get_tactile_shear_force_fields()."""
        out = {}
        for key, config in self.tactile_shear_field_configs_dict.items():
            out[key] = self.get_luxtac_normal_forces(
                config["indenter_link_rb_id"], config["elastomer_link_rb_id"], cfg)
        return out

    # -------------------------------------------------------------------- overrides

    def get_penalty_based_tactile_forces(self, indenter_link_id, elastomer_link_id):
        """Drop-in replacement for the parent's 3-tuple contract.

        Returns (penetration, normal_force, shear_force) where shear is an explicit
        zero tensor of the right shape. Anything downstream that unpacks three values
        keeps working; anything that reads the shear channel will correctly see no
        tangential signal rather than a stale or fabricated one.
        """
        res = self.get_luxtac_normal_forces(indenter_link_id, elastomer_link_id)
        normal_force = res["normalized"] if self.luxtac_cfg.normalize_forces else res["fn"]
        shear_force = torch.zeros((*normal_force.shape, 2), device=normal_force.device,
                                  dtype=normal_force.dtype)
        return res["penetration"], normal_force, shear_force
