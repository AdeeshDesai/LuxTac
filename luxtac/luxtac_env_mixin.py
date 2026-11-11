"""Env-side wiring for the LuxTac sensor.

LuxTacSensors is a mixin an env composes to get tactile sensing, as TacSLSensors is
in tacsl_env_insertion.py. It uses the frames defined in
assets/luxtac/urdf/luxtac_franka_gel45.urdf:

    luxtac_pad_left / luxtac_pad_right          lattice attach frames
    env.luxtac.compliance_links                 collision bodies carrying the compliance

    self.setup_luxtac_sensors()                 once, in create_envs() after actors exist
    obs = self.get_luxtac_tensors_dict()        per step

Task-config keys (see LuxTacTaskGrasp.yaml):
    env.luxtac.num_rows / num_cols / point_distance / subsamples / normal_offset
    env.luxtac.response / per_taxel_curves, or stiffness / max_force for the spring
    env.luxtac.compliance_stiffness / compliant_damping / compliance_links
"""

from isaacgym import gymapi, gymtorch   # must precede torch
import torch

from .luxtac_cfg import LuxTacSensorCfg
from .luxtac_sensor import LuxTacFieldSensor


class LuxTacSensors(LuxTacFieldSensor):
    """Mixin wiring LuxTacFieldSensor into a TacSL-style env."""

    # ------------------------------------------------------------------ config

    def _luxtac_cfg_from_task_cfg(self):
        """Build a LuxTacSensorCfg from cfg_task.env.luxtac."""
        ft = getattr(self.cfg_task.env, "luxtac", None)
        get = (lambda k, d: getattr(ft, k, d)) if ft is not None else (lambda k, d: d)
        return LuxTacSensorCfg(
            num_rows=int(get("num_rows", 8)),
            num_cols=int(get("num_cols", 8)),
            point_distance=float(get("point_distance", 0.002)),
            subsamples=int(get("subsamples", 1)),
            response=str(get("response", "spring")),
            per_taxel_curves=bool(get("per_taxel_curves", True)),
            normal_axis=1,                      # +y_pad is the pad normal by URDF convention
            # Sensing plane at the gel surface: grasp penetration is sub-millimetre,
            # so an inward offset would act as a dead zone.
            normal_offset=float(get("normal_offset", 0.0)),
            stiffness=float(get("stiffness", 5000.0)),
            max_force=float(get("max_force", 10.0)),
            # linear_peak only; ignored by the other two responses.
            peak_floor_force=float(get("peak_floor_force", 0.420)),
            peak_floor_scale=float(get("peak_floor_scale", 1.680)),
            peak_gate=bool(get("peak_gate", True)),
            peak_scope=str(get("peak_scope", "frame")),
            sdf_tool=str(get("sdf_tool", "physx")),
        )

    def _luxtac_compliance_links(self):
        """(left, right) bodies that carry the contact compliance.

        TacSL's configure_compliant_dynamics sets only the FIRST shape of a link, so a
        gel block has to be its own link, as the GelSight elastomer is. Fingers without
        one fall back to the finger bodies.
        """
        ft = getattr(self.cfg_task.env, "luxtac", None)
        links = getattr(ft, "compliance_links", None) if ft is not None else None
        return tuple(links) if links else ('panda_leftfinger', 'panda_rightfinger')

    def luxtac_indenters(self):
        """The objects the pads measure against, in a fixed order. Override to add more.

        One entry per object: {'actor_name', 'urdf_path'}. The default is the single
        'plug' actor of the insertion tasks. A task with several objects returns them
        all, and the sensor takes the union of their SDFs (see get_luxtac_all_pads);
        an object away from the gripper contributes zero depth.
        """
        return [{'actor_name': 'plug', 'urdf_path': self.asset_file_paths["plug"]}]

    def _compose_luxtac_configs(self):
        """Analogue of _compose_tactile_force_field_configs() in the TacSL env.

        elastomer_link_name is the link compliance is set on; the pad marker link is
        carried separately and is what the lattice attaches to.

        Emits one config per pad x indenter, because initialize_penalty_based_tactile
        builds its SDF oracle per config; setup_luxtac_sensors collapses them back to
        two pads.
        """
        ft = getattr(self.cfg_task.env, "luxtac", None)
        get = (lambda k, d: getattr(ft, k, d)) if ft is not None else (lambda k, d: d)
        left_link, right_link = self._luxtac_compliance_links()

        base = dict([
            ('elastomer_actor_name', 'franka'),
            ('elastomer_parent_urdf_path', self.asset_file_paths["franka"]),
            ('actor_handle', self.actor_handles['franka']),
            ('compliance_stiffness', float(get("compliance_stiffness", 500.0))),
            ('compliant_damping', float(get("compliant_damping", 1.0))),
            ('use_acceleration_spring', False),
        ])
        pads = [
            dict(base, name='luxtac_left', elastomer_link_name=left_link,
                 elastomer_tip_link_name='luxtac_pad_tip_left',
                 pad_link_name='luxtac_pad_left'),
            dict(base, name='luxtac_right', elastomer_link_name=right_link,
                 elastomer_tip_link_name='luxtac_pad_tip_right',
                 pad_link_name='luxtac_pad_right'),
        ]
        out = []
        for k, ind in enumerate(self.luxtac_indenters()):
            actor_id_env = self.gym.find_actor_index(
                self.env_ptrs[0], ind['actor_name'], gymapi.DOMAIN_ENV)
            assert actor_id_env > -1, f"indenter actor {ind['actor_name']} not found"
            rb_names = self.gym.get_actor_rigid_body_names(self.env_ptrs[0], actor_id_env)
            for pad in pads:
                # Indenter 0 keeps the bare pad name, which is the observation key.
                out.append(dict(pad,
                                name=pad['name'] if k == 0 else f"{pad['name']}#{k}",
                                pad_name=pad['name'],
                                indenter_index=k,
                                indenter_urdf_path=ind['urdf_path'],
                                indenter_actor_name=ind['actor_name'],
                                indenter_link_name=ind.get('link_name', rb_names[0])))
        return out

    # ------------------------------------------------------------------- setup

    def set_elastomer_compliance(self, compliance_stiffness, compliant_damping):
        """Override of the TacSL env hook (called from _create_actors), which targets
        the GelSight elastomer_left/right links."""
        for link in self._luxtac_compliance_links():
            self._match_finger_surface(link)
            self.configure_compliant_dynamics(actor_handle=self.actor_handles['franka'],
                                              elastomer_link_name=link,
                                              compliance_stiffness=compliance_stiffness,
                                              compliant_damping=compliant_damping,
                                              use_acceleration_spring=False)

    def _match_finger_surface(self, link):
        """Give a gel link the surface properties the env sets on the finger shapes.

        The env sets friction on the first shape of the finger links only, and the
        object touches the gel.
        """
        handle = self.actor_handles['franka']
        for env_ptr in self.env_ptrs:
            body_id = self.gym.get_actor_rigid_body_dict(env_ptr, handle)[link]
            shape_id = self.gym.get_actor_rigid_body_shape_indices(env_ptr, handle)[body_id].start
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            props[shape_id].friction = self.cfg_base.env.franka_friction
            props[shape_id].rolling_friction = 0.0
            props[shape_id].torsion_friction = 0.0
            props[shape_id].restitution = 0.0
            props[shape_id].thickness = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, handle, props)

    def setup_luxtac_sensors(self):
        """Call once from create_envs() after actors exist. Mirrors the sequence the
        TacSL env runs for its force field: compliance -> sdf tensor -> lattice."""
        self.luxtac_cfg = self._luxtac_cfg_from_task_cfg()
        for w in self.luxtac_cfg.validate():
            print(f"[LuxTac] config warning: {w}")
        print(f"[LuxTac] {self.luxtac_cfg.describe()}")

        luxtac_configs = self._compose_luxtac_configs()
        self.set_compliant_dynamics_for_tactile_sensors(luxtac_configs)

        self.sdf_tool = self.luxtac_cfg.sdf_tool
        n_ind = len(self.luxtac_indenters())
        n_pads = len({c['pad_name'] for c in luxtac_configs})

        # One SDF view tensor, (envs, n_indenters, n_pads*points_per_pad, 4), so a whole
        # frame is a single refresh_sdf_view_tensor call. Separate refreshes per pad or
        # per indenter over a shared buffer race, and a few percent of frames then read
        # the other pad's distances (see get_luxtac_all_pads).
        #
        # setup_tactile_force_field hardcodes num_shapes_per_env=1, so the config
        # post-processing is reused and the acquire is done here. Must stay before
        # gym.prepare_sim.
        self.tactile_shear_field_configs_dict = self.post_process_shear_field_configs(
            luxtac_configs)
        self.sdf_tensor = None
        if self.sdf_tool == 'physx':
            self.sdf_tensor = gymtorch.wrap_tensor(self.gym.acquire_sdf_view_tensor(
                self.sim, n_ind, self.luxtac_cfg.num_points * n_pads))

        # Build the per-indenter SDF oracles, lattice dicts and lookup state the TacSL
        # query path expects. generate_tactile_points is overridden in
        # LuxTacFieldSensor, so this registers the analytic lattice, not a raycast.
        #
        # Every config entry loads its indenter mesh, and the outputs are keyed by
        # indenter_link_rb_id, so pass one config per indenter and then restore the
        # full dict.
        full_cfgs = self.tactile_shear_field_configs_dict
        seen, dedup = set(), {}
        for name, c in full_cfgs.items():
            if c['indenter_link_rb_id'] in seen:
                continue
            seen.add(c['indenter_link_rb_id'])
            dedup[name] = c
        self.tactile_shear_field_configs_dict = dedup
        try:
            self.initialize_penalty_based_tactile(
                num_divs=[self.luxtac_cfg.sample_rows, self.luxtac_cfg.sample_cols])
        finally:
            self.tactile_shear_field_configs_dict = full_cfgs

        # Collapse the pad x indenter configs back to one entry per pad, in order,
        # plus one indenter list shared by every pad.
        self._luxtac_pads = {}
        self._luxtac_indenter_ids = [None] * n_ind
        for config in self.tactile_shear_field_configs_dict.values():
            self._luxtac_indenter_ids[config['indenter_index']] = \
                config['indenter_link_rb_id']
            if config['indenter_index'] != 0:
                continue
            pad_link = config['pad_link_name']
            pad_rb_id = self.gym.find_actor_rigid_body_handle(
                self.env_ptrs[0], self.actor_handles['franka'], pad_link)
            assert pad_rb_id > -1, f"pad link {pad_link} not found; " \
                "is franka_urdf_file set to a LuxTac franka URDF?"
            self._luxtac_pads[config['pad_name']] = {'pad_link_rb_id': pad_rb_id}
        assert all(i is not None for i in self._luxtac_indenter_ids)
        assert len(self._luxtac_pads) == n_pads

        # (envs, n_indenters) of PhysX global shape ids, the column order matching
        # _luxtac_indenter_ids, so one refresh covers every object.
        self._luxtac_sdf_ids = torch.cat(
            [self.sdf_shape_global_ids_per_env_dict[i]
             for i in self._luxtac_indenter_ids], dim=1).contiguous()
        print(f"[LuxTac] {n_pads} pads x {n_ind} indenter(s) "
              f"{[c['indenter_actor_name'] for c in luxtac_configs[::n_pads]]}, "
              f"one SDF refresh of {tuple(self.sdf_tensor.shape)}"
              if self.sdf_tensor is not None else "[LuxTac] non-physx sdf tool")

    # ----------------------------------------------------------------- per-step

    def get_luxtac_tensors_dict(self):
        """Per-step observation dict, one (num_envs, rows, cols) grid per pad, plus a
        stacked (num_envs, num_pads, rows, cols) tensor under 'luxtac'."""
        out = self.get_luxtac_all_pads(self._luxtac_pads, self.luxtac_cfg)
        stacked = [out[n]['grid'] for n in self._luxtac_pads]
        out['luxtac'] = torch.stack(stacked, dim=1) if stacked else None
        return out
