"""Configuration for the LuxTac tactile pad: an SDF-penalty, normal-only sensor."""

from dataclasses import dataclass, field


@dataclass
class LuxTacSensorCfg:
    """Penalty-based normal-force tactile sensor, LuxTac pad geometry."""

    # ------------------------------------------------------------------ taxel grid
    num_rows: int = 12
    num_cols: int = 32
    point_distance: float = 0.002        # m, taxel pitch

    # Sample points per taxel, per axis. Each taxel averages an n x n grid across its
    # cell, so a contact between two taxel centres still registers. 1 = centre only.
    subsamples: int = 1

    # "spring":      fn = clamp(stiffness * depth, 0, max_force).
    # "photodiode":  the bench curve in photodiode.py, reported as the real driver
    #                does: s = clip((1 - raw/S0) / (R/b), 0, 1), gated at 3 sigma.
    # "linear_peak": fn = k * depth with the bench-fitted per-taxel k (N/mm), then
    #                per-frame peak normalisation, s = fn / max(fn). Only the pattern
    #                survives; absolute magnitude does not.
    response: str = "spring"
    # photodiode and linear_peak: each taxel's own equations, or the average of all.
    per_taxel_curves: bool = True

    # ------------------------------------------------- linear_peak only
    # A frame whose peak is below peak_floor_force is divided by peak_floor_scale
    # instead of by its own peak, so a light touch is not normalised up to 1.0.
    peak_floor_force: float = 0.420      # N
    peak_floor_scale: float = 1.680      # N
    # Zero a taxel shallower than its own 3-sigma deadband first, as the hardware does.
    peak_gate: bool = True
    # "frame": one divisor across both pads, which keeps their relative magnitude.
    # "pad":   each pad normalised on its own, as a standalone sensor would be.
    peak_scope: str = "frame"

    # Local axis of the pad normal (0=x, 1=y, 2=z) and how far along it to sink the
    # sample plane (negative = inside the finger surface).
    normal_axis: int = 1
    normal_offset: float = -0.0032       # m

    # ------------------------------------------------------- patch pose in body frame
    patch_offset_pos_b: tuple = (0.0, 0.0, 0.0)
    patch_offset_quat_b: tuple = (1.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------- force computation
    # fn = clamp(stiffness * penetration, 0, max_force); optionally / max_force.
    # Only the ratio d_sat = max_force / stiffness matters when normalize_forces is True.
    stiffness: float = 5_000.0           # N/m
    max_force: float = 10.0              # N
    normalize_forces: bool = True

    # ------------------------------------------------------------------- SDF backend
    # Passed to TacSL's query_collision(). 'physx' runs on the GPU; 'pysdf' and
    # 'trimesh' run on the CPU.
    sdf_tool: str = "physx"

    # ------------------------------------------------------------------ debug / viz
    debug_vis_force_threshold: float = 1.0e-6

    @property
    def num_taxels(self) -> int:
        return self.num_rows * self.num_cols

    @property
    def sample_rows(self) -> int:
        return self.num_rows * self.subsamples

    @property
    def sample_cols(self) -> int:
        return self.num_cols * self.subsamples

    @property
    def num_points(self) -> int:
        """Sample points on one pad."""
        return self.sample_rows * self.sample_cols

    @property
    def saturation_depth(self) -> float:
        """Penetration in metres at which the sensor saturates."""
        return self.max_force / self.stiffness

    @property
    def pad_extent(self) -> tuple:
        """(rows, cols) physical extent in metres at the configured pitch."""
        return (self.num_rows * self.point_distance,
                self.num_cols * self.point_distance)

    def validate(self) -> list:
        """Return a list of human-readable warnings; empty if nothing looks wrong."""
        warnings = []
        if self.normal_axis not in (0, 1, 2):
            warnings.append(f"normal_axis must be 0/1/2, got {self.normal_axis}")
        if self.subsamples < 1:
            warnings.append(f"subsamples must be >= 1, got {self.subsamples}")
        if self.response not in ("spring", "photodiode", "linear_peak"):
            warnings.append("response must be spring, photodiode or linear_peak, "
                            f"got {self.response}")
        if self.response == "linear_peak":
            if self.peak_scope not in ("frame", "pad"):
                warnings.append(f"peak_scope must be frame or pad, got {self.peak_scope}")
            if self.peak_floor_force <= 0.0:
                warnings.append(f"peak_floor_force must be > 0, got {self.peak_floor_force}")
            if self.peak_floor_scale <= 0.0:
                warnings.append(f"peak_floor_scale must be > 0, got {self.peak_floor_scale}")
            if self.peak_floor_scale < self.peak_floor_force:
                warnings.append(
                    f"peak_floor_scale={self.peak_floor_scale} is below "
                    f"peak_floor_force={self.peak_floor_force}: an untouched frame "
                    "would read HIGHER than a just-touched one")
        if self.stiffness <= 0.0:
            warnings.append(f"stiffness must be > 0, got {self.stiffness}")
        if self.max_force <= 0.0:
            warnings.append(f"max_force must be > 0, got {self.max_force}")
        if self.normal_offset > 0.0:
            warnings.append(
                f"normal_offset={self.normal_offset} is positive; the sample plane "
                "should sit *inside* the surface, with a negative value")
        if self.response == "spring" and abs(self.normal_offset) < self.saturation_depth:
            warnings.append(
                f"|normal_offset|={abs(self.normal_offset)*1e3:.2f} mm is smaller than "
                f"d_sat={self.saturation_depth*1e3:.2f} mm; the sample plane can exit "
                "the far side of the finger before the sensor saturates")
        return warnings

    def describe(self) -> str:
        rows_mm, cols_mm = (e * 1e3 for e in self.pad_extent)
        head = (
            f"LuxTac pad: {self.num_rows}x{self.num_cols} = {self.num_taxels} taxels "
            f"@ {self.point_distance*1e3:.1f} mm pitch ({rows_mm:.1f} x {cols_mm:.1f} mm), "
            f"{self.subsamples}x{self.subsamples} samples per taxel = {self.num_points} points\n"
            f"  normal: local axis {self.normal_axis}, offset {self.normal_offset*1e3:+.2f} mm\n")
        if self.response == "photodiode":
            return head + f"  signal: bench curve, per_taxel_curves={self.per_taxel_curves}\n  sdf   : {self.sdf_tool}"
        if self.response == "linear_peak":
            return (head +
                    f"  signal: fn = k * d, bench-fitted k, "
                    f"per_taxel_curves={self.per_taxel_curves}\n"
                    f"  norm  : peak over the {self.peak_scope}; floor "
                    f"{self.peak_floor_force:.3f} N -> /{self.peak_floor_scale:.3f} N, "
                    f"gate={self.peak_gate}\n"
                    f"  sdf   : {self.sdf_tool}")
        return (
            head +
            f"  force : fn = clamp({self.stiffness:.0f} * pen, 0, {self.max_force:.1f} N)"
            f"{' / ' + format(self.max_force, '.1f') if self.normalize_forces else ''}\n"
            f"  d_sat : {self.saturation_depth*1e3:.2f} mm\n"
            f"  sdf   : {self.sdf_tool}"
        )


# Spring presets: d_sat = 0.48 mm and d_sat = 2.00 mm.
UPSTREAM_CFG_DEFAULT = LuxTacSensorCfg(max_force=2.4)
UPSTREAM_CFG_ALOHA = LuxTacSensorCfg(max_force=10.0)

# The 8x4 photodiode pad, read as a linear spring and peak-normalised.
LUXTAC_CFG_LINEAR_PEAK = LuxTacSensorCfg(
    num_rows=4, num_cols=8, response="linear_peak")
