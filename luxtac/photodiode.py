"""Bench-calibrated response of the in-lab LuxTac photodiode taxel.

Sensor stack, finger outwards: photodiodes, 0.5 mm acrylic, a few mm of silicone gel.
Pressing the gel lowers the photodiode reading, so the raw signal is the fractional drop

    drop = 1 - raw / S0        (S0 = that taxel's reading at the start of the episode)

Two independent bench characterisations feed this module, both on the same pad.

1. calibration/photodiode_15pass_20261101.json -- our fit, one pair of equations per
   taxel over every frame of all 266 presses across 7 sessions, load and unload:

       drop = c * d^e
       F    = a * d^b             newtons

   d is depth in mm from the bench zero at the gel surface. Every press ends on the 6 N
   force limit rather than the configured 2.5 mm depth limit -- median peak depth
   1.85 mm, deepest 2.10 mm, none of the 266 past 2.4 mm -- so depth is clipped at
   1.9 mm and widening that clip would be extrapolating past the data.

   A single-session fit (68 presses, photodiode_run_20261101.json) gives the same
   drop curve (c 0.1001 vs 0.1004, e 1.342 vs 1.336) and a 6% lower force gain a.

2. calibration/luxtac_calib_20261103.csv -- an independent CNC/ATI characterisation of the
   same 16 taxels, giving each one's drop in counts at ~6 N (R), its noise (sigma) and
   its baseline (b). This is what the real driver (luxtac.py, MARS Lab) ships, and it
   defines the observation:

       s = clip((b - x) / R, 0, 1),    s = 0 where (b - x) < 3*sigma

   Since drop = (b - x)/b, that is exactly s = clip(drop / (R/b), 0, 1) with a gate at
   3*sigma/b. R is rescaled by the live baseline on the real sensor, so R/b is constant
   and the sim needs only the ratio: no counts, no S0, no LED drift. See normalize().

3. calibration/photodiode_linear_20261104.json -- the same 266 presses refitted as a
   STRAIGHT line per taxel, F = k * (d - z0)+, for the peak-normalised readout
   (response="linear_peak"). Refitting shows the b = 1.48 exponent in fit 1 is
   mostly an artifact of forcing the stage zero onto contact: free the offset and the
   exponent falls to 1.17, and at equal parameter count the straight line ties the
   power law on 1.06 M frames. k spans 3.45-3.75 N/mm, +-2.3% across the pad.

   z0 (0.15-0.34 mm) is where the bench stage sat above the gel surface. It is an
   instrument offset, not a gel property, and sim penetration is an SDF distance from
   the surface itself, so ONLY k transfers. Fitting through the origin instead returns
   2.93 N/mm, a 19% underestimate.

The two agree, which is the main reason to trust either. Feeding fit 1 into rule 2:
s reaches 1 at a depth of 1.67-1.92 mm across the 16 taxels (mean 1.78), just inside
the 1.9 mm clip, and the force there is 5.46-5.92 N (mean 5.77) against the driver's
"s = 1 is about 6 N" and its s = 0.02 + 0.170*F line, which puts s = 1 at 5.76 N.
R/b averages 0.2187 against the README's 0.22 +- 0.03. The 3*sigma gate is a deadband
of about 60 um of indentation.

Only one 4x4 block was calibrated. The second block of the 8x4 sensor copies it, which
is an assumption. Each taxel averages its curve over its full 4.5 mm cell, which is
also an assumption: every bench press was at a taxel centre, and the real pitch is
4.415 x 4.796 mm rather than isotropic 4.5.

LAYOUT. curve_maps() puts bench S01 at (row 0, col 0) and counts DOWN the rows first,
so sim row 0 is S01/S05/S09/S13. The driver's pad.read()['s'][0] is S01..S04. Each 4x4
block is therefore the transpose of the driver's grid; transpose before comparing sim
to hardware.

Pure torch and numpy, with no IsaacGym import, so verify_analytic can test the real code.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).parent / "calibration"
_CAL = json.loads((_DIR / "photodiode_15pass_20261101.json").read_text())

_FIT = _CAL["per_taxel"]
AVERAGE = {k: float(_FIT["average"][k]) for k in "abce"}
PER_TAXEL = {k: np.asarray(_FIT[k], dtype=np.float32) for k in "abce"}

NOISE_K = 3.0               # luxtac.py NOISE_K: the gate is 3 sigma below baseline
R_FRAC = 0.22               # luxtac.py R_FRAC, its fallback when no per-pad file exists


def _load_luxtac_calib(path):
    """R/baseline and the 3*sigma/baseline gate, both in drop units, per taxel."""
    n = len(PER_TAXEL["c"])
    r, gate = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            k = int(row["taxel"]) - 1
            b = float(row["baseline"])
            r[k] = float(row["R_counts"]) / b
            gate[k] = NOISE_K * float(row["sigma"]) / b
    if not (r > 0).all():
        raise ValueError(f"{path} does not cover all {n} taxels")
    return r, gate


PER_TAXEL["r"], PER_TAXEL["n"] = _load_luxtac_calib(_DIR / "luxtac_calib_20261103.csv")
# The uniform maps stand in for the driver's calib=None path, which uses one R_FRAC for
# every taxel. sigma is still measured live there, so the gate stays the measured mean.
AVERAGE["r"] = R_FRAC
AVERAGE["n"] = float(PER_TAXEL["n"].mean())

_LIN = json.loads((_DIR / "photodiode_linear_20261104.json").read_text())
LINEAR_PER_TAXEL = {
    "k": np.asarray(_LIN["per_taxel"]["k_N_per_mm"], dtype=np.float32),
    "gate": np.asarray(_LIN["per_taxel"]["gate_depth_mm"], dtype=np.float32),
}
LINEAR_AVERAGE = {"k": float(_LIN["shared"]["k_N_per_mm"]),
                  "gate": float(_LIN["shared"]["gate_depth_mm"])}

DEPTH_CLIP_MM = 1.9
FULL_SCALE_DROP = 0.3       # raw-drop scale, for plotting drop rather than s


def sample_axis(num_taxels, subsamples, pitch):
    """Sample positions along one axis, centred on the pad.

    Each taxel is a cell one pitch wide, split into `subsamples` equal parts with a
    point at the middle of each. subsamples=1 gives the taxel centres.
    """
    index = np.arange(num_taxels * subsamples)
    return ((index + 0.5) / subsamples - num_taxels / 2.0) * pitch


def curve_maps(num_rows, num_cols, per_taxel=True):
    """{a, b, c, e, r, n}, each a (rows, cols) array.

    a, b, c, e are the fitted curves; r and n are the driver's normalization constants
    (see normalize). Bench numbering: S01 at (+x, +y), rising towards -x, then towards
    -y. Bench x is taken as across the finger (rows) and bench y as along it (cols).
    Block 2 copies block 1, further along the finger. per_taxel=False gives the average
    of all 16, and for r the driver's uniform 0.22.
    """
    if not per_taxel:
        return {k: np.full((num_rows, num_cols), v, dtype=np.float32)
                for k, v in AVERAGE.items()}
    if num_rows != 4 or num_cols % 4:
        raise ValueError("per-taxel curves need 4 rows and blocks of 4 cols, "
                         f"got {num_rows}x{num_cols}")
    return {k: np.tile(v.reshape(4, 4).T, (1, num_cols // 4))  # [bench x, bench y] -> [row, col]
            for k, v in PER_TAXEL.items()}


def linear_curve_maps(num_rows, num_cols, per_taxel=True):
    """{k, gate}, each a (rows, cols) array, for the linear/peak-normalised readout.

    k is N per mm of indentation; gate is that taxel's 3-sigma deadband in mm. Same
    bench-to-grid layout as curve_maps(). Under peak normalisation only the RATIOS of
    k across the pad survive -- the overall scale divides out -- so per_taxel=False,
    which gives every taxel the shared 3.56 N/mm, differs from per_taxel=True by at
    most 8.5% on any one taxel.
    """
    if not per_taxel:
        return {k: np.full((num_rows, num_cols), v, dtype=np.float32)
                for k, v in LINEAR_AVERAGE.items()}
    if num_rows != 4 or num_cols % 4:
        raise ValueError("per-taxel curves need 4 rows and blocks of 4 cols, "
                         f"got {num_rows}x{num_cols}")
    return {k: np.tile(v.reshape(4, 4).T, (1, num_cols // 4))
            for k, v in LINEAR_PER_TAXEL.items()}


def pool_to_taxels(samples, num_rows, num_cols, subsamples):
    """(envs, points) -> (envs, rows, cols), averaging each taxel's sample points."""
    n = subsamples
    return samples.view(samples.shape[0], num_rows, n, num_cols, n).mean(dim=(2, 4))


def depth_mm(penetration_m):
    return (penetration_m * 1e3).clamp(0.0, DEPTH_CLIP_MM)


def read_pad(penetration_m, num_rows, num_cols, subsamples, curves):
    """Sample-point depths in metres -> taxel readings, before normalization.

    Each sample point goes through its taxel's equations, then the taxel averages its
    points. Averaging depth first would be wrong because the curves are nonlinear.

    Args:
        curves: {a, b, c, e} tensors of shape (rows, cols).
    Returns:
        (drop, force_N), both (envs, rows, cols). Pass drop through normalize() to get
        what the policy sees.
    """
    n = subsamples
    d = depth_mm(penetration_m).view(-1, num_rows, n, num_cols, n)
    per_point = lambda p: p.view(num_rows, 1, num_cols, 1)
    drop = curves["c"] * (d ** per_point(curves["e"])).mean(dim=(2, 4))
    force = curves["a"] * (d ** per_point(curves["b"])).mean(dim=(2, 4))
    return drop, force


def read_pad_linear(penetration_m, num_rows, num_cols, subsamples, curves):
    """Sample-point depths in metres -> per-taxel force in newtons, before normalising.

    fn = k * d, straight line, no exponent and no offset: the bench z0 belongs to the
    stage, not the gel (see the module docstring). Depth is clipped at DEPTH_CLIP_MM
    exactly as the curve readout clips it, so the two responses saturate together.

    Args:
        curves: {k} tensor of shape (rows, cols), N per mm.
    Returns:
        (fn, depth_mm), both (envs, rows, cols). fn is newtons.
    """
    n = subsamples
    d = depth_mm(penetration_m).view(-1, num_rows, n, num_cols, n)
    d_tax = d.mean(dim=(2, 4))
    return curves["k"] * d_tax, d_tax


def normalize_peak(fn, floor_force, floor_scale, depth_mm_tax=None, curves=None):
    """VT-Refine's frame-wise peak normalisation, one pad at a time.

    s = fn / max(fn over this pad's taxels), per environment per frame. A uniform
    error in the gel stiffness therefore divides straight back out -- which is the
    point: it buys immunity to the sim pad being softer than the real one, at the cost
    of all absolute magnitude. Follows easysim-envs aloha.py:337-341, including its
    floor branch: when the loudest taxel is below `floor_force` the frame is treated as
    untouched and divided by the fixed `floor_scale` instead, so sensor noise in an
    empty frame cannot be normalised up to 1.0.

    Args:
        fn: (envs, rows, cols) newtons from read_pad_linear.
        floor_force: peak below which the frame counts as untouched.
        floor_scale: the fixed divisor used for those frames.
        depth_mm_tax, curves: if both given, taxels shallower than their own 3-sigma
            deadband (curves['gate'], mm) are zeroed first, as the hardware does.
    Returns:
        (envs, rows, cols) in [0, 1].
    """
    f = fn.clamp(min=0.0)
    if depth_mm_tax is not None and curves is not None and "gate" in curves:
        f = torch.where(depth_mm_tax < curves["gate"], torch.zeros_like(f), f)
    peak = f.flatten(1).max(dim=1).values
    denom = torch.where(peak < floor_force,
                        torch.full_like(peak, float(floor_scale)), peak)
    return (f / denom.view(-1, *([1] * (f.dim() - 1)))).clamp(0.0, 1.0)


def normalize(drop, curves):
    """Fractional drop -> the driver's s in [0, 1], the policy observation.

    Identical to LuxTac.normalize() in luxtac.py, rewritten in drop units: the counts
    (b - x) and R both scale with the live baseline, so their ratio is all the sim
    needs. Applied per TAXEL, after read_pad has pooled the sample points, because the
    hardware only ever sees one number per taxel -- in particular the 3*sigma gate is
    on the taxel's own reading, not on any sub-sample.

    Args:
        drop: (envs, rows, cols) fractional drop from read_pad.
        curves: {r, n} tensors of shape (rows, cols).
    """
    s = (drop / curves["r"]).clamp(0.0, 1.0)
    return torch.where(drop < curves["n"], torch.zeros_like(s), s)


def drop_to_raw(drop, s0):
    """Inverse of the raw observation: what the photodiode would report."""
    return s0 * (1.0 - drop)
