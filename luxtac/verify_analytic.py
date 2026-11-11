"""
Verify the LuxTac force law against closed-form expectations.

Runs standalone -- no IsaacGym, no GPU, no sim. Exercises the pure penalty law and the
analytic lattice geometry, which is where frame/axis/units mistakes actually hide.

    python -m luxtac.verify_analytic
"""

import sys

import numpy as np
import torch

from .luxtac_cfg import (
    LuxTacSensorCfg, UPSTREAM_CFG_ALOHA, UPSTREAM_CFG_DEFAULT)
from . import luxtac_viz as viz
from . import photodiode as pd


def penalty(pen, cfg):
    """Reference implementation, independent of the sensor class."""
    fn = np.clip(cfg.stiffness * pen, 0.0, cfg.max_force)
    return fn, (fn / cfg.max_force if cfg.normalize_forces else fn)


class _Checks:
    def __init__(self):
        self.failed = 0
        self.passed = 0

    def close(self, name, got, want, tol=1e-6):
        ok = np.allclose(got, want, atol=tol)
        self.passed += ok
        self.failed += (not ok)
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name}")
        if not ok:
            print(f"         got  {np.round(np.asarray(got), 6)}")
            print(f"         want {np.round(np.asarray(want), 6)}")

    def eq(self, name, got, want):
        ok = got == want
        self.passed += ok
        self.failed += (not ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + ("" if ok else f"  got {got}, want {want}"))


def linear_peak_checks(c):
    """response="linear_peak": fn = k*d, then VT-Refine's frame-wise peak norm."""
    cfg = LuxTacSensorCfg(num_rows=4, num_cols=8, point_distance=0.0045,
                            response="linear_peak", normal_offset=0.0)
    print("\n" + cfg.describe())
    cm = pd.linear_curve_maps(cfg.num_rows, cfg.num_cols)
    cur = {k: torch.tensor(v) for k, v in cm.items()}

    print("\nlinear force law")
    c.eq("k is tiled from 16 bench taxels", cm["k"].shape, (4, 8))
    c.close("block 2 copies block 1", cm["k"][:, 4:], cm["k"][:, :4])
    pen = torch.full((1, cfg.num_points), 1e-3)                 # 1 mm everywhere
    fn, d = pd.read_pad_linear(pen, cfg.num_rows, cfg.num_cols, cfg.subsamples, cur)
    c.close("fn = k * d, 1 mm -> k newtons", fn[0].numpy(), cm["k"], tol=1e-5)
    c.close("depth passes through in mm", d[0].numpy(), np.ones((4, 8)), tol=1e-6)
    pen2 = torch.full((1, cfg.num_points), 5e-3)                # past the 1.9 mm clip
    fn2, _ = pd.read_pad_linear(pen2, cfg.num_rows, cfg.num_cols, cfg.subsamples, cur)
    c.close("depth is clipped at 1.9 mm, as the curve readout is",
            fn2[0].numpy(), cm["k"] * pd.DEPTH_CLIP_MM, tol=1e-4)

    print("\npeak normalisation (VT-Refine aloha.py:337-341)")
    pen = torch.zeros(1, 4, 8)
    pen[0, 1, 2], pen[0, 2, 3] = 1.0e-3, 0.5e-3
    fn, d = pd.read_pad_linear(pen.flatten(1), 4, 8, 1, cur)
    s_ = pd.normalize_peak(fn, cfg.peak_floor_force, cfg.peak_floor_scale, d, cur)
    c.close("the deepest taxel reads exactly 1", float(s_[0].max()), 1.0)
    c.eq("untouched taxels stay 0", int((s_[0] == 0).sum()), 30)
    ratio = (cm["k"][2, 3] * 0.5) / (cm["k"][1, 2] * 1.0)
    c.close("half the depth reads the depth ratio, scaled by its own k",
            float(s_[0, 2, 3]), float(ratio), tol=1e-5)

    print("\nthe gain divides out -- the whole point of this response")
    for scale in (0.25, 4.0):
        s_scaled = pd.normalize_peak(fn * scale, cfg.peak_floor_force,
                                     cfg.peak_floor_scale, d, cur)
        c.close(f"a {scale}x stiffness error changes nothing",
                s_scaled.numpy(), s_.numpy(), tol=1e-6)

    print("\nfloor branch: an untouched frame must not normalise noise to 1")
    quiet = torch.zeros(1, 4, 8)
    quiet[0, 0, 0] = 1e-5                                        # 10 nN, far below floor
    dq = torch.full((1, 4, 8), 1e-4)                             # 0.1 um, under the gate
    sq = pd.normalize_peak(quiet, cfg.peak_floor_force, cfg.peak_floor_scale, dq, cur)
    c.close("sub-floor frame stays near zero, not 1", float(sq.max()), 0.0, tol=1e-6)
    loud = torch.zeros(1, 4, 8)
    loud[0, 0, 0] = cfg.peak_floor_force * 1.01
    dl = torch.full((1, 4, 8), 1.0)
    sl = pd.normalize_peak(loud, cfg.peak_floor_force, cfg.peak_floor_scale, dl, cur)
    c.close("just over the floor normalises by the frame max, reaching 1",
            float(sl.max()), 1.0)
    c.eq("s stays in [0, 1]", bool(((s_ >= 0) & (s_ <= 1)).all()), True)

    print("\n  a feather touch must NOT read as a full press")
    k_avg = pd.LINEAR_AVERAGE["k"]          # not a literal: a refit must move these
    for d_mm, want in ((0.03, 0.03 * k_avg / cfg.peak_floor_scale),
                       (0.06, 0.06 * k_avg / cfg.peak_floor_scale)):
        pf = torch.zeros(1, 4, 8)
        pf[0, 1, 1] = k_avg * d_mm
        df = torch.zeros(1, 4, 8)
        df[0, 1, 1] = d_mm
        sf = pd.normalize_peak(pf, cfg.peak_floor_force, cfg.peak_floor_scale, df, cur)
        c.close(f"{d_mm:.2f} mm on one taxel reads {want:.3f}, not 1.0",
                float(sf.max()), want, tol=1e-4)
    c.eq("the floor threshold sits well above the 3-sigma gate",
         bool(cfg.peak_floor_force / k_avg > 5 * float(cm["gate"].max())), True)
    c.eq("and well below a 0.164 mm frame peak, so a light hold does not sit on "
         "the branch",
         bool(cfg.peak_floor_force / k_avg < 0.164 / 1.3), True)

    print("\ngate: shallower than 3 sigma reads exactly 0, as the hardware does")
    shallow = torch.full((1, 4, 8), 1e-6)                        # 1 nm
    dgate = torch.tensor(cm["gate"]).unsqueeze(0) * 0.9
    sg = pd.normalize_peak(shallow, cfg.peak_floor_force, cfg.peak_floor_scale,
                           dgate, cur)
    c.eq("every sub-gate taxel is zeroed", int((sg == 0).sum()), 32)

    print("\npeak_scope='frame': one divisor across both pads, so the left/right")
    print("  force ratio survives. Per-pad normalisation deletes it by construction.")
    c.eq("frame is the default scope", cfg.peak_scope, "frame")
    for ratio in (0.833, 0.50, 0.95):
        dl = torch.zeros(1, 4, 8); dr = torch.zeros(1, 4, 8)
        dr[0, 1, 1], dr[0, 2, 2] = 0.60, 0.30            # mm, the deeper pad
        dl[0, 1, 1], dl[0, 2, 2] = 0.60 * ratio, 0.30 * ratio
        fl, fr = k_avg * dl, k_avg * dr
        both = torch.stack([fl, fr], dim=1)              # (1, 2, 4, 8), as the pads stack
        dboth = torch.stack([dl, dr], dim=1)
        sf = pd.normalize_peak(both, cfg.peak_floor_force, cfg.peak_floor_scale, dboth, cur)
        got = float(sf[0, 0].max()) / float(sf[0, 1].max())
        c.close(f"frame scope: pads at a {ratio:.3f} force ratio read {ratio:.3f}",
                got, ratio, tol=1e-4)
        per_l = pd.normalize_peak(fl, cfg.peak_floor_force, cfg.peak_floor_scale, dl, cur)
        per_r = pd.normalize_peak(fr, cfg.peak_floor_force, cfg.peak_floor_scale, dr, cur)
        c.eq(f"  pad scope reports 1.000/1.000 for that same {ratio:.3f}",
             (round(float(per_l.max()), 3), round(float(per_r.max()), 3)), (1.0, 1.0))
        # the within-pad pattern must be identical under either scope
        c.close("  the pattern inside a pad is unchanged by the scope",
                (sf[0, 1] / sf[0, 1].max()).numpy(), (per_r[0] / per_r[0].max()).numpy(),
                tol=1e-5)


def photodiode_checks(c):
    cfg = LuxTacSensorCfg(num_rows=4, num_cols=8, point_distance=0.0045, subsamples=5,
                            response="photodiode", normal_offset=0.0)
    print("\n" + cfg.describe())

    print("\nphotodiode lattice")
    c.eq("800 sample points per pad", cfg.num_points, 800)
    rows = pd.sample_axis(4, 5, 0.0045)
    cols = pd.sample_axis(8, 5, 0.0045)
    c.close("samples are 0.9 mm apart", np.diff(cols), 0.0009)
    c.close("samples stay inside the 18 x 36 mm pad",
            [rows.max(), cols.max()], [0.009 - 0.00045, 0.018 - 0.00045])
    c.close("one sample per taxel lands on the 4.5 mm centres",
            pd.sample_axis(4, 1, 0.0045), [-0.00675, -0.00225, 0.00225, 0.00675])
    c.close("each taxel's 5 samples average to its centre",
            rows.reshape(4, 5).mean(axis=1), pd.sample_axis(4, 1, 0.0045))

    print("\nphotodiode curves, S01")
    m = pd.curve_maps(4, 8)
    d = torch.tensor([0.0, 1.0, 1.5])
    # Literals, not m[...] read back at itself: these are transcribed from the
    # 15-pass fit so the test fails if the calibration file is swapped or edited.
    c.close("drop = 0.10501 d^1.2969, from the bench fit",
            (m["c"][0, 0] * d ** float(m["e"][0, 0])).numpy(),
            [0.0, 0.10501, 0.10501 * 1.5 ** 1.2969], tol=1e-3)
    c.close("F = 2.4211 d^1.4852, from the bench fit",
            (m["a"][0, 0] * d ** float(m["b"][0, 0])).numpy(),
            [0.0, 2.4211, 2.4211 * 1.5 ** 1.4852], tol=2e-2)
    # Not widened to the rig's 2.5 mm setting: no press of 266 reached past 2.4 mm,
    # because they all stop on the 6 N force limit first.
    c.close("depth is clipped at 1.9 mm", pd.depth_mm(torch.tensor([0.005])).numpy(), [1.9])
    c.close("negative depth reads zero", pd.depth_mm(torch.tensor([-0.001])).numpy(), [0.0])

    print("\nsecond block copies the first")
    c.eq("curve map shape", m["c"].shape, (4, 8))
    for k in "abce":
        c.close(f"block 2 == block 1 for {k}", m[k][:, 4:], m[k][:, :4])
    c.close("S01 at row 0 col 0", m["c"][0, 0], pd.PER_TAXEL["c"][0])
    c.close("S02 is the next row, S05 the next col",
            [m["c"][1, 0], m["c"][0, 1]], [pd.PER_TAXEL["c"][1], pd.PER_TAXEL["c"][4]])
    c.close("average map is uniform", pd.curve_maps(4, 8, per_taxel=False)["c"],
            pd.AVERAGE["c"])

    print("\npooling")
    T = {k: torch.as_tensor(v) for k, v in m.items()}
    drop, force = pd.read_pad(torch.full((1, 800), 0.001), 4, 8, 5, T)
    c.close("flat 1 mm press reads each taxel's own c", drop[0].numpy(), m["c"], tol=1e-5)
    c.close("flat 1 mm press reads each taxel's own a", force[0].numpy(), m["a"], tol=1e-4)
    c.eq("taxels differ from each other", bool(np.ptp(m["c"]) > 0.02), True)

    # Fill one row of sample points inside taxel (2, 3): 1/5 of that taxel is covered.
    grid = torch.zeros(1, 20, 40)
    grid[0, 12, 15:20] = 0.001
    drop, _ = pd.read_pad(grid.view(1, -1), 4, 8, 5, T)
    want = np.zeros((4, 8))
    want[2, 3] = m["c"][2, 3] / 5
    c.close("a strip over 1/5 of a taxel reads 1/5, in that taxel only",
            drop[0].numpy(), want, tol=1e-5)

    # Nonlinear curve: half the taxel at 1 mm is not the whole taxel at 0.5 mm.
    half = torch.zeros(1, 20, 40)
    half[0, :, ::2] = 0.001
    drop_half, _ = pd.read_pad(half.view(1, -1), 4, 8, 5, T)
    drop_even, _ = pd.read_pad(torch.full((1, 800), 0.0005), 4, 8, 5, T)
    c.eq("curve is applied before averaging, not after",
         bool(abs(drop_half[0, 0, 0] - drop_even[0, 0, 0]) > 1e-3), True)
    c.close("drop_to_raw inverts the raw observation",
            pd.drop_to_raw(0.1, 1000.0), 900.0)

    print("\nnormalization, the LuxTac driver's rule")
    # Literals transcribed from calibration/luxtac_calib_20261103.csv, S01:
    # R/b = 150.347/675.29, gate = 3*0.5618/675.29.
    c.close("S01 R/b = 0.222641, from luxtac_calib.csv", m["r"][0, 0], 0.222641, tol=1e-6)
    c.close("S01 gate = 0.00249582 in drop units", m["n"][0, 0], 0.00249582, tol=1e-8)
    c.close("the uniform map uses the driver's calib=None fallback of 0.22",
            pd.curve_maps(4, 8, per_taxel=False)["r"], np.full((4, 8), 0.22))
    probe = torch.stack([T["r"], T["r"] * 0.5, T["n"] * 1.01, T["n"] * 0.99, T["r"] * 2.0])
    s_norm = pd.normalize(probe, T)
    ones, zeros = np.ones((4, 8)), np.zeros((4, 8))
    c.close("drop == R/b reads s = 1", s_norm[0].numpy(), ones, tol=1e-6)
    c.close("half of R/b reads s = 0.5", s_norm[1].numpy(), ones * 0.5, tol=1e-6)
    c.close("just over the gate passes through, not snapped to 0",
            s_norm[2].numpy(), (T["n"] * 1.01 / T["r"]).numpy(), tol=1e-6)
    c.close("just under the gate reads exactly 0", s_norm[3].numpy(), zeros)
    c.eq("nothing survives the gate as a tiny nonzero",
         float(s_norm[3].abs().max()), 0.0)
    c.close("twice R/b is clipped to 1", s_norm[4].numpy(), ones, tol=1e-6)
    c.eq("s stays in [0, 1] over the whole depth range", bool(
        all(0.0 <= float(x.min()) and float(x.max()) <= 1.0
            for x in (pd.normalize(pd.read_pad(torch.full((1, 800), d), 4, 8, 5, T)[0], T)
                      for d in (-0.001, 0.0, 0.0001, 0.001, 0.0019, 0.01)))), True)

    print("\nthe fit and the driver's characterisation agree")
    # Independent evidence: the power-law fit is ours, R/b is the independent CNC/ATI run.
    # Push the fit through the driver's rule and the two must meet at ~6 N.
    r_, c_, e_, a_, b_ = (pd.PER_TAXEL[k].astype(np.float64) for k in ("r", "c", "e", "a", "b"))
    d_sat = (r_ / c_) ** (1.0 / e_)
    c.close("s = 1 is reached at 1.67-1.92 mm, inside the 1.9 mm clip",
            [d_sat.min(), d_sat.max()], [1.6676, 1.9178], tol=1e-3)
    c.close("and the fitted force there is 5.46-5.92 N, vs the driver's 'about 6 N'",
            [(a_ * d_sat ** b_).min(), (a_ * d_sat ** b_).max()], [5.4589, 5.9150], tol=2e-3)
    d_gate = (pd.PER_TAXEL["n"].astype(np.float64) / c_) ** (1.0 / e_)
    c.close("the 3-sigma gate is a 43-81 um deadband",
            [d_gate.min(), d_gate.max()], [0.0434, 0.0807], tol=1e-4)
    c.eq("every taxel is saturated or within 2% of it at the clip",
         bool((c_ * pd.DEPTH_CLIP_MM ** e_ / r_ >= 0.98).all()), True)


def main():
    c = _Checks()
    cfg = UPSTREAM_CFG_ALOHA

    print(cfg.describe())
    print()

    print("saturation depth")
    c.close("aloha cfg d_sat == 2.00 mm", cfg.saturation_depth, 0.002)
    c.close("upstream default d_sat == 0.48 mm",
            UPSTREAM_CFG_DEFAULT.saturation_depth, 0.00048)

    print("\npenalty law, flat press")
    for pen_mm, want_norm, want_n in [(0.0, 0.0, 0.0),
                                      (0.6, 0.30, 3.0),
                                      (1.2, 0.60, 6.0),
                                      (2.0, 1.00, 10.0),
                                      (5.0, 1.00, 10.0)]:  # clamped
        fn, norm = penalty(pen_mm / 1000.0, cfg)
        c.close(f"pen={pen_mm:.1f}mm -> {want_norm:.2f} ({want_n:.1f} N)",
                [fn, norm], [want_n, want_norm])

    print("\nmax_force divides out (only the ratio matters)")
    a = LuxTacSensorCfg(stiffness=5000.0, max_force=10.0)
    b = LuxTacSensorCfg(stiffness=1.0, max_force=0.002)
    pen = np.linspace(0.0, 0.004, 17)
    c.close("k=5000/F=10 and k=1/F=0.002 give identical normalized output",
            penalty(pen, a)[1], penalty(pen, b)[1])

    print("\ntilted contact ramp (3 deg across a 32-col pad)")
    cols = (np.arange(cfg.num_cols) - (cfg.num_cols - 1) / 2.0) * cfg.point_distance
    sdf = cols * np.tan(np.deg2rad(3.0))          # signed distance along the tilt
    pen_t = np.clip(-sdf, 0.0, None)
    _, norm_t = penalty(pen_t, cfg)
    c.eq("about half the row is in contact", int((pen_t > 0).sum()), 16)
    c.eq("no taxel saturates at 3 deg grazing", int((norm_t >= 1.0).sum()), 0)
    # Contact is on the negative-column side, where penetration is deepest, so the
    # ramp falls off as the column index rises.
    contacting = norm_t[norm_t > 0]
    c.eq("strictly decreasing across the contacting span",
         bool(np.all(np.diff(contacting) < 0)), True)
    c.eq("deepest taxel is at column 0", int(np.argmax(norm_t)), 0)
    c.close("deepest taxel matches tan(3deg) * 31 mm",
            contacting.max(), 0.031 * np.tan(np.deg2rad(3.0)) / cfg.saturation_depth)

    print("\nlattice geometry")
    rows = (np.arange(cfg.num_rows) - (cfg.num_rows - 1) / 2.0) * cfg.point_distance
    c.close("row extent == 22 mm (12 taxels, 11 gaps @ 2mm)",
            rows.max() - rows.min(), 0.022)
    c.close("col extent == 62 mm (32 taxels, 31 gaps @ 2mm)",
            cols.max() - cols.min(), 0.062)
    c.eq("taxel count", cfg.num_taxels, 384)
    c.close("lattice is centred on the origin", [rows.mean(), cols.mean()], [0.0, 0.0])

    print("\ntorch path matches numpy path")
    pen_t2 = torch.linspace(0.0, 0.004, 17)
    fn_t = (cfg.stiffness * pen_t2).clamp(0.0, cfg.max_force) / cfg.max_force
    c.close("torch == numpy", fn_t.numpy(), penalty(pen_t2.numpy(), cfg)[1], tol=1e-6)

    print("\nconfig validation")
    warns = LuxTacSensorCfg(normal_offset=+0.003).validate()
    c.eq("positive normal_offset is flagged",
         any("positive" in w for w in warns), True)
    warns = LuxTacSensorCfg(normal_offset=-0.001, max_force=10.0).validate()
    c.eq("offset shallower than d_sat is flagged",
         any("far side" in w for w in warns), True)
    c.eq("aloha cfg is clean", cfg.validate(), [])

    print("\nheatmap render (tilted row broadcast over all 12 rows)")
    grid = np.tile(norm_t, (cfg.num_rows, 1))
    print(viz.grid_to_ascii(grid))
    print("  " + viz.summarize(grid, cfg))
    rgb = viz.normalized_to_rgb(grid)
    c.eq("rgb shape", rgb.shape, (cfg.num_rows, cfg.num_cols, 3))
    # Column 0 is the deepest taxel here; the no-contact taxels are at the far end.
    c.close("zero -> black", rgb[0, -1], [0, 0, 0], tol=1)
    c.close("one  -> white", viz.normalized_to_rgb(np.ones((1, 1)))[0, 0], [255, 255, 255], tol=1)
    c.close("mid  -> mid grey", viz.normalized_to_rgb(np.full((1, 1), 0.5))[0, 0],
            [128, 128, 128], tol=1)
    c.eq("grayscale channels equal", bool((rgb[..., 0] == rgb[..., 2]).all()), True)

    photodiode_checks(c)
    linear_peak_checks(c)

    print(f"\n{c.passed} passed, {c.failed} failed")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
