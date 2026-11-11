"""One LINEAR force equation per taxel, for the peak-normalised readout.

WHY LINEAR. The stiffening exponent b = 1.4785 in
photodiode_15pass_20261101.json is mostly an artifact of fit_15pass.py forcing the
stage zero onto contact. Free that offset and b falls to 1.17; at equal parameter
count F = k (z - z0)+ and a z^b are a tie on 1.06 M frames. So fit the straight line.

WHY PER TAXEL ANYWAY. The force gain is near-uniform (k spans 3.45-3.75 N/mm, +-2.3%),
so per-taxel k is nearly free and nearly invisible. It is fitted because it costs one
line, not because it is expected to show. The OPTICAL gain is the one that varies
(81% across the pad) and that channel keeps its own calibration.

WHAT z0 IS, AND WHY SIM DOES NOT GET ONE. z0 (0.15-0.34 mm) is the bench stage zeroing
above the gel surface -- an instrument offset, not a property of the gel. Sim depth is
an SDF distance from the actual gel surface, so sim contact is at d = 0. Only the
SLOPE transfers. Fitting through the origin instead would give 2.93 N/mm, a 19%
underestimate of the real stiffness, which is the mistake this file exists to avoid.

GATE. The hardware zeroes a taxel below 3 sigma of its own baseline noise. Converted
to depth through that taxel's optical slope, that is the deadband in mm recorded here,
so the peak-normalised readout can reproduce it.

usage: fit_linear.py <frames.npz> <out.json>
"""
import sys
import json

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from luxtac.photodiode import PER_TAXEL          # noqa: E402  (R and 3-sigma gate)

N = 16
D = np.load(sys.argv[1], allow_pickle=True)
k, z, F, F0, raw, s0, load = (D[n] for n in ("k", "z", "F", "F0", "raw", "s0", "load"))
F = F - F0
drop = 1.0 - raw / s0
sel = load & (z > 0) & (F < 5.5)          # presses end on a 6 N limit; drop the hold


def line(x, y, p0):
    """F = q0 * (z - q1)+ , robust."""
    return least_squares(lambda q: q[0] * np.clip(x - q[1], 0, None) - y, p0,
                         loss="soft_l1", f_scale=(0.3 if p0[0] > 1 else 0.03)).x


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


kf, z0f, kd, z0d, res = [], [], [], [], []
for i in range(N):
    s = sel & (k == i)
    a, b = line(z[s], F[s], (3.5, 0.2))
    c, d = line(z[s], drop[s], (0.13, 0.2))
    kf.append(a); z0f.append(b); kd.append(c); z0d.append(d)
    res.append(rmse(a * np.clip(z[s] - b, 0, None) - F[s]))
kf, z0f, kd, z0d = map(np.asarray, (kf, z0f, kd, z0d))

gate_mm = PER_TAXEL["n"] / kd              # 3-sigma deadband, in mm of indentation
sat_mm = PER_TAXEL["r"] / kd               # depth at which the real driver's s hits 1

kf_s, z0f_s = line(z[sel], F[sel], (3.5, 0.2))
print(f"frames {sel.sum()} of {len(z)}")
print(f"{'taxel':>6} {'k N/mm':>8} {'z0 mm':>7} {'rmse N':>7} {'gate um':>8} {'s=1 at mm':>10}")
for i in range(N):
    print(f"  S{i+1:02d} {kf[i]:8.3f} {z0f[i]:7.3f} {res[i]:7.4f} "
          f"{gate_mm[i]*1e3:8.1f} {sat_mm[i]:10.2f}")
print(f"\n  pooled      k {kf_s:.4f} N/mm   z0 {z0f_s:.4f} mm")
print(f"  per-taxel k mean {kf.mean():.4f}  +-{100*kf.std()/kf.mean():.1f}%  "
      f"max/min {kf.max()/kf.min():.3f}")
print(f"  -> after peak normalisation, worst-case taxel error from using one shared k "
      f"instead of 16: {100*(kf.max()/kf.min()-1):.1f}%")
print(f"  gate  mean {gate_mm.mean()*1e3:.1f} um   s=1 at mean {sat_mm.mean():.2f} mm")

json.dump(dict(
    source="266 presses, 7 bench sessions 2026-11-01, load phase below the 6 N stop",
    frames=int(sel.sum()),
    form=("F_N = k * (z_mm - z0_mm)+ . z0 is the BENCH STAGE offset above the gel "
          "surface and does NOT transfer to sim, where depth is an SDF distance from "
          "the surface itself. Use k alone; see the module docstring."),
    per_taxel=dict(
        k_N_per_mm=[round(float(v), 4) for v in kf],
        z0_mm=[round(float(v), 4) for v in z0f],
        gate_depth_mm=[round(float(v), 5) for v in gate_mm],
        saturation_depth_mm=[round(float(v), 4) for v in sat_mm],
        rmse_N=[round(float(v), 4) for v in res]),
    shared=dict(k_N_per_mm=round(float(kf_s), 4), z0_mm=round(float(z0f_s), 4),
                gate_depth_mm=round(float(gate_mm.mean()), 5),
                saturation_depth_mm=round(float(sat_mm.mean()), 4)),
    k_spread_max_over_min=round(float(kf.max() / kf.min()), 4),
    note=("the optical slope used for the gate and saturation depths is fitted on the "
          "same frames; corr(R_i, optical k_i) = +0.963 against the independent "
          "CNC/ATI calibration"),
    second_block="next 4x4 has no bench data: block 2 copies block 1"),
    open(sys.argv[2], "w"), indent=1)
print(f"\nwrote {sys.argv[2]}")
