"""One depth->force and one depth->drop equation per taxel, pooled over all runs.

Stage depth, no z0 offsets, load and unload together: 266 presses over 7 sessions.
Validation holds out a whole session. <old.json> is the single-session fit
(photodiode_run_20261101.json), printed alongside for comparison.

usage: fit_15pass.py <frames.npz> <old.json> <out.json>
"""
import sys
import json

import numpy as np
from scipy.optimize import least_squares

np.set_printoptions(precision=3, suppress=True, linewidth=200)
N = 16

D = np.load(sys.argv[1], allow_pickle=True)
k, run, trial, z, F, raw, s0, F0 = (D[n] for n in ("k", "run", "trial", "z", "F", "raw", "s0", "F0"))
runs = [str(r) for r in D["runs"]]
F = F - F0
r = 1 - raw / s0
m = z > 0.0


def power(y, mask, p0):
    x, v = z[mask], y[mask]
    if mask.sum() < 10:
        return np.array([np.nan, np.nan])
    return least_squares(lambda q: q[0] * x ** q[1] - v, p0,
                         loss="soft_l1", f_scale=(0.3 if p0[0] > 1 else 0.03)).x


def per_taxel(y, mask, p0):
    return np.array([power(y, mask & (k == i), p0) for i in range(N)])


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


FP, RP = per_taxel(F, m, (2.5, 1.4)), per_taxel(r, m, (0.1, 1.3))
Fs, Rs = power(F, m, (2.5, 1.4)), power(r, m, (0.1, 1.3))
print(f"frames {m.sum()} of {len(z)}   presses {len(np.unique(trial * 100 + k))}")
print("force a", FP[:, 0]); print("force b", FP[:, 1]); print("force average", Fs)
print("drop  c", RP[:, 0]); print("drop  e", RP[:, 1]); print("drop  average", Rs)

# ---- per-run stability: is pooling hiding a session effect? -------------------
print("\nper-run drop coefficient c, per taxel (nan = that taxel not pressed in that run)")
cr = np.full((len(runs), N), np.nan)
for ri in range(len(runs)):
    cr[ri] = per_taxel(r, m & (run == ri), (0.1, 1.3))[:, 0]
    print(f"  {runs[ri]:22s} {np.round(cr[ri], 4)}")
with np.errstate(invalid="ignore"):
    spread = (np.nanmax(cr, 0) - np.nanmin(cr, 0)) / np.nanmean(cr, 0)
print(f"  across-run spread of c, per taxel (max-min)/mean: {np.round(spread, 3)}")
print(f"  worst taxel S{int(np.nanargmax(spread)) + 1:02d} at {np.nanmax(spread) * 100:.0f}%, "
      f"median {np.nanmedian(spread) * 100:.0f}%")

# ---- held out by SESSION ------------------------------------------------------
print("\nheld-out by run, error on frames past first contact")
res = {"Fp": [], "Fs": [], "Rp": [], "Rs": []}
for ri in range(len(runs)):
    a, b = m & (run != ri), m & (run == ri) & (F > 0.2)
    if b.sum() < 200:
        continue
    fp, rp = per_taxel(F, a, (2.5, 1.4)), per_taxel(r, a, (0.1, 1.3))
    fs, rs = power(F, a, (2.5, 1.4)), power(r, a, (0.1, 1.3))
    x, kk = z[b], k[b]
    res["Fp"].append(rmse(fp[kk, 0] * x ** fp[kk, 1] - F[b]))
    res["Fs"].append(rmse(fs[0] * x ** fs[1] - F[b]))
    res["Rp"].append(rmse(rp[kk, 0] * x ** rp[kk, 1] - r[b]))
    res["Rs"].append(rmse(rs[0] * x ** rs[1] - r[b]))
    print(f"  hold out {runs[ri]:22s} force per-taxel {res['Fp'][-1]:.3f} N  "
          f"shared {res['Fs'][-1]:.3f} N   drop per-taxel {res['Rp'][-1]:.4f}  shared {res['Rs'][-1]:.4f}")
s0k = np.array([np.median(s0[k == i]) for i in range(N)])
print(f"  MEAN force N: per taxel {np.mean(res['Fp']):.3f} | average {np.mean(res['Fs']):.3f}")
print(f"  MEAN drop   : per taxel {np.mean(res['Rp']):.4f} | average {np.mean(res['Rs']):.4f}")
print(f"  raw counts  : per taxel {np.mean(res['Rp']) * np.median(s0k):.1f} | "
      f"average {np.mean(res['Rs']) * np.median(s0k):.1f}")

# ---- old vs new ---------------------------------------------------------------
old = json.load(open(sys.argv[2]))["per_taxel"]
print("\n          old (68 presses)              new (266 presses)")
print("taxel     a      b      c       e        a      b      c       e")
for i in range(N):
    print(f"  S{i+1:02d}  {old['a'][i]:6.3f} {old['b'][i]:6.3f} {old['c'][i]:7.4f} {old['e'][i]:6.3f}"
          f"   {FP[i,0]:6.3f} {FP[i,1]:6.3f} {RP[i,0]:7.4f} {RP[i,1]:6.3f}")
oa = old["average"]
print(f"  avg  {oa['a']:6.3f} {oa['b']:6.3f} {oa['c']:7.4f} {oa['e']:6.3f}"
      f"   {Fs[0]:6.3f} {Fs[1]:6.3f} {Rs[0]:7.4f} {Rs[1]:6.3f}")

json.dump(dict(
    source=f"{len(runs)} bench sessions 2026-11-01, 266 presses (15 complete 16-taxel passes "
           f"+ 4 partial trials), every frame, load and unload",
    frames=int(m.sum()), presses=266, runs=runs,
    form="F_N = a*z_mm^b and 1 - raw/S0 = c*z_mm^e, z is stage depth, no offsets",
    per_taxel=dict(
        a=[round(float(v), 4) for v in FP[:, 0]], b=[round(float(v), 4) for v in FP[:, 1]],
        c=[round(float(v), 5) for v in RP[:, 0]], e=[round(float(v), 4) for v in RP[:, 1]],
        average=dict(a=round(float(Fs[0]), 4), b=round(float(Fs[1]), 4),
                     c=round(float(Rs[0]), 5), e=round(float(Rs[1]), 4))),
    s0_counts=[float(v) for v in np.round(s0k)],
    held_out_by_run=dict(force_N_per_taxel=round(float(np.mean(res["Fp"])), 4),
                         force_N_shared=round(float(np.mean(res["Fs"])), 4),
                         drop_per_taxel=round(float(np.mean(res["Rp"])), 5),
                         drop_shared=round(float(np.mean(res["Rs"])), 5)),
    across_run_spread_of_c=[round(float(v), 4) for v in spread],
    tested_force_max_N=6.0, tested_depth_max_mm=round(float(z.max()), 2),
    note="presses end on the 6 N force limit, not the 2.5 mm depth limit: median peak "
         "depth 1.85 mm, max 2.10 mm, none past 2.4 mm. The 1.9 mm clip still covers the data.",
    second_block="next 4x4 has no bench data: block 2 copies block 1"),
    open(sys.argv[3], "w"), indent=1)
print(f"\nwrote {sys.argv[3]}")
