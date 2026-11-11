"""Compare the real pad with the simulated pad and print the numbers.

    python alignment/compare.py [sim_hold_dir]      default: alignment/sim/hold

Real: alignment/real/grasps.npz, 12 grasps x 50 frames x 16 taxels, as recorded.
Sim:  one npz per grasp from scripts/run_holds.sh, 50 frames each.

Every frame is divided by its largest taxel (or by 0.36 if that is below 0.02). The
score of two frames is the cosine between their 16 values. A grasp's score is the mean
over all 50 x 50 pairs of a real and a simulated frame.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
SIM = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "sim" / "hold"
TAU, S_MAX = 0.02, 0.36
SIM_TAG = {"vert": "vert", "horz": "horz", "45deg": "deg45"}


def peak_normalise(frame):
    g = np.clip(np.asarray(frame, dtype=float), 0.0, None)
    peak = g.max()
    return g / (peak if peak >= TAU else S_MAX)


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def sim_frames(name):
    """(50, 16) simulated frames of one grasp, in the real pad's taxel order."""
    hold = np.load(SIM / f"{name}.npz")[f"lux_{name}__hold_s"]
    return np.stack([peak_normalise(f.T).ravel() for f in hold])


real = np.load(HERE / "real" / "grasps.npz")
obj = [str(x) for x in real["label"]]
ori = [str(x) for x in real["orient"]]
n = len(obj)

R = np.stack([[peak_normalise(f) for f in real["grid"][i]] for i in range(n)])       # (12, 50, 16)
S = np.stack([sim_frames(f"{obj[i]}_{SIM_TAG[ori[i]]}") for i in range(n)])
Ru, Su = unit(R), unit(S)

# M[i, j]: mean cosine between the real frames of grasp i and the sim frames of grasp j
M = np.einsum("iak,jbk->ij", Ru, Su) / (R.shape[1] * S.shape[1])
matched = np.diag(M)

# upper reference: two different real frames of the same hold
off = ~np.eye(R.shape[1], dtype=bool)
ceiling = np.mean([(Ru[i] @ Ru[i].T)[off].mean() for i in range(n)])

# lower references
flat = unit(np.ones(16))
uniform = (Ru @ flat).mean()
other_object = np.mean([M[i, j] for i in range(n) for j in range(n) if obj[i] != obj[j]])
other_orientation = np.mean([M[i, j] for i in range(n) for j in range(n)
                             if obj[i] == obj[j] and ori[i] != ori[j]])
rng = np.random.default_rng(0)
shuffled = np.mean([(Ru[i] @ unit(S[i][:, rng.permutation(16)]).T).mean()
                    for i in range(n) for _ in range(20)])

# 45 degree grasps against the simulation along the opposite diagonal
diag = [i for i in range(n) if ori[i] == "45deg" and obj[i] != "cuboid"]
mirrored = np.mean([(Ru[i] @ unit(sim_frames(f"{obj[i]}_deg135")).T).mean() for i in diag])

# one-to-one assignment of real grasps to simulated grasps
rows, cols = linear_sum_assignment(-M)
recovered = sum(obj[i] == obj[j] for i, j in zip(rows, cols))

# distribution of normalised values, pooled over all taxels and frames
q = np.linspace(0, 1, 401)
w1 = np.abs(np.quantile(R.ravel(), q) - np.quantile(S.ravel(), q)).mean()

print(f"matched simulation                    {matched.mean():.3f}   "
      f"(median {np.median(matched):.3f}, lowest grasp {matched.min():.3f})")
print(f"real frames of the same hold          {ceiling:.4f}")
print(f"uniform pad                           {uniform:.3f}")
print(f"simulation of a different object      {other_object:.3f}")
print(f"same object, different orientation    {other_orientation:.3f}")
print(f"matched simulation, taxels shuffled   {shuffled:.3f}")
print(f"45 deg, correct / mirrored diagonal   {matched[diag].mean():.3f} / {mirrored:.3f}")
print(f"one-to-one assignment, object         {recovered} / {n}")
print(f"Wasserstein-1 of pooled values        {w1:.3f}")
print()
for i in range(n):
    print(f"  {obj[i]:9s}{ori[i]:6s} {matched[i]:.3f}")
