"""Pool every press of a multi-run photodiode bench archive into one frames.npz.

`trial` is a globally unique index per (run, trial), and a separate `run` array is
carried so validation can hold out a whole session.

Conventions:
  photo_valid_mask == 65535
  s0, F0  = mean of the first 20 frames of that press
  raw     = S<kk>_raw of the taxel being pressed
  load    = phase == "load"

usage: load_15pass.py <archive_dir> <out.npz>
"""
import sys
import glob
import os

import numpy as np
import pandas as pd

S = [f"S{i:02d}" for i in range(1, 17)]
USE = (["photo_valid_mask", "z_depth_mm", "Fz_corrected", "phase", "timestamp",
        "stop_reason", "max_depth_mm", "max_force_N"] + [s + "_raw" for s in S])


def main(root, out):
    files = sorted(f for f in glob.glob(f"{root}/run_*/trial_*/S*/taxel_data.csv")
                   if os.path.getsize(f) > 1024)
    runs = sorted({f.split(os.sep)[-4] for f in files})
    run_id = {r: i for i, r in enumerate(runs)}
    trial_id, cols, press = {}, [], []
    for f in files:
        run, tr, tax = f.split(os.sep)[-4:-1]
        x = pd.read_csv(f, low_memory=False, usecols=lambda c: c in USE)
        x = x[x.photo_valid_mask == 65535]
        if len(x) < 25:
            continue
        k = int(tax[1:]) - 1
        key = (run, tr)
        if key not in trial_id:
            trial_id[key] = len(trial_id)
        raw = x[S[k] + "_raw"].values.astype(float)
        z = x.z_depth_mm.values.astype(float)
        F = x.Fz_corrected.values.astype(float)
        cols.append(dict(
            run=np.full(len(x), run_id[run], np.int16),
            trial=np.full(len(x), trial_id[key], np.int16),
            k=np.full(len(x), k, np.int8),
            z=z, F=F, raw=raw,
            s0=np.full(len(x), raw[:20].mean()),
            F0=np.full(len(x), F[:20].mean()),
            load=(x.phase.values == "load"),
            neigh=x[[s + "_raw" for s in S]].values.astype(np.float32),
        ))
        press.append(dict(run=run, trial=tr, k=k, n=len(x), zmax=float(z.max()),
                          Fmax=float(F.max() - F[:20].mean()),
                          stop=str(x.stop_reason.dropna().iloc[-1]) if x.stop_reason.notna().any() else ""))
    D = {kk: np.concatenate([c[kk] for c in cols]) for kk in cols[0]}
    np.savez_compressed(out, runs=np.array(runs), **D)
    P = pd.DataFrame(press)

    print(f"{len(P)} presses, {len(D['z'])} frames, {len(runs)} runs, {len(trial_id)} trials")
    print(f"\ndepth reached per press (mm): "
          f"min {P.zmax.min():.2f}  median {P.zmax.median():.2f}  max {P.zmax.max():.2f}")
    print(f"peak force per press (N):   min {P.Fmax.min():.2f}  median {P.Fmax.median():.2f}  "
          f"max {P.Fmax.max():.2f}")
    print(f"presses reaching past 1.9 mm: {(P.zmax > 1.9).sum()} of {len(P)}")
    print(f"presses reaching past 2.4 mm: {(P.zmax > 2.4).sum()} of {len(P)}")
    print("\nstop_reason:")
    print(P.stop.value_counts().to_string() or "  (none recorded)")
    print("\ndepth reached, by run:")
    print(P.groupby("run").zmax.agg(["count", "min", "median", "max"]).round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
