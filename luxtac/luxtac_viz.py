"""Display helpers for the LuxTac pad: orientation, colour lookup, ASCII, pooling."""

import numpy as np

_ASCII_RAMP = " .:-=+*#%@"


def gray_lut(n=256):
    """(n, 3) uint8 black-to-white ramp. 0 = no contact, 1 = full contact."""
    v = np.linspace(0, 255, n).round().astype(np.uint8)
    return np.stack([v, v, v], axis=-1)


def orient_for_display(grid, mirror_width=False):
    """Rotate a (rows, cols) pad grid into the orientation it physically has.

    Rows span the finger's width and cols run along its length, so the grid is
    transposed: image width = finger width, image height = finger length, with the
    column nearest the hand at the top and the one nearest the fingertip at the bottom.

    mirror_width reverses the across-finger axis. The left pad's frame is rotated pi
    about its finger so its normal faces the object, which reverses its row axis
    relative to the right pad; mirror_width=True for the left pad makes motion read the
    same way in both displays. Display only.
    """
    out = np.asarray(grid).T
    return out[:, ::-1] if mirror_width else out


def normalized_to_rgb(grid, lut=None):
    """(rows, cols) float in [0,1] -> (rows, cols, 3) uint8 RGB (grayscale by default)."""
    lut = gray_lut() if lut is None else lut
    idx = np.clip(np.asarray(grid, dtype=np.float64), 0.0, 1.0)
    idx = (idx * (len(lut) - 1)).round().astype(np.int32)
    return lut[idx]


def grid_to_ascii(grid, ramp=_ASCII_RAMP, header=True):
    """Render a [0,1] grid as ASCII. For terminals and headless logs."""
    g = np.clip(np.asarray(grid, dtype=np.float64), 0.0, 1.0)
    rows, cols = g.shape
    lines = []
    if header:
        lines.append("     " + "".join(str(c % 10) for c in range(cols)))
    for r in range(rows):
        chars = "".join(ramp[min(len(ramp) - 1, int(v * len(ramp)))] for v in g[r])
        lines.append(f"  {r:2d} {chars}")
    return "\n".join(lines)


def summarize(grid, cfg=None):
    """One-line stats for logging."""
    g = np.asarray(grid, dtype=np.float64)
    n_contact = int((g > 0).sum())
    n_sat = int((g >= 1.0 - 1e-9).sum())
    line = (f"peak={g.max():.3f} mean={g.mean():.4f} "
            f"contact={n_contact}/{g.size} saturated={n_sat}")
    if cfg is not None:
        line += f" d_sat={cfg.saturation_depth * 1e3:.2f}mm"
    return line


def pool_subsamples(depth, num_rows, num_cols):
    """(rows*n, cols*n) sample grid -> (rows, cols) per-taxel mean.

    NOTE this is the mean DEPTH, which is not what the taxel reports: the response
    curve is applied per sample and averaged after, so mean(c*d^e) != c*mean(d)^e.
    It is here to read the geometry in millimetres, not to reproduce the readout.
    """
    d = np.asarray(depth, dtype=np.float64)
    n_r, n_c = d.shape[0] // num_rows, d.shape[1] // num_cols
    return d.reshape(num_rows, n_r, num_cols, n_c).mean(axis=(1, 3))
