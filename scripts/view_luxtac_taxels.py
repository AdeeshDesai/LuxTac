#!/usr/bin/env python
"""Live per-taxel readout of one LuxTac alignment object, driven from the keyboard.

The object comes from the asset table luxtac_asset_info_objects.yaml through hydra
overrides.

Shown per taxel, for the scored pad (left unless --pad right):

    SDF depth mm   the raw geometric penetration of the object into the gel surface,
                   pooled from the sample lattice to the 4x4 taxels. Clipped by the
                   response at 1.9 mm.
    s (absolute)   the driver's observation, photodiode.normalize(): each sub-sample
                   goes through its own bench curve drop = c*d^e, the taxel averages
                   its points, then s = clip(drop/R, 0, 1) with a 3-sigma gate. It
                   saturates at 1.67-1.92 mm.
    peak-norm      s / max(s) over the pad this frame, floored at SMAX when the pad is
                   untouched. The alignment numbers are computed on this.
    REAL           the matching held grasp from the 12-episode LuxTac dataset, mean of
                   its 50-frame hold, peak-normalised the same way.

and the live cosine between the last two.

FRAME. The sim grid is [row = across the finger, col = along it]; the real driver is
[row = along, col = across]. Every panel is printed in the real frame (the sim grids
are transposed, as in the analysis scripts): S01 top-left, top row = robot side,
bottom row = floor side.

ARM. fingertip_midpoint_rot_noise and pos_noise are zeroed, so the gripper is level
and perpendicular to the ground, as on the bench.

KEYS (terminal must have focus; the IsaacGym viewer window is just for looking)

    c / o    close / open the gripper by 0.10 mm of DOF
    C / O    close / open by 0.50 mm
    down/up  same as c / o
    r        reset to the starting width
    p        write the current frame to --snap as an npz
    q        quit

One object per process, and one IsaacGym per GPU.

    ./scripts/run_view_luxtac.sh --object peg_vert
    ./scripts/run_view_luxtac.sh --object peg_vert --once --indent 4.5   # no keyboard
"""
import argparse
import os
import select
import sys
import termios
import tty
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_CONFIG = str(_REPO / "luxtac" / "config")

SHAPES = ("cuboid", "cylinder", "ellipse", "peg")
ORIENTS = ("vert", "horz", "deg45", "deg135", "insert")   # insert: 90mm peg for teleop, no real episode
# extent along the grip axis, mm
THICK = {"cuboid": 20.0, "cylinder": 25.0, "ellipse": 20.0, "peg": 15.0}

# The analysis scripts' peak normalisation, copied so this display cannot drift from it.
TAU, SMAX = 0.02, 0.36

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--object", default="peg_vert",
                help="shape_orient, e.g. peg_vert / cuboid_horz / cylinder_deg45")
ap.add_argument("--indent", type=float, default=4.5,
                help="starting target indentation, mm. The scored sweep used "
                     "1.5..4.5; 4.5 lands near the global operating point.")
ap.add_argument("--proud-mm", type=float, default=1.0,
                help="how far the gel stands proud of the blade (gel45 URDF: 1.0)")
ap.add_argument("--urdf", default="luxtac_franka_gel45.urdf")
ap.add_argument("--k", type=float, default=3100.0, help="gel compliance_stiffness, N/m")
ap.add_argument("--pad", default="left", choices=["left", "right"],
                help="which pad fills the four panels. The alignment study scored "
                     "the LEFT pad; the right one is summarised on one line either way.")
ap.add_argument("--ref", default=str(_REPO / "alignment" / "real" / "grasps.npz"),
                help="the 12 real holds. Omit the panel by pointing at nothing.")
ap.add_argument("--ep", type=int, default=-1,
                help="force a real episode index 0-11 instead of matching on name")
ap.add_argument("--substep", type=int, default=6,
                help="sim steps between redraws")
ap.add_argument("--settle", type=int, default=60,
                help="steps to settle before the first frame (and in --once)")
ap.add_argument("--once", action="store_true",
                help="settle, print one frame, exit. No keyboard. Use this to check "
                     "the viewer reproduces a scored run.")
ap.add_argument("--sweep", type=float, nargs="*", default=None,
                help="walk these target indentations in order, recording BOTH pads at "
                     "each, then exit. No keyboard. Defaults to the scored sweep "
                     "1.5..4.5 when given with no values. Stepping through the list in "
                     "order reproduces align_luxtac_objects.py's loading path, so the "
                     "left-pad output should match alignment/sim/sweep/<object>.npz exactly.")
ap.add_argument("--trace", type=int, default=0,
                help="hold the starting width and record the finger DOF and pad depth "
                     "for this many steps, then report the ringing. The finger drive "
                     "is P=400 D=2 (task.ctrl.all), i.e. damping ratio ~0.05 against "
                     "a 3100 N/m gel, so it oscillates. Pass "
                     "task.ctrl.all.gripper_deriv_gains='[20,20]' to compare.")
ap.add_argument("--headless", action="store_true", help="no IsaacGym viewer window")
ap.add_argument("--no-colour", action="store_true")
ap.add_argument("--snap", default="/tmp/luxtac_frame.npz")
# Hydra also parses sys.argv, so take only our flags and hand the rest on.
_A, _REST = ap.parse_known_args()
sys.argv = [sys.argv[0]] + _REST


def parse_object(name):
    for s in SHAPES:
        for o in ORIENTS:
            if name == f"{s}_{o}":
                return s, o
    raise SystemExit(f"unknown --object {name!r}; expected <shape>_<orient> with "
                     f"shape in {SHAPES} and orient in {ORIENTS}")


# --------------------------------------------------------------------- rendering

def peaknorm(g):
    """pn() of the analysis scripts: divide by the pad peak, or by SMAX if nothing is touching."""
    x = np.clip(np.asarray(g, dtype=float), 0.0, None)
    pk = x.max()
    return x / (pk if pk >= TAU else SMAX)


def _bg(v):
    """White -> dark blue, the ramp the pad viewer uses. v in [0, 1]."""
    v = float(np.clip(v, 0.0, 1.0))
    r = int(round(255 + (10 - 255) * v))
    g = int(round(255 + (35 - 255) * v))
    b = int(round(255 + (110 - 255) * v))
    fg = "0;0;0" if v < 0.55 else "255;255;255"
    return f"\x1b[48;2;{r};{g};{b}m\x1b[38;2;{fg}m"


def panel(title, g, scale, colour=True, note=""):
    """A 4x4 grid as a list of lines, shaded by value/scale."""
    g = np.asarray(g, dtype=float)
    w = g.shape[1] * 7
    head = title[:w].ljust(w)
    lines = [f"\x1b[1m{head}\x1b[0m" if colour else head]
    for r in range(g.shape[0]):
        cells = []
        for c in range(g.shape[1]):
            v = g[r, c]
            txt = f" {v:5.3f} " if abs(v) < 10 else f"{v:6.2f} "
            cells.append(_bg(v / scale) + txt + "\x1b[0m" if colour else txt)
        lines.append("".join(cells))
    tail = note[:w].ljust(w)
    lines.append(f"\x1b[2m{tail}\x1b[0m" if colour else tail)
    return lines


def side_by_side(panels, gap=3, cell_w=4 * 7):
    """Lay panels out in one row if the terminal is wide enough, else 2 per row.

    len() cannot measure these lines -- they carry ANSI escapes -- so the width comes
    from the grid shape instead.
    """
    pw = max(len(p) for p in panels)            # rows per panel (all equal)
    try:
        width = os.get_terminal_size().columns if sys.stdout.isatty() else 200
    except OSError:
        width = 200
    per_row = len(panels) if width >= len(panels) * (cell_w + gap) else 2
    out = []
    for i in range(0, len(panels), per_row):
        chunk = panels[i:i + per_row]
        for r in range(pw):
            out.append((" " * gap).join(p[r] for p in chunk))
        out.append("")
    return out


# --------------------------------------------------------------------- keyboard

class Keys:
    """Non-blocking single keypresses, with the terminal put back on the way out."""

    def __init__(self, enabled=True):
        self.enabled = enabled and sys.stdin.isatty()
        self.fd = sys.stdin.fileno() if self.enabled else None
        self.saved = None

    def __enter__(self):
        if self.enabled:
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
        return False

    def poll(self):
        """Every key waiting right now, arrow escapes folded to 'o'/'c'.

        os.read, NOT sys.stdin.read: an arrow arrives as the three bytes ESC [ A in a
        single write, and TextIOWrapper drains the whole fd into its own buffer on the
        first read(1). select() then reports nothing pending and the '[A' is lost, so
        every arrow press is silently swallowed while plain letters still work.
        """
        if not self.enabled:
            return []
        buf = ""
        while select.select([sys.stdin], [], [], 0)[0]:
            chunk = os.read(self.fd, 64)
            if not chunk:
                break
            buf += chunk.decode(errors="ignore")
        out, i = [], 0
        while i < len(buf):
            if buf[i] == "\x1b" and buf[i + 1:i + 2] == "[":
                out.append({"A": "o", "B": "c"}.get(buf[i + 2:i + 3], ""))
                i += 3
            else:
                out.append(buf[i])
                i += 1
        return [c for c in out if c]


# --------------------------------------------------------------------- real data

def load_reference(path, shape, orient, forced):
    """(4x4 peak-normalised real grid, caption) or (None, why not)."""
    p = Path(path)
    if not p.is_file():
        return None, f"no reference at {p}"
    d = np.load(p, allow_pickle=True)
    lab = [str(x) for x in d["label"]]
    ori = [str(x) for x in d["orient"]]
    if forced >= 0:
        i = forced
    else:
        tag = {"vert": "vert", "horz": "horz", "deg45": "45deg"}.get(orient)
        hits = [j for j in range(12) if lab[j] == shape and ori[j] == tag]
        if not hits:
            return None, f"no real episode for {shape} {orient}"
        i = hits[0]
    g = np.asarray(d["grid"][i], dtype=float).mean(axis=0).reshape(4, 4)
    return peaknorm(g), f"ep{i}  {lab[i]} {ori[i]}  mean of 50 held frames"


# --------------------------------------------------------------------- main

def main():
    import hydra
    from omegaconf import DictConfig
    from isaacgym import gymapi
    from luxtac.envs.env_wrapper import IsaacEnvWrapper      # isaacgym before torch
    import torch
    from luxtac import luxtac_viz

    shape, orient = parse_object(_A.object)
    sub = f"lux_{shape}_{orient}"
    T = THICK[shape]
    real, real_note = load_reference(_A.ref, shape, orient, _A.ep)
    colour = (not _A.no_colour) and sys.stdout.isatty()

    @hydra.main(version_base="1.1", config_path=_CONFIG,
                config_name="isaacgym_config_luxtac")
    def run(cfg: DictConfig):
        cfg.headless = bool(_A.headless)
        cfg.num_envs = 1
        # Level arm, as on the bench.
        cfg.task.randomize.fingertip_midpoint_rot_noise = [0.0, 0.0, 0.0]
        cfg.task.randomize.fingertip_midpoint_pos_noise = [0.0, 0.0, 0.0]

        e = cfg.task.env
        e.asset_info_filename = "luxtac_asset_info_objects.yaml"
        e.franka_urdf_file = _A.urdf
        e.luxtac.num_cols = 4
        # obsDims is declared separately from env.luxtac and the task allocates its
        # observation buffers from it, so num_cols alone throws a size mismatch out of
        # compute_observations.
        _nr, _ss = int(e.luxtac.num_rows), int(e.luxtac.subsamples)
        for _side in ("left", "right"):
            e.obsDims[f"luxtac_{_side}"] = [_nr, 4]
            e.obsDims[f"luxtac_depth_{_side}"] = [_nr * _ss, 4 * _ss]
        e.luxtac.response = "photodiode"
        kd = float(2 * np.sqrt(_A.k * 0.03))             # contact damping
        e.luxtac.compliance_stiffness = float(_A.k)
        e.compliance_stiffness = float(_A.k)
        e.luxtac.compliant_damping = kd
        e.compliant_damping = kd
        e.desired_subassemblies = [sub]

        env = IsaacEnvWrapper(cfg)
        env.seed(0)
        env.reset()
        t_ = env.envs
        gym = t_.gym
        zero = np.zeros(env.action_space.shape[-1], dtype=np.float32)

        gel_id = {s: gym.find_actor_rigid_body_index(
            t_.env_ptrs[0], t_.actor_handles["franka"],
            f"luxtac_gel_{s}", gymapi.DOMAIN_ENV) for s in ("left", "right")}
        for s, i in gel_id.items():
            assert i > -1, f"luxtac_gel_{s} not found"

        # Four steps at the reset width before closing, so the arm has settled.
        for _ in range(4):
            env.step(zero)

        dof0 = (T - 2.0 * _A.indent + 2.0 * _A.proud_mm) / 2.0
        dof_mm = dof0

        def apply(width_mm, n):
            # The grip must be written to cfg_task.env.franka_close_gripper_width every
            # step: action[6] is not passed through to the finger drive in this task,
            # so without this the fingers never move and the pad reads a constant
            # self-contact that looks like a settled grasp.
            for _ in range(n):
                t_.cfg_task.env.franka_close_gripper_width = float(width_mm * 1e-3)
                env.step(zero)

        def centring():
            """Where the object sits between the jaws, in the gripper's own frame.

            Returns (across, along, grip) mm from the fingertip midpoint. `grip` is the
            finger-travel axis: if it is not ~0 the object is riding on one pad and the
            two pads cannot read the same depth.
            """
            from isaacgym.torch_utils import quat_rotate_inverse
            d = (t_.plug_pos[0] - t_.fingertip_midpoint_pos[0]).view(1, 3)
            local = quat_rotate_inverse(t_.fingertip_midpoint_quat[0].view(1, 4), d)
            return (local[0].cpu().numpy() * 1e3)

        def jaws():
            """Where both jaws and the object sit on the grip axis, in the hand frame.

            The object is placed at exactly xy = 0 between the jaws at reset
            (plug_pos_in_gripper_noise_xy is [0,0]), so any offset later developed
            while closing. This says which of the three causes it is: the two finger
            DOFs disagreeing, the two gels being mounted asymmetrically, or the object
            sliding between two symmetric jaws.
            """
            from isaacgym.torch_utils import quat_rotate_inverse
            gym.refresh_rigid_body_state_tensor(t_.sim)
            q = t_.fingertip_midpoint_quat[0].view(1, 4)
            o = t_.fingertip_midpoint_pos[0]
            loc = lambda p: quat_rotate_inverse(q, (p - o).view(1, 3))[0].cpu().numpy() * 1e3
            return (float(t_.dof_pos[0, 7]) * 1e3, float(t_.dof_pos[0, 8]) * 1e3,
                    loc(t_.body_pos[0, gel_id["left"]])[1],
                    loc(t_.body_pos[0, gel_id["right"]])[1],
                    loc(t_.plug_pos[0])[1])

        def read(side):
            pads = t_.get_luxtac_tensors_dict()[f"luxtac_{side}"]
            s = pads["grid"][0].cpu().numpy()
            raw = pads["depth_grid"][0].cpu().numpy() * 1e3      # depth at each sample point
            # mean depth per taxel (equal to the sample itself with one point per taxel)
            dmm = luxtac_viz.pool_subsamples(raw, *s.shape)
            gym.refresh_net_contact_force_tensor(t_.sim)
            f = float(torch.norm(t_.contact_force_net[0, gel_id[side]]))
            return (s, dmm, f, int(pads["num_saturated"]),
                    int(pads["num_contacting"]), raw)

        apply(dof_mm, _A.settle)

        cos = lambda a, b: float(a.ravel() @ b.ravel() /
                                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        def draw():
            s, dmm, F, nsat, ncon, raw = read(_A.pad)
            other = "right" if _A.pad == "left" else "left"
            s2, dmm2, F2, _, _, raw2 = read(other)
            # sim -> real frame: a plain transpose (as in the analysis scripts)
            sT, dT = s.T, dmm.T
            nT = peaknorm(sT)
            ind = (T + 2.0 * _A.proud_mm - 2.0 * dof_mm) / 2.0
            jw = jaws()
            act = float(t_.dof_pos[0, 7]) * 1e3

            ps = [panel("SDF depth mm (taxel mean)", dT, 1.9, colour,
                        f"mean-max {dT.max():.3f}  peak {raw.max():.3f}"),
                  panel("s  driver absolute", sT, 1.0, colour,
                        f"max {sT.max():.3f}  lit {int((sT > TAU).sum())}/16"),
                  panel("peak-norm  0-1", nT, 1.0, colour,
                        f"tau {TAU}  floor {SMAX}")]
            if real is not None:
                ps.append(panel("REAL  peak-norm", real, 1.0, colour, real_note))

            # Home the cursor and clear DOWNWARD at the end rather than clearing the
            # whole screen first: a full erase every frame flickers.
            head = "\x1b[H" if colour else ""
            out = [head +
                   f"\x1b[1m{sub}\x1b[0m   pad \x1b[1m{_A.pad}\x1b[0m (scored)   "
                   f"k={_A.k:.0f} N/m   gel proud {_A.proud_mm:.1f} mm"
                   if colour else f"{sub}   pad {_A.pad}   k={_A.k:.0f}",
                   "S01 top-left | top row = ROBOT side | bottom row = FLOOR side"
                   "  (all panels in the real frame)",
                   "",
                   f"DOF cmd {dof_mm:7.3f} mm   achieved L {jw[0]:7.3f} "
                   f"R {jw[1]:7.3f} mm (delta {jw[0] - jw[1]:+.4f})   "
                   f"target indent {ind:5.2f} mm",
                   f"grip axis, hand frame mm:  gel_L {jw[2]:+7.3f}   "
                   f"object {jw[4]:+7.3f}   gel_R {jw[3]:+7.3f}   | "
                   f"jaw midpoint {(jw[2] + jw[3]) / 2:+.3f}  "
                   f"object vs midpoint {jw[4] - (jw[2] + jw[3]) / 2:+.3f}",
                   f"gel force {F:7.3f} N   d_max {dmm.max():.3f} mm (taxel mean) / "
                   f"{raw.max():.3f} mm (peak sample)   "
                   f"contacting {ncon}/{raw.size}   saturated {nsat}",
                   ""]
            out += side_by_side(ps)

            # The other pad, same three readouts. Its pattern should be the MIRROR of
            # the scored pad, not a copy: panda_rightfinger is the left finger rotated
            # 180 deg about the finger axis, so the across-finger axis is reversed.
            s2T, d2T, n2T = s2.T, dmm2.T, peaknorm(s2.T)
            flip = cos(nT, peaknorm(s2[::-1, :].T))
            out += side_by_side([
                panel(f"{other} SDF depth mm", d2T, 1.9, colour,
                      f"mean-max {d2T.max():.3f}  peak {raw2.max():.3f}"),
                panel(f"{other} s absolute", s2T, 1.0, colour,
                      f"max {s2T.max():.3f}  lit {int((s2T > TAU).sum())}/16  "
                      f"F {F2:.2f} N"),
                panel(f"{other} peak-norm", n2T, 1.0, colour,
                      f"vs {_A.pad}: {cos(nT, n2T):.3f}  mirrored {flip:.3f}")])
            if real is not None:
                out.append(f"\x1b[1mcosine(peak-norm sim, peak-norm real) = "
                           f"{cos(nT, real):.4f}\x1b[0m" if colour else
                           f"cosine = {cos(nT, real):.4f}")
            out.append(f"  {other} pad is shown for reference only -- the alignment "
                       f"study scored {_A.pad}")
            off = centring()
            # z pins the frame: it reads 43.19 against the 43.2 mm tip_dist the assets
            # were built for, so x is across the finger and y is the finger-travel axis.
            out.append(f"  object centre vs fingertip midpoint, mm:  "
                       f"across {off[0]:+.2f}   \x1b[1mgrip axis {off[1]:+.2f}\x1b[0m   "
                       f"along {off[2]:+.2f}  (design 43.20)" if colour else
                       f"  object centre: across {off[0]:+.2f}  grip {off[1]:+.2f}  "
                       f"along {off[2]:+.2f}")
            if abs(off[1]) > 0.3:
                out.append(f"  ^ off-centre by {abs(off[1]):.2f} mm toward the "
                           f"{'right' if off[1] < 0 else 'left'} jaw, so the two pads "
                           f"cannot read the same depth at the same width")
            out.append("")
            out.append("  c/o or down/up = close/open 0.10 mm   C/O = 0.50 mm   "
                       "r = reset   p = snapshot   q = quit")
            # \x1b[K after every line: a line that SHRINKS between frames (the
            # off-centre warning appearing and going) would otherwise leave its tail
            # on screen. \x1b[J then clears whatever is below.
            sep = "\x1b[K\n" if colour else "\n"
            print(sep.join(out) + ("\x1b[K\x1b[J" if colour else ""), flush=True)
            return dict(s=s, depth=dmm, norm=nT, real=real if real is not None else np.zeros((4, 4)),
                        dof_cmd_mm=dof_mm, dof_act_mm=act, F=F, d_max=float(dmm.max()))

        def footprint(raw, name):
            """The sample-point contact, as align_luxtac_objects.py --diag prints it.

            Here for BOTH pads, because a taxel-level 'lit 16/16' does not say whether
            the contact is a stripe that the 3-sigma gate has spread across the pad or
            a genuine full-pad blob.
            """
            m = raw.max()
            print(f"\n  {name} pad, sample lattice (. none, - + # deepening), "
                  f"peak {m:.3f} mm:")
            if m <= 0:
                print("    (no contact)")
                return
            for r in range(raw.shape[0]):
                print("    " + "".join("." if v <= 0 else
                      ("#" if v > 0.75 * m else ("+" if v > 0.35 * m else "-"))
                      for v in raw[r]))

        if _A.trace:
            # Does the grasp settle, or does it ring? If it rings, the depth recorded
            # after a fixed number of settle steps is sampled at an arbitrary phase,
            # which would make "achieved depth at a commanded width" irreproducible.
            n = int(_A.trace)
            dof = np.zeros(n); dmx = np.zeros(n); frc = np.zeros(n); off = np.zeros(n)
            # Finger drive ringing (dof) and arm rocking (fingertip position, orientation).
            apos = np.zeros((n, 3)); aquat = np.zeros((n, 4))
            for i in range(n):
                apply(dof_mm, 1)
                ls, ld, lF, _, _, lraw = read("left")
                dof[i] = float(t_.dof_pos[0, 7]) * 1e3
                dmx[i] = float(lraw.max()); frc[i] = lF; off[i] = float(centring()[1])
                apos[i] = t_.fingertip_midpoint_pos[0].cpu().numpy() * 1e3
                aquat[i] = t_.fingertip_midpoint_quat[0].cpu().numpy()
            tail = slice(n // 2, n)          # judge on the second half only
            print(f"\n{sub}  held at DOF {dof_mm:.3f} mm for {n} steps "
                  f"(dt = {float(t_.cfg_base.sim.dt):.5f} s)")
            print(f"{'signal':>14} {'mean':>9} {'min':>9} {'max':>9} "
                  f"{'pk-pk':>9} {'pk-pk %':>8}")
            # wrist rocking, as the angle between each frame's orientation and the mean
            qm = aquat[tail].mean(axis=0); qm /= np.linalg.norm(qm)
            ang = np.degrees(2 * np.arccos(np.clip(np.abs(aquat @ qm), 0, 1)))
            for nm, v, u in (("finger DOF mm", dof, "mm"),
                             ("peak depth mm", dmx, "mm"),
                             ("gel force N", frc, "N"),
                             ("grip offset mm", off, "mm"),
                             ("wrist x mm", apos[:, 0], "mm"),
                             ("wrist y mm", apos[:, 1], "mm"),
                             ("wrist z mm", apos[:, 2], "mm"),
                             ("wrist tilt deg", ang, "deg")):
                w = v[tail]
                pk = w.max() - w.min()
                print(f"{nm:>14} {w.mean():9.4f} {w.min():9.4f} {w.max():9.4f} "
                      f"{pk:9.4f} {100 * pk / (abs(w.mean()) + 1e-9):8.1f}")
            # dominant period of the depth signal, by zero crossings of the residual
            w = dmx[tail] - dmx[tail].mean()
            zc = np.nonzero(np.diff(np.signbit(w)))[0]
            if len(zc) > 2:
                per = 2.0 * float(np.mean(np.diff(zc)))
                print(f"\n  depth rings with a period of {per:.1f} steps "
                      f"= {per * float(t_.cfg_base.sim.dt) * 1e3:.1f} ms "
                      f"({1.0 / (per * float(t_.cfg_base.sim.dt)):.1f} Hz)")
            else:
                print("\n  no sustained oscillation in the second half (settled)")
            np.savez(_A.snap, dof_mm=dof, peak_depth_mm=dmx, force_N=frc,
                     grip_off_mm=off, object=np.array(sub))
            print(f"saved -> {_A.snap}")
            return 0

        if _A.sweep is not None:
            # Both pads, same loading path as the scored runs.
            steps = _A.sweep if len(_A.sweep) else [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
            rec = {k: [] for k in
                   ("indent", "dof_mm", "dof_act_mm",
                    "left_s", "left_depth", "left_peak", "left_F",
                    "right_s", "right_depth", "right_peak", "right_F", "grip_off_mm")}
            print(f"{'tgt':>6} {'DOF':>7} {'act':>7} | "
                  f"{'L d_max':>8} {'L peak':>7} {'L F':>6} {'Llit':>5} | "
                  f"{'R d_max':>8} {'R peak':>7} {'R F':>6} {'Rlit':>5} | {'off':>6}")
            for ind in steps:
                w = (T - 2.0 * ind + 2.0 * _A.proud_mm) / 2.0
                apply(w, _A.settle)
                ls, ld, lF, _, _, lraw = read("left")
                rs, rd, rF, _, _, rraw = read("right")
                off = float(centring()[1])
                for k, v in (("indent", ind), ("dof_mm", w),
                             ("dof_act_mm", float(t_.dof_pos[0, 7]) * 1e3),
                             ("left_s", ls), ("left_depth", ld),
                             ("left_peak", float(lraw.max())), ("left_F", lF),
                             ("right_s", rs), ("right_depth", rd),
                             ("right_peak", float(rraw.max())), ("right_F", rF),
                             ("grip_off_mm", off)):
                    rec[k].append(v)
                print(f"{ind:6.2f} {w:7.3f} {rec['dof_act_mm'][-1]:7.3f} | "
                      f"{ld.max():8.3f} {lraw.max():7.3f} {lF:6.2f} "
                      f"{int((ls > TAU).sum()):3d}/16 | "
                      f"{rd.max():8.3f} {rraw.max():7.3f} {rF:6.2f} "
                      f"{int((rs > TAU).sum()):3d}/16 | {off:+6.2f}")
            np.savez_compressed(_A.snap,
                                **{k: np.array(v) for k, v in rec.items()},
                                object=np.array(sub))
            print(f"\nsaved -> {_A.snap}")
            return 0

        if _A.once:
            fr = draw()
            lraw = read("left")[5]
            rraw = read("right")[5]
            footprint(lraw, "left (scored)")
            footprint(rraw, "right")
            np.savez(_A.snap, **{k: np.asarray(v) for k, v in fr.items()},
                     object=np.array(sub))
            print(f"\nsaved -> {_A.snap}")
            return 0

        if not sys.stdin.isatty():
            raise SystemExit("stdin is not a terminal, so there would be no way to "
                             "close the gripper or quit. Run it from a shell, or "
                             "pass --once for a single frame.")

        with Keys() as keys:
            frame = draw()
            while True:
                for ch in keys.poll():
                    if ch == "q":
                        print("\nbye")
                        return 0
                    if ch == "c":
                        dof_mm -= 0.10
                    elif ch == "o":
                        dof_mm += 0.10
                    elif ch == "C":
                        dof_mm -= 0.50
                    elif ch == "O":
                        dof_mm += 0.50
                    elif ch == "r":
                        dof_mm = dof0
                    elif ch == "p":
                        np.savez(_A.snap,
                                 **{k: np.asarray(v) for k, v in frame.items()},
                                 object=np.array(sub))
                    dof_mm = float(np.clip(dof_mm, 0.2, 20.0))
                apply(dof_mm, _A.substep)
                frame = draw()

    return run()


if __name__ == "__main__":
    sys.exit(main() or 0)
