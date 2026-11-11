"""SpaceMouse teleop on a LuxTac task, with live pads and episode recording.

    ./scripts/run_teleop_luxtac.sh --object peg_insert --indent 6.0   # peg-in-hole
    ./scripts/run_teleop_luxtac.sh --object usb --indent 3.0          # USB insertion
    ./scripts/run_teleop_luxtac.sh --object cylinder_deg45            # a test object

Episodes are appended to <out>/replay_buffer.zarr. Keys: d saves the episode, r
discards it, q quits.

Keyboard state and the gripper loop live in teleop_base; its listener writes
module-level globals, so they are read through the module (dt.done, dt.quit_all).
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import hydra
from omegaconf import DictConfig
import cv2

from luxtac.envs.env_wrapper import IsaacEnvWrapper      # isaacgym before torch

import teleop_base as dt
from luxtac import luxtac_viz

# Extent along the grip axis, mm; sets the starting gripper target.
THICK = {"cuboid": 20.0, "cylinder": 25.0, "ellipse": 20.0, "peg": 15.0,
         "usb": 12.0}
# Socket height on the table for the plug task, as in the TacSL USB config.
SOCKET_Z = {"usb": 0.025}

ap = argparse.ArgumentParser()
ap.add_argument("--object", default="peg_vert",
                help="peg_insert (peg-in-hole), usb (USB insertion), or an alignment "
                     "object <shape>_<orientation>, e.g. peg_vert, cylinder_deg45")
ap.add_argument("--indent", type=float, default=0.8,
                help="indentation the gripper starts at, mm. Arrow keys move it from "
                     "there. Real contact in the recorded episodes was 0.9-1.8 mm.")
ap.add_argument("--proud-mm", type=float, default=1.0)
ap.add_argument("--k", type=float, default=3100.0,
                help="gel compliance_stiffness. 3100 reaches 1.6 mm before PhysX loses "
                     "the contact; 6000 tops out at 1.0 mm but never fails.")
ap.add_argument("--urdf", default="luxtac_franka_gel45.urdf")
ap.add_argument("--sm-sign", default="++-",
                help="sign per action axis, e.g. '-+-'. Use if the arm moves opposite "
                     "to the puck on one or more axes.")
ap.add_argument("--sm-order", default="102",
                help="device axis feeding each action axis. Default '102' (device "
                     "y,x,z). Use '012' if left/right and forward/back are swapped.")
ap.add_argument("--sm-raw", action="store_true",
                help="print the raw device axes in the heartbeat, to read the mapping")
ap.add_argument("--grip-step", type=float, default=None,
                help="mm per arrow-key tick (default 0.5 from teleop_base). "
                     "The key repeat is 100 Hz, so 0.5 moves fast; try 0.1 for fine work.")
ap.add_argument("--no-record", action="store_true",
                help="live only; do not save episodes")
ap.add_argument("--selftest", type=int, default=0, metavar="N",
                help="no window: record one N-step episode, save it and exit")
ap.add_argument("--finger-friction", type=float, default=None,
                help="override the finger joints' friction (the robot file has 5.0)")
ap.add_argument("--finger-damping", type=float, default=100.0,
                help="passive damping on the finger joints, N s/m. Without it the joint "
                     "friction in the robot file makes both fingers jump ~1.2 mm sideways "
                     "every step. 0 restores that behaviour.")
ap.add_argument("--out", default=None,
                help="episode folder (default <repo>/data/teleop/<object>)")
_A, _REST = ap.parse_known_args()
sys.argv = [sys.argv[0]] + _REST          # hydra parses argv too

CAMS = ("front", "wrist")                  # stored per step in the replay buffer
STATE_KEYS = ("luxtac_left", "luxtac_right", "luxtac_depth_left", "luxtac_depth_right",
              "ee_pos", "ee_quat", "plug_pos", "plug_quat", "socket_pos", "socket_quat",
              "dof_pos", "dof_vel")

SHAPE = _A.object.rsplit("_", 1)[0]
if SHAPE not in THICK:
    raise SystemExit(f"--object must start with one of {sorted(THICK)}, got {_A.object!r}")


# --------------------------------------------------------------------- SpaceMouse
class SpaceMouseHID:
    """Minimal 3Dconnexion reader over raw HID.

    Compact firmware sends report 1 = translation and report 2 = rotation, each three
    little-endian int16 at roughly +-350 full scale; some packs all six into report 1.
    Both layouts are handled.

    Reads are non-blocking at ~250 Hz while the device reports at 60-125 Hz, so most
    polls return nothing. The last value is HELD rather than zeroed -- zeroing on an
    empty read wiped the motion between frames and made it stutter. The device sends
    an explicit all-zero report when the puck is released, which is what stops it.
    """

    VENDOR, PRODUCT, FULL_SCALE, DEADZONE = 0x256F, 0xC635, 350.0, 0.03
    ORDER = (1, 0, 2)                      # device (x,y,z) -> action (y,x,z)
    SIGN = np.array([1.0, 1.0, 1.0])

    def __init__(self, pos_sensitivity=1.0, rot_sensitivity=1.0):
        import hid
        self.dev = hid.device()
        self.dev.open(self.VENDOR, self.PRODUCT)
        self.dev.set_nonblocking(True)
        self.pos_s, self.rot_s = pos_sensitivity, rot_sensitivity
        self.trans = np.zeros(3)
        self.rot = np.zeros(3)
        self._stop = False
        print(f"spacemouse  {self.dev.get_product_string()}  "
              f"sign {''.join('+' if v > 0 else '-' for v in self.SIGN)}  "
              f"order {''.join(str(i) for i in self.ORDER)}")

    @staticmethod
    def _axes(buf, n):
        return np.array([int.from_bytes(buf[2 * i:2 * i + 2], "little", signed=True)
                         for i in range(n)], dtype=float)

    def start_control(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            try:
                r = self.dev.read(64)
            except Exception:
                r = None
            if r:
                rid, body = r[0], bytes(r[1:])
                if rid == 1 and len(body) >= 12:
                    a = self._axes(body, 6)
                    self.trans, self.rot = a[:3], a[3:]
                elif rid == 1:
                    self.trans = self._axes(body, 3)
                elif rid == 2:
                    self.rot = self._axes(body, 3)
            time.sleep(0.004)

    def read(self):
        """-> (d_pos[3], d_rpy[3]) in [-1, 1]."""
        t = np.clip(self.trans / self.FULL_SCALE, -1, 1) * self.pos_s
        r = np.clip(self.rot / self.FULL_SCALE, -1, 1) * self.rot_s
        t[np.abs(t) < self.DEADZONE] = 0.0     # the puck never rests at exactly 0
        r[np.abs(r) < self.DEADZONE] = 0.0
        d_pos = np.array([t[i] for i in self.ORDER]) * self.SIGN
        d_rpy = np.array([r[i] for i in self.ORDER]) * self.SIGN
        return d_pos, d_rpy

    def raw(self):
        return self.trans.copy(), self.rot.copy()


def get_action(sm):
    """6-DoF deltas from the SpaceMouse plus the gripper target."""
    d_pos, d_rpy = sm.read()
    return np.concatenate([d_pos, d_rpy, [dt.gripper_action]]).astype(np.float32)


def zarr_episodes(path):
    """Number of episodes already in the replay buffer at `path`."""
    import zarr
    if not Path(path).exists():
        return 0
    return len(zarr.open(str(path), mode="r")["meta/episode_ends"])


def zarr_append_episode(path, arrays, fps):
    """Append one episode to a zarr replay buffer.

    data/<key> holds every step of every episode back to back, and meta/episode_ends
    holds the index one past the last step of each episode.
    """
    import zarr
    from numcodecs import Blosc
    root = zarr.open(str(path), mode="a")
    data, meta = root.require_group("data"), root.require_group("meta")
    comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    n = len(next(iter(arrays.values())))
    for key, arr in arrays.items():
        if key in data:
            data[key].append(arr)
        else:
            step = max(1, min(n, (1 << 21) // max(1, arr[0].nbytes)))      # ~2 MB chunks
            data.create_dataset(key, data=arr, chunks=(step,) + arr.shape[1:],
                                compressor=comp, dtype=arr.dtype)
    if "episode_ends" in meta:
        meta["episode_ends"].append([int(meta["episode_ends"][-1]) + n])
    else:
        meta.create_dataset("episode_ends", data=np.array([n], dtype=np.int64), chunks=(1024,))
    root.attrs["fps"] = list(root.attrs.get("fps", [])) + [float(fps)]


# ------------------------------------------------------------------------ display
TAU, S_MAX_FIXED = 0.02, 0.36       # the same two constants every alignment test used
def blue_lut(n=256):
    """(n, 3) uint8 ramp, WHITE at no contact -> DARK BLUE at full contact.

    Built here rather than added to luxtac_viz so the other teleops keep the
    black-to-white ramp they were written against. Linear in each channel, so the
    shade still reads as proportional to s: red and green fall to zero while blue
    only falls to 110, which keeps the deepest cells blue rather than black.
    """
    t = np.linspace(0.0, 1.0, n)[:, None]          # 0 = no contact, 1 = full
    white = np.array([[255.0, 255.0, 255.0]])
    dark_blue = np.array([[10.0, 35.0, 110.0]])
    return (white + (dark_blue - white) * t).round().astype(np.uint8)


_LUT = blue_lut()                           # WHITE = no contact, DARK BLUE = full
_CELL, _GAP = 110, 3                        # px per taxel, px between taxels


def peak_normalise(grid):
    """VT-Refine's per-frame peak rule, identical to the comparisons.

    s / max(s) when the frame's peak clears TAU, so exactly one taxel reads 1.0 and
    renders fully black; otherwise s / S_MAX_FIXED, so an untouched frame stays
    untouched instead of having its noise normalised up to full scale.
    """
    g = np.clip(np.asarray(grid, dtype=np.float64), 0.0, None)
    peak = g.max()
    return g / (peak if peak >= TAU else S_MAX_FIXED)


def show_pad(obs):
    """Write BOTH pads as a bare 4x4 taxel grid, each peak-normalised per frame.

    The viewer polls one PNG per pad, so both must be written or that window never
    appears. Each pad is normalised by ITS OWN peak, matching how the real episodes
    were processed -- the driver reads one pad at a time.
    """
    out = {}
    for key in ("luxtac_left", "luxtac_right"):
        grid = dt.get_obs_np(obs, key)
        if grid is None:
            continue
        g = luxtac_viz.orient_for_display(
            peak_normalise(np.asarray(grid).reshape(4, 4)))
        rows, cols = g.shape
        img = np.full((rows * _CELL + (rows + 1) * _GAP,
                       cols * _CELL + (cols + 1) * _GAP, 3), 128, np.uint8)
        rgb = luxtac_viz.normalized_to_rgb(g, _LUT)
        for r in range(rows):
            for c in range(cols):
                y, x = _GAP + r * (_CELL + _GAP), _GAP + c * (_CELL + _GAP)
                img[y:y + _CELL, x:x + _CELL] = rgb[r, c]
        tmp = dt._PNG_DIR / f".{key}.tmp.png"
        cv2.imwrite(str(tmp), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        tmp.rename(dt._PNG_DIR / f"{key}.png")
        out[key] = g
    return out


# --------------------------------------------------------------------------- main
@hydra.main(version_base="1.1", config_path="../config",
            config_name="isaacgym_config_luxtac")
def main(cfg: DictConfig):
    cfg.headless = bool(_A.selftest)
    cfg.num_envs = 1
    e = cfg.task.env
    e.asset_info_filename = "luxtac_asset_info_objects.yaml"
    e.desired_subassemblies = [f"lux_{_A.object}"]
    if SHAPE in SOCKET_Z:
        cfg.task.randomize.socket_pos_xyz_initial = [0.5, 0.0, SOCKET_Z[SHAPE]]
    e.franka_urdf_file = _A.urdf
    e.luxtac.num_cols = 4
    # obsDims is declared SEPARATELY from env.luxtac and the task allocates the
    # observation buffers from it, so changing num_cols alone throws
    #   "expanded size of the tensor (8) must match the existing size (4)"
    # out of luxtac_task_insertion.compute_observations.
    nr, ss = int(e.luxtac.num_rows), int(e.luxtac.subsamples)
    for side in ("left", "right"):
        e.obsDims[f"luxtac_{side}"] = [nr, 4]
        e.obsDims[f"luxtac_depth_{side}"] = [nr * ss, 4 * ss]
    e.luxtac.response = "photodiode"
    e.luxtac.compliance_stiffness = float(_A.k)
    e.compliance_stiffness = float(_A.k)
    kd = float(2 * np.sqrt(_A.k * 0.03))          # contact damping
    e.luxtac.compliant_damping = kd
    e.compliant_damping = kd

    # Gripper target that puts the gel --indent mm into the object.
    #   gap = 2*DOF - 2*proud  and  gap = thickness - 2*indent
    dof_m = (THICK[SHAPE] - 2 * _A.indent + 2 * _A.proud_mm) / 2.0 * 1e-3
    e.franka_close_gripper_width = float(dof_m)

    print(f"object      lux_{_A.object}  ({THICK[SHAPE]:.0f} mm across the grip axis)")
    print(f"gripper     DOF {dof_m*1e3:.2f} mm -> gap {(2*dof_m*1e3 - 2*_A.proud_mm):.2f} mm"
          f" -> {_A.indent:.2f} mm indent")
    print(f"gel         k={_A.k:.0f}  kd={kd:.1f}  urdf={_A.urdf}  grid 4x4")

    if _A.grip_step is not None:
        dt.gripper_step = float(_A.grip_step) * 1e-3
    print(f"keys        LEFT = close, RIGHT = open, {dt.gripper_step*1e3:.2f} mm per tick"
          f"  (range {dt.gripper_min*1e3:.0f}-{dt.gripper_max*1e3:.0f} mm DOF)"
          f"   d = end, q = quit")
    if not _A.selftest:                         # the listener is global: typing in any
        dt.start_keyboard_listener()            # window would end the test
    threading.Thread(target=dt.gripper_control_loop, daemon=True).start()

    env = IsaacEnvWrapper(cfg)
    SpaceMouseHID.SIGN = np.array([1.0 if c == "+" else -1.0 for c in _A.sm_sign])
    SpaceMouseHID.ORDER = tuple(int(c) for c in _A.sm_order)
    try:
        space_mouse = SpaceMouseHID()
        space_mouse.start_control()
    except Exception as exc:                 # keep the pad display usable regardless
        print(f"[warn] no SpaceMouse ({exc}); arm held still, gripper keys still work")
        space_mouse = None
    print("Both pads, each peak-normalised per frame: one taxel per pad is always "
          "fully black.\n")

    def set_finger_props():
        if _A.finger_friction is None and _A.finger_damping is None:
            return
        t = env.envs
        for env_ptr in t.env_ptrs:
            props = t.gym.get_actor_dof_properties(env_ptr, t.actor_handles["franka"])
            if _A.finger_friction is not None:
                props["friction"][7:9] = _A.finger_friction
            if _A.finger_damping is not None:
                props["damping"][7:9] = _A.finger_damping
            t.gym.set_actor_dof_properties(env_ptr, t.actor_handles["franka"], props)

    record = not _A.no_record
    out_dir = Path(_A.out) if _A.out else Path(__file__).resolve().parents[2] / "data" / "teleop" / _A.object
    buffer = out_dir / "replay_buffer.zarr"
    if record:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"recording   {out_dir}   d = save + next, r = discard + retry, q = quit "
              f"(the unfinished episode is dropped)")

    while not dt.quit_all:
        ep = zarr_episodes(buffer) if record else 0
        env.seed(ep)
        obs = env.reset()
        set_finger_props()
        dt.gripper_action = float(dof_m)
        dt.done = False
        dt.retry = False
        rows, t0 = [], time.time()
        raw = {c: [] for c in CAMS}
        print(f"episode {ep} started", flush=True)

        step, hb = 0, time.time()
        while not (dt.quit_all or dt.done or dt.retry):
            actions = (get_action(space_mouse) if space_mouse is not None
                       else np.concatenate([np.zeros(6), [dt.gripper_action]]).astype(np.float32))
            # The grip has to be written to cfg_task.env.franka_close_gripper_width, which
            # is what this task reads; action[6] is NOT passed through to the finger drive
            # here. Without this the fingers never move and the pad reads a constant
            # self-contact that looks like a settled grasp.
            env.envs.cfg_task.env.franka_close_gripper_width = float(dt.gripper_action)
            if record:
                row = {"action": actions.astype(np.float32), "t": np.float64(time.time() - t0)}
                for k in STATE_KEYS:
                    dt.maybe_add(row, obs, k, dtype=np.float32)
                row["finger_dof"] = env.envs.dof_pos[0, 7:9].detach().cpu().numpy().astype(np.float32)
                row["hand_pos"] = env.envs.hand_pos[0].detach().cpu().numpy().astype(np.float32)
                rows.append(row)
                for c in CAMS:
                    im = dt.get_obs_np(obs, c)
                    if im is not None:
                        raw[c].append((np.clip(im[..., :3], 0, 1) * 255).astype(np.uint8))
            obs, _, _, _ = env.step(actions)
            show_pad(obs)
            step += 1
            if _A.selftest and step >= _A.selftest:
                dt.done = True

            if time.time() - hb >= 2.0:
                pad = dt.get_obs_np(obs, "luxtac_left")
                dep = dt.get_obs_np(obs, "luxtac_depth_left")
                dmax = f"{float(np.max(dep)) * 1e3:.3f} mm" if dep is not None else "--"
                g = np.asarray(pad).reshape(4, 4) if pad is not None else None
                lit = int((g > TAU).sum()) if g is not None else -1
                dof_act = float(env.envs.dof_pos[0, 7]) * 1e3      # what the finger DID
                print(f"[hb] step {step:4d} cmd {dt.gripper_action * 1e3:5.2f} -> act {dof_act:5.2f} mm"
                      f" | gap {(2 * dof_act - 2 * _A.proud_mm):6.2f} mm"
                      f" | depth {dmax:>9}"
                      f" | mean s {float(g.mean()) if g is not None else -1:.4f}"
                      f"  peak {float(g.max()) if g is not None else -1:.3f}"
                      f"  lit {lit:2d}/16"
                      f" | right lit {int((np.asarray(dt.get_obs_np(obs, 'luxtac_right')) > TAU).sum())}/16", flush=True)
                if _A.sm_raw and space_mouse is not None:
                    tr, ro = space_mouse.raw()
                    print(f"      spacemouse raw  trans {np.round(tr).astype(int).tolist()}"
                          f"  rot {np.round(ro).astype(int).tolist()}", flush=True)
                hb = time.time()

        if not record:
            if dt.done:
                break
            continue
        if dt.done and rows:
            fps = max(1.0, len(rows) / max(rows[-1]["t"], 1e-6))
            arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
            arrays["step_idx"] = np.arange(len(rows), dtype=np.int32)
            for c, frames in raw.items():          # camera observations, float in [0, 1]
                if frames:
                    arrays[c] = np.stack(frames).astype(np.float32) / 255.0
            zarr_append_episode(buffer, arrays, fps)
            print(f"saved episode {ep}: {len(rows)} steps, {fps:.1f} fps, "
                  f"keys {sorted(arrays)} -> {buffer}", flush=True)
        elif dt.retry:
            print(f"discarded episode {ep}", flush=True)
        if _A.selftest:
            break

    cv2.destroyAllWindows()
    print("done.")


if __name__ == "__main__":
    main()
