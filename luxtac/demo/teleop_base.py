"""Keyboard state, gripper loop and observation helpers for demo_teleop_luxtac.py.

The listener writes module-level globals, so read them through the module
(teleop_base.done, teleop_base.quit_all), not by importing the names.

    LEFT / RIGHT arrow   close / open the gripper
    d                    finish the episode
    r                    retry (discard the episode)
    q                    quit
"""
import time
from pathlib import Path

import numpy as np
from pynput import keyboard

gripper_action = 0.0        # finger DOF target, m
gripper_min = 0.0
gripper_max = 0.05
gripper_step = 0.0005       # m per 10 ms tick while an arrow is held

left_pressed = False
right_pressed = False
done = False
retry = False
quit_all = False

# Pad images are written here and shown by luxtac_pad_viewer.py in its own process:
# cv2.imshow in the simulator's process deadlocks against the IsaacGym viewer.
_PNG_DIR = Path("../data/luxtac_live")
_PNG_DIR.mkdir(parents=True, exist_ok=True)


def on_press(key):
    global left_pressed, right_pressed, done, retry, quit_all
    if key == keyboard.Key.left:
        left_pressed = True
    elif key == keyboard.Key.right:
        right_pressed = True
    char = getattr(key, "char", None)
    if char == "d":
        done = True
        print(">>> done: saving episode")
    elif char == "r":
        retry = True
        print(">>> retry: discarding episode")
    elif char == "q":
        quit_all = True
        done = True
        print(">>> quit")


def on_release(key):
    global left_pressed, right_pressed
    if key == keyboard.Key.left:
        left_pressed = False
    elif key == keyboard.Key.right:
        right_pressed = False


def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()


def gripper_control_loop():
    global gripper_action
    while True:
        if left_pressed:
            gripper_action = max(gripper_min, gripper_action - gripper_step)
        if right_pressed:
            gripper_action = min(gripper_max, gripper_action + gripper_step)
        time.sleep(0.01)


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_obs_np(obs, key, first_env=True):
    """obs[key] as numpy (env 0 by default), or None if the key is absent."""
    if key not in obs:
        return None
    value = to_numpy(obs[key])
    if value is None:
        return None
    if first_env and value.ndim >= 1:
        return value[0]
    return value


def maybe_add(data, obs, key, out_key=None, first_env=True, dtype=None):
    """Copy obs[key] into data if the observation has it."""
    value = get_obs_np(obs, key, first_env=first_env)
    if value is None:
        return
    if dtype is not None:
        value = value.astype(dtype)
    data[out_key or key] = value
