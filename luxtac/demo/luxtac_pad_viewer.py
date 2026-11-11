"""Live LuxTac pad viewer. Runs as its own process.

The teleop writes one PNG per pad each step; this script polls and shows them.
(cv2.imshow in the simulator's process deadlocks against the IsaacGym viewer.)

    python luxtac_pad_viewer.py [png_dir]

Windows: 'luxtac_left' and 'luxtac_right'. Press q or ESC in either to close.
"""

import sys
import time
from pathlib import Path

import cv2

PNG_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../data/luxtac_live")
KEYS = ("luxtac_left", "luxtac_right")

last_mtime = {k: 0.0 for k in KEYS}
shown = False

while True:
    for k in KEYS:
        p = PNG_DIR / f"{k}.png"
        try:
            m = p.stat().st_mtime
        except FileNotFoundError:
            continue
        if m > last_mtime[k]:
            img = cv2.imread(str(p))
            if img is not None:
                cv2.imshow(k, img)
                last_mtime[k] = m
                shown = True
    # waitKey pumps the window event loop; poll at about 20 Hz
    if shown and cv2.waitKey(50) in (ord("q"), 27):
        break
    if not shown:
        time.sleep(0.2)

cv2.destroyAllWindows()
