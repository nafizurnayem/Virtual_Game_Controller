"""Drive RunnerController with synthetic gestures on a fake clock.

Run from the project root:  python test_swipe.py
Useful for re-tuning the SWIPE_* thresholds without a camera in your hand.
"""
import sys

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import steering_wheel as sw

ASPECT = 640 / 480


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def tick(self, dt):
        self.t += dt


class FakeKeyboard:
    def __init__(self):
        self.taps = []

    def press(self, key):
        self.taps.append(str(key))

    def release(self, key):
        pass


clock = FakeClock()
kb = FakeKeyboard()
sw.time = clock
sw.keyboard = kb

DT = 1 / 30.0   # rebound per frame rate below


def feed(ctrl, points, pose=True):
    for p in points:
        ctrl.update(pose, p, ASPECT)
        clock.tick(DT)


def hold(x, y, n):
    return [(x, y)] * n


def move(x0, y0, dx, dy, n):
    return [(x0 + dx * i, y0 + dy * i) for i in range(1, n + 1)]


def check(name, expect, fps):
    ok = kb.taps == expect
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name}: taps={kb.taps} expected={expect}")
    return ok


def fresh():
    """New controller, settled and armed."""
    kb.taps.clear()
    ctrl = sw.RunnerController()
    feed(ctrl, hold(0.5, 0.5, int(0.5 / DT)))   # ~0.5s of stillness to arm
    return ctrl


def swipe_frames(fps):
    """A swipe covering ~0.18 frame-widths over ~0.1s, at whatever the frame rate is."""
    n = max(3, int(round(0.1 * fps)))
    return 0.135 / n, n


def run_suite(fps):
    global DT
    DT = 1.0 / fps
    print(f"\n--- {fps} fps ---")
    step, n = swipe_frames(fps)
    results = []

    for name, dx, dy, expect in [
        ("swipe right", step, 0, ["Key.right"]),
        ("swipe left", -step, 0, ["Key.left"]),
        ("swipe up", 0, -step, ["Key.up"]),
        ("swipe down", 0, step, ["Key.down"]),
    ]:
        ctrl = fresh()
        feed(ctrl, move(0.5, 0.5, dx, dy, n))
        results.append(check(name, expect, fps))

    # Slow drift must NOT fire
    ctrl = fresh()
    feed(ctrl, move(0.5, 0.5, 0.12 / fps, 0, int(fps)))   # 0.12/sec for 1s
    results.append(check("slow drift ignored", [], fps))

    # Ambiguous 45-degree diagonal must NOT fire
    ctrl = fresh()
    feed(ctrl, move(0.5, 0.5, step, step, n))
    results.append(check("diagonal ignored", [], fps))

    # Return stroke must not fire the opposite direction
    ctrl = fresh()
    feed(ctrl, move(0.5, 0.5, step, 0, n))
    feed(ctrl, move(0.5 + step * n, 0.5, -step, 0, n))
    results.append(check("return stroke ignored", ["Key.right"], fps))

    # Holding the pose still must not repeat-fire
    ctrl = fresh()
    feed(ctrl, move(0.5, 0.5, step, 0, n))
    feed(ctrl, hold(0.5 + step * n, 0.5, int(1.0 / DT)))
    results.append(check("no repeat while held", ["Key.right"], fps))

    # Two deliberate swipes with a pause -> two taps
    ctrl = fresh()
    feed(ctrl, move(0.5, 0.5, step, 0, n))
    feed(ctrl, hold(0.5, 0.5, int(0.4 / DT)))
    feed(ctrl, move(0.5, 0.5, step, 0, n))
    results.append(check("two swipes", ["Key.right", "Key.right"], fps))

    # Brief pose dropout mid-swipe must not abort the gesture
    ctrl = fresh()
    half = max(2, n // 2)
    feed(ctrl, move(0.5, 0.5, step, 0, half))
    feed(ctrl, [None, None], pose=False)          # 2 dropped frames (grace is 4)
    feed(ctrl, move(0.5 + step * half, 0.5, step, 0, n - half))
    results.append(check("survives 2-frame dropout", ["Key.right"], fps))

    return results


all_results = run_suite(30) + run_suite(60)
print(f"\n{sum(all_results)}/{len(all_results)} passed")
sys.exit(0 if all(all_results) else 1)
