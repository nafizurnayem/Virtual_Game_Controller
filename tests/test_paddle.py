"""Drive PaddleShifter with synthetic hand states on a fake clock.

Run from the project root:  python test_paddle.py
"""
import sys

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import steering_wheel as sw

FPS = 60.0
DT = 1.0 / FPS


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def tick(self, dt=DT):
        self.t += dt


class FakeKeyboard:
    def __init__(self):
        self.taps = []

    def press(self, key):
        self.taps.append(key)

    def release(self, key):
        pass


clock = FakeClock()
kb = FakeKeyboard()
sw.time = clock
sw.keyboard = kb

UP = sw.GEAR_UP_KEY
DOWN = sw.GEAR_DOWN_KEY


def feed(shifter, flicks, frames):
    for _ in range(frames):
        shifter.update(flicks)
        clock.tick()


def fresh():
    kb.taps.clear()
    s = sw.PaddleShifter()
    feed(s, {}, 5)          # idle, both paddles armed
    kb.taps.clear()
    return s


def check(name, got, expect):
    ok = got == expect
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got} expected {expect}")
    return ok


results = []

# Right flick -> upshift, once
s = fresh()
feed(s, {"Right": True}, 10)
results.append(check("right flick upshifts once", kb.taps, [UP]))
results.append(check("  gear advanced", s.gear, sw.GEAR_MIN + 1))

# Left flick -> downshift
s = fresh()
feed(s, {"Right": True}, 6)      # up to gear 2 first
feed(s, {}, 6)
kb.taps.clear()
feed(s, {"Left": True}, 10)
results.append(check("left flick downshifts", kb.taps, [DOWN]))
results.append(check("  gear back down", s.gear, sw.GEAR_MIN))

# Holding two fingers up must not rattle through gears
s = fresh()
feed(s, {"Right": True}, int(2.0 * FPS))
results.append(check("held flick fires once", kb.taps, [UP]))

# Release then flick again -> second shift
s = fresh()
feed(s, {"Right": True}, 6)
feed(s, {}, 6)
feed(s, {"Right": True}, 20)     # past the 0.22s cooldown
results.append(check("re-flick shifts again", kb.taps, [UP, UP]))

# A brief pose blip shorter than PADDLE_HOLD_FRAMES must not shift.
# This is the fist->open transition case: index+middle extend a moment before
# ring+pinky do, briefly looking exactly like the paddle pose.
s = fresh()
feed(s, {"Right": True}, sw.PADDLE_HOLD_FRAMES - 1)
feed(s, {}, 10)
results.append(check("short blip ignored (fist->open)", kb.taps, []))

# Both hands flicking at once -> one shift each, no crosstalk
s = fresh()
feed(s, {"Left": True, "Right": True}, 30)
results.append(check("both paddles fire once each", sorted(kb.taps), sorted([UP, DOWN])))

# Gear clamps at the top
s = fresh()
for _ in range(sw.GEAR_MAX + 4):
    feed(s, {"Right": True}, 6)
    feed(s, {}, 10)
results.append(check("gear clamps at max", s.gear, sw.GEAR_MAX))

# Gear clamps at the bottom
s = fresh()
for _ in range(3):
    feed(s, {"Left": True}, 6)
    feed(s, {}, 10)
results.append(check("gear clamps at min", s.gear, sw.GEAR_MIN))

# Cooldown blocks a too-fast repeat
s = fresh()
feed(s, {"Right": True}, sw.PADDLE_HOLD_FRAMES)
feed(s, {}, sw.PADDLE_RELEASE_FRAMES)
feed(s, {"Right": True}, sw.PADDLE_HOLD_FRAMES)   # well inside PADDLE_COOLDOWN
results.append(check("cooldown blocks fast repeat", kb.taps, [UP]))

# Swapped hands honour the config
sw.PADDLE_SWAP_HANDS = True
s = fresh()
feed(s, {"Right": True}, 10)
results.append(check("swap makes right downshift", kb.taps, [DOWN]))
sw.PADDLE_SWAP_HANDS = False

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
